import os
from copy import deepcopy
from tqdm import tqdm, trange
from glob import glob
from omegaconf import OmegaConf

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from einops import rearrange

from methods.base import BaseLearner
from ldm import DDIMSampler, LatentDiffusion
from utils import (
    IncrementalNet, 
    SupConLoss, 
    GenDataset, 
    DatasetSplit, 
    DataIter,
    kd_loss,
    partition_data, 
    average_weights, 
    setup_seed
)


train_transform = {
    "cifar100": transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]),
    "tiny_imagenet": transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}


class TaskSynImageDataset(Dataset):
    def __init__(self, root, task_id, size_per_cls, transform=None):
        self.root = os.path.join(root, "task_{}".format(task_id))
        img_paths = glob(os.path.join(self.root, "*", "*.jpg"))
        img_paths = [tp for tp in img_paths 
                     if int(tp.split("-")[-1].rstrip(".jpg")) < size_per_cls]
        self.images, self.labels = [], []
        for tp in img_paths:
            self.labels.append(int(tp.split("/")[-2]))
            with Image.open(tp) as tmp_img:
                self.images.append(np.array(tmp_img))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return idx, img, label

    def __len__(self):
        return len(self.images)


class OURS(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)
        self.generator_init()
        if args["w_scl"] > 0:
            self.scl_criterion = SupConLoss()
        else:
            self.scl_criterion = None
        self.need_syn_imgs = args['syn_image_path'] is None
        if args['syn_image_path'] is not None:
            self.syn_imgs_dir = args['syn_image_path']
        else:
            self.syn_imgs_dir = os.path.join(args['save_dir'], "syn_imgs")

    def generator_init(self):
        self.config = OmegaConf.load(self.args['config'])
        self.config.model.params.ckpt_path = self.args['ldm_ckpt']
        self.config['model']["params"]['personalization_config']["params"]['num_classes'] = \
            self.args['increment']
        self._generator = LatentDiffusion(**self.config['model']["params"])
        self._generator.load_state_dict(
            torch.load(self.args['ldm_ckpt'], map_location="cpu")["state_dict"], 
            strict=False)
        self.generator_init_embedding = deepcopy(self._generator.embedding_manager.state_dict())
        self._generator.learning_rate =  self.config.data.params.batch_size * self.config.model.base_learning_rate
        print("Setting learning rate to {:.2e} =  {} (batchsize) * {:.2e} (base_lr)".format(
                self._generator.learning_rate, 
                self.config.data.params.batch_size, 
                self.config.model.base_learning_rate))
                
    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
    
    def incremental_train(self, data_manager):
        setup_seed(self.seed)
        self._cur_task += 1
        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self.init_dataloader(data_manager)
        if self.need_syn_imgs:
            inv_text_embeds = self._class_inversion() # class inversion for current class
            self._synthesis_imgs(inv_text_embeds)    # data synthesize for current class
        self._init_syn_dataloader()
        self._fl_train()

    def _fl_train(self):
        self._network.cuda()
        prog_bar = tqdm(range(self.args["com_round"]))
        for _, com in enumerate(prog_bar):
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for _, idx in enumerate(idxs_users):
                if self._cur_task == 0:
                    w = self._local_update(deepcopy(self._network), self.local_cur_loaders[idx])
                else:
                    w = self._local_finetune(self._old_network, deepcopy(self._network), 
                                             self.local_cur_loaders[idx])
                local_weights.append(deepcopy(w))
            # update global weights
            global_weights = average_weights(local_weights, dp_si=self.args["classifer_dp"])
            self._network.load_state_dict(global_weights)
            # test
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info=("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                self._cur_task, com + 1, self.args["com_round"], test_acc,))
            prog_bar.set_description(info)

    def _local_update_g(self, generator, gen_data_loader):
        generator.train()
        generator = generator.cuda()
        optim = generator.configure_optimizers()
        for _ in range(self.args["g_local_train_steps"]):
            batch = gen_data_loader.next()
            batch["image"] = batch["image"].cuda()
            loss, _ = generator.shared_step(batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
        return generator.embedding_manager.state_dict()

    def _local_update(self, model, cur_data_loader):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.args["local_ep"]):
            for _, (_, images, labels) in enumerate(cur_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)
                loss = F.cross_entropy(output["logits"], labels)
                if self.args["w_scl"] > 0:
                    loss_scl = self.scl_criterion(output['scl_emb'], labels)
                    loss = loss + self.args["w_scl"] * loss_scl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()
    
    def _local_finetune(self, teacher, model, cur_data_loader):
        model.train()
        teacher.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.args["local_ep"]):
            for _, images, labels in cur_data_loader:
                _, pre_imgs, pre_labels = self.pre_syn_data_iter.next()
                images, labels, pre_imgs, pre_labels = images.cuda(), labels.cuda(), pre_imgs.cuda(), pre_labels.cuda()
                fake_targets = labels - self._known_classes
                output = model(images)
                loss = F.cross_entropy(output["logits"][:, self._known_classes:], fake_targets)
                if self.args["w_ce_pre"] > 0:
                    s_out = model(pre_imgs)
                    loss_ce_pre = F.cross_entropy(s_out["logits"][:, :self._known_classes], pre_labels)
                    loss = loss + self.args["w_ce_pre"] * loss_ce_pre
                if self.args["w_kd"] > 0:
                    with torch.no_grad():
                        t_out = teacher(pre_imgs.detach())["logits"]
                    loss_kd = kd_loss(
                        s_out["logits"][:, : self._known_classes],   # logits on previous tasks
                        t_out.detach(),
                        2)
                    loss = loss + self.args["w_kd"] * loss_kd 
                if self.args["w_scl"] > 0:
                    loss_scl = self.scl_criterion(output['scl_emb'], labels)
                    loss = loss + self.args["w_scl"] * loss_scl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def _class_inversion(self):
        self._generator.cuda()
        self._generator.embedding_manager.load_state_dict(self.generator_init_embedding)
        prog_bar = tqdm(range(self.args["com_round_gen"]), desc='Class Inversion')
        for _ in prog_bar:
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                if self.gen_data_iters[idx] is None:
                    continue
                w = self._local_update_g(deepcopy(self._generator),
                                        self.gen_data_iters[idx])
                local_weights.append(deepcopy(w))
            global_weights = average_weights(local_weights, self.args['g_sigma'])
            self._generator.embedding_manager.load_state_dict(global_weights)
        inv_text_embeds = deepcopy(self._generator.embedding_manager.string_to_param_dict)
        if self.args["save_cls_embeds"]:
            cls_embeds_path = os.path.join(self.save_dir, 
                'cls_embeds_ckpt', '%d-%d_embedding_manager.pt' % (self.min_class_id, self.max_class_id))
            os.makedirs(os.path.dirname(cls_embeds_path), exist_ok=True)
            torch.save(self._generator.embedding_manager.state_dict(), cls_embeds_path)
        return inv_text_embeds

    def _synthesis_imgs(self, inv_text_embeds):
        self._generator.embedding_manager.string_to_param_dict = inv_text_embeds
        sampler = DDIMSampler(self._generator)
        # outdir = os.path.join('gen_result_tmp', str(idx))
        outdir = os.path.join(self.syn_imgs_dir, "task_{}".format(self._cur_task))
        os.makedirs(outdir, exist_ok=True)
        prompt = "a photo of *"
        n_samples = 40
        scale = 10.0
        ddim_steps = 50
        ddim_eta = 0.0
        H = 256
        W = 256
        with torch.no_grad():
            for tmp_cls in self.all_classes:
                base_count = 0
                with self._generator.ema_scope():
                    uc = None
                    tmp_cls_tensor = torch.LongTensor([tmp_cls - self.min_class_id,] * n_samples)
                    if scale != 1.0:
                        uc = self._generator.get_learned_conditioning(n_samples * [""], tmp_cls_tensor)
                    for _ in trange(self.args['n_iter'], desc="Sampling"):
                        c = self._generator.get_learned_conditioning(n_samples * [prompt], tmp_cls_tensor)
                        shape = [4, H//8, W//8]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                        conditioning=c,
                                                        batch_size=n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=uc,
                                                        eta=ddim_eta)
                        x_samples_ddim = self._generator.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            if not os.path.exists(os.path.join(outdir, str(tmp_cls))):
                                os.makedirs(os.path.join(outdir, str(tmp_cls)))
                            Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(outdir, str(tmp_cls), f"{tmp_cls}-{base_count}.jpg"))
                            base_count += 1

    def _init_syn_dataloader(self):
        cur_syn_dataset = TaskSynImageDataset(
            self.syn_imgs_dir, self._cur_task, self.args['cur_size'],
            transform=train_transform[self.args['dataset']])
        self.local_cur_loaders = []
        for idx in range(self.args["num_users"]):
            self.local_cur_loaders.append(
                DataLoader(ConcatDataset([self.local_train_dataset[idx], cur_syn_dataset]), 
                           batch_size=self.args["local_bs"], shuffle=True, num_workers=4))
        if self._cur_task > 0: 
            pre_syn_dataset = ConcatDataset(
                [TaskSynImageDataset(self.syn_imgs_dir, i, self.args['pre_size'],
                                     transform=train_transform[self.args['dataset']]) 
                 for i in range(self._cur_task)])
            pre_syn_data_loader = DataLoader(
                pre_syn_dataset, batch_size=128, shuffle=True,
                num_workers=4, pin_memory=True)
            self.pre_syn_data_iter = DataIter(pre_syn_data_loader)

    def init_dataloader(self, data_manager):
        train_dataset = data_manager.get_dataset(   #* get the data for one task
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        gen_dataset = GenDataset(
            input_np_array=train_dataset.images,
            class_ids=train_dataset.labels
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )
        user_groups, label_ratio = partition_data(train_dataset.labels, 
            beta=self.args["beta"], n_parties=self.args["num_users"], return_ratio=True)
        self.label_ratio = label_ratio if self.label_ratio is None \
            else np.concatenate((self.label_ratio, label_ratio), axis=1)
        self.local_train_dataset = []
        for idx in range(len(user_groups)):
            self.local_train_dataset.append(DatasetSplit(train_dataset, user_groups[idx]))
        test_idx_map = [[] for _ in range(self.args["num_users"])]
        for k in range(self._total_classes):
            idx_k = np.where(test_dataset.labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = (np.cumsum(self.label_ratio[:, k]) * len(idx_k)).astype(int)[:-1]
            test_idx_map = [idx_j + idx.tolist() for idx_j, idx in zip(test_idx_map, np.split(idx_k, proportions))]
        self.local_test_loader = []
        for idx in range(self.args["num_users"]):
            self.local_test_loader.append(DataLoader(DatasetSplit(test_dataset, test_idx_map[idx]), 
                batch_size=self.args["local_bs"], shuffle=True, num_workers=4))
        self.all_classes = np.unique(train_dataset.labels)
        self.min_class_id, self.max_class_id = np.min(self.all_classes), np.max(self.all_classes)
        self.gen_data_iters = []
        for idx in range(self.args["num_users"]):
            local_gen_dataset = deepcopy(gen_dataset)
            local_gen_dataset.set_subset(user_groups[idx])
            local_gen_dataset.min_class_id = self.min_class_id
            if len(local_gen_dataset) == 0:
                self.gen_data_iters.append(None)
            else:
                local_gen_loader = DataLoader(local_gen_dataset, batch_size=self.args['g_local_bs'],
                        num_workers=4, shuffle=True)
                self.gen_data_iters.append(DataIter(local_gen_loader))
