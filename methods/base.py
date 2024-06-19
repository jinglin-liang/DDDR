import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from utils import (
    tensor2numpy, 
    grouped_accuracy, 
    cal_forget_measure, 
    partition_data, 
    DatasetSplit
)


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None

        self.args = args
        self.each_task = args["increment"]
        self.seed = args["seed"]
        self.tasks = args["tasks"]
        self.save_dir = args["save_dir"]
        self.dataset_name = args["dataset"]
        self.nums = args["nums"]

        self.label_ratio = None
        # metrics
        self.per_cls_acc_dict = {i: [] for i in range(self.args["num_users"])}
        self.forgetting_measure_dict = {i: [] for i in range(self.args["num_users"])}
        self.group_acc_dict = {i: [] for i in range(self.args["num_users"])}
        self.per_cls_acc_dict["global"] = []
        self.forgetting_measure_dict["global"] = []
        self.group_acc_dict["global"] = []
        self.local_acc_mean_std = []
        self.local_fm_mean_std = []

    def after_task(self):
        pass

    def incremental_train(self):
        pass

    def _fl_train(self):
        pass

    def eval_task(self):
        global_cls_acc, global_preds, global_targets = self.per_cls_acc(self.test_loader, self._network)
        self.per_cls_acc_dict["global"].append(global_cls_acc)
        self.group_acc_dict["global"].append(grouped_accuracy(global_preds, global_targets, 
                                                              self._known_classes, increment=self.each_task))
        for i in range(self.args["num_users"]):
            tmp_cls_acc, tmp_preds, tmp_targets = self.per_cls_acc(self.local_test_loader[i], self._network)
            self.per_cls_acc_dict[i].append(tmp_cls_acc)
            self.group_acc_dict[i].append(grouped_accuracy(tmp_preds, tmp_targets, self._known_classes, increment=self.each_task))
        local_acc = np.array([self.group_acc_dict[i][-1]['total'] for i in range(self.args["num_users"])])
        self.local_acc_mean_std.append({"mean": local_acc.mean(), "std": local_acc.std()})
        if self._cur_task > 0:
            self.forgetting_measure_dict["global"].append(cal_forget_measure(self.per_cls_acc_dict["global"]))
            for i in range(self.args["num_users"]):
                self.forgetting_measure_dict[i].append(cal_forget_measure(self.per_cls_acc_dict[i]))
            local_fm = np.array([self.forgetting_measure_dict[i][-1] for i in range(self.args["num_users"])])
            self.local_fm_mean_std.append({"mean": local_fm.mean(), "std": local_fm.std()})

    def log_metrics(self, print_to_console=True, save_to_file=True):
        log_txt_pth = os.path.join(self.save_dir, "log.txt")
        os.makedirs(os.path.dirname(log_txt_pth), exist_ok=True)
        log_str  = "\n==========================task{}==========================\n".format(self._cur_task)
        log_str += "global_test:\n"
        log_str += "\tACC curve: {}\n".format([tmp_dict['total'] for tmp_dict in self.group_acc_dict["global"]])
        log_str += "\tACC: {}\n".format(self.group_acc_dict["global"][-1])
        if len(self.forgetting_measure_dict["global"]) > 0:
            log_str += "\tForget Measure: {}\n".format(self.forgetting_measure_dict["global"][-1])

        log_str += "local_test:\n"
        log_str += "\tACC: {:.2f} \u00B1 {:.2f}\n".format(self.local_acc_mean_std[-1]["mean"], self.local_acc_mean_std[-1]["std"])
        if len(self.local_fm_mean_std) > 0:
            log_str += "\tForget Measure: {:.2f} \u00B1 {:.2f}\n".format(self.local_fm_mean_std[-1]["mean"], self.local_fm_mean_std[-1]["std"])
        for i in range(self.args["num_users"]):
            log_str += "\tuser{}:".format(i)
            log_str += "\tACC: {}\n".format(self.group_acc_dict[i][-1])
            if len(self.forgetting_measure_dict[i]) > 0:
                log_str += "\t\t\tForget Measure: {}\n".format(self.forgetting_measure_dict[i][-1])
        log_str += "----------------------------------------------------------\n"
        if save_to_file:
            with open(log_txt_pth, "a") as file:
                file.write(log_str)
        if print_to_console:
            print(log_str)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    def per_cls_acc(self, val_loader, model):
        model.eval()
        model = model.cuda()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for _, (_, input, target) in enumerate(val_loader):
                input, target = input.cuda(), target.cuda()
                # compute output
                output = model(input)["logits"]
                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / (cls_cnt + 1e-8)
        return cls_acc, np.array(all_preds), np.array(all_targets)

    def init_dataloader(self, data_manager):
        train_dataset = data_manager.get_dataset(   #* get the data for one task
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )
        user_groups, label_ratio = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"], return_ratio=True)
        self.label_ratio = label_ratio if self.label_ratio is None \
                else np.concatenate((self.label_ratio, label_ratio), axis=1)
        self.local_train_loaders = []
        for idx in range(len(user_groups)):
            self.local_train_loaders.append(DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                    batch_size=self.args["local_bs"], shuffle=True, num_workers=4))
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
