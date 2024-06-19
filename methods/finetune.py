from copy import deepcopy
import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

from methods.base import BaseLearner
from utils import (
    IncrementalNet, 
    average_weights, 
    setup_seed
)


class Finetune(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        setup_seed(self.seed)
        self._cur_task += 1
        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self.init_dataloader(data_manager)
        self._fl_train()

    def _local_update(self, model, train_data_loader):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.args["local_ep"]):
            for _, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def _local_finetune(self, model, train_data_loader):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.args["local_ep"]):
            for _, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                fake_targets = labels - self._known_classes
                output = model(images)["logits"]
                loss = F.cross_entropy(output[:, self._known_classes :], fake_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def _fl_train(self):
        self._network.cuda()
        prog_bar = tqdm(range(self.args["com_round"]))
        for _, com in enumerate(prog_bar):
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                if self._cur_task == 0:
                    w = self._local_update(deepcopy(self._network), self.local_train_loaders[idx])
                else:
                    w = self._local_finetune(deepcopy(self._network), self.local_train_loaders[idx])
                local_weights.append(deepcopy(w))
            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            # test
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info=("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                self._cur_task, com + 1, self.args["com_round"], test_acc,))
            prog_bar.set_description(info)
