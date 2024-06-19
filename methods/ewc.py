from copy import deepcopy
import torch
from torch.nn import functional as F
from torch import optim
import numpy as np
from tqdm import tqdm

from methods.base import BaseLearner
from utils import (
    IncrementalNet, 
    average_weights, 
    setup_seed
)


lamda = 1000
fishermax = 0.0001


class EWC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.fisher_list = {}
        self.mean_list = {}
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

    def _local_update(self, model, train_data_loader, idx):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for _ in range(self.args["local_ep"]):
            for _, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.fisher_list[idx] = self.getFisherDiagonal(train_data_loader, model)
        self.mean_list[idx] = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        return model.state_dict()

    def _local_finetune(self, model, train_data_loader, idx):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for _ in range(self.args["local_ep"]):
            for _, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                fake_targets = labels - self._known_classes
                output = model(images)["logits"]
                #* finetune on the new tasks
                loss_clf = F.cross_entropy(output[:, self._known_classes :], fake_targets)
                loss_ewc = self.compute_ewc(idx)
                loss = loss_clf + lamda * loss_ewc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        alpha = self._known_classes / self._total_classes
        new_finsher = self.getFisherDiagonal(train_data_loader, model)
        for n, p in new_finsher.items():
            new_finsher[n][: len(self.fisher_list[idx][n])] = (
                alpha * self.fisher_list[idx][n]
                + (1 - alpha) * new_finsher[n][: len(self.fisher_list[idx][n])]
            )
        self.fisher_list[idx] = new_finsher
        self.mean_list[idx] = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
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
                    w = self._local_update(deepcopy(self._network), self.local_train_loaders[idx], idx)
                else:
                    w = self._local_finetune(deepcopy(self._network), self.local_train_loaders[idx], idx)
                local_weights.append(deepcopy(w))
            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            # test
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info=("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                self._cur_task, com + 1, self.args["com_round"], test_acc,))
            prog_bar.set_description(info)

    def compute_ewc(self, idx):
        loss = 0
        for n, p in self._network.named_parameters():
            if n in self.fisher_list[idx].keys():
                loss += torch.sum((self.fisher_list[idx][n]) * 
                                  (p[: len(self.mean_list[idx][n])] - 
                                   self.mean_list[idx][n]).pow(2)) / 2
        return loss

    def getFisherDiagonal(self, train_loader, model):
        fisher = {
            n: torch.zeros(p.shape).cuda()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        for _, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = model(inputs)["logits"]
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(fishermax))
        return fisher
