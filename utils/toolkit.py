from tqdm import tqdm
import numpy as np
import torch
from PIL import Image


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot

    
def cal_forget_measure(per_class_acc):
    assert len(per_class_acc) >= 2, "At least two tasks."
    acc_map = np.zeros(shape=(len(per_class_acc), len(per_class_acc[-2])))
    for i, tmp_acc_ls in enumerate(per_class_acc):
        tmp_len = min(len(tmp_acc_ls), acc_map.shape[-1])
        acc_map[i, :tmp_len] = tmp_acc_ls[:tmp_len]
    per_cls_fm = acc_map.max(axis=0) - acc_map[-1]
    return per_cls_fm.mean()


def grouped_accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs): # all img names
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def split_images_labels2(imgs): # all img names
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in tqdm(imgs, desc="load dataset"):
        image = Image.open(item[0])
        image = image.convert("RGB")
        image = torch.tensor([list(image.getdata())], dtype=torch.float32)
        image = image.view(64, 64, 3)
        image = image.numpy().astype(np.uint8)
        images.append(image)
        labels.append(item[1])

    return np.stack(images, axis=0), np.array(labels)
