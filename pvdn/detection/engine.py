from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import numpy as np
from torch.nn import Module
import torch

def train_one_epoch(model, dataloader: DataLoader, criterion, optimizer: Optimizer, device):
    model.train()
    model = model.to(device)
    loss_hist = []
    pred_dict = {}
    for imgs, labels, bb_coords, ids in tqdm(dataloader, desc=f"Training"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels.unsqueeze(1))

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        loss_hist.append(loss.item())
        for i, img_id in enumerate(ids):
            img_id = img_id.item()
            if str(img_id) not in pred_dict.keys():
                pred_dict[str(img_id)] = {"boxes": [bb_coords[i].tolist()], "scores": [outputs[
                                                                                      i].item()]}
            else:
                pred_dict[str(img_id)]["boxes"].append(bb_coords[i].tolist())
                pred_dict[str(img_id)]["scores"].append(outputs[i].item())

    return np.mean(loss_hist), pred_dict


@torch.no_grad()
def val_one_epoch(model: Module, dataloader: DataLoader, criterion, device, task="Validation"):
    model.eval()
    model = model.to(device)
    loss_hist = []
    pred_dict = {}
    loss = None
    for imgs, labels, bb_coords, ids in tqdm(dataloader, desc=task):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        if criterion:
            loss = criterion(outputs, labels.unsqueeze(1))
            loss_hist.append(loss.item())

        # logging
        for i, img_id in enumerate(ids):
            img_id = img_id.item()
            if str(img_id) not in pred_dict.keys():
                pred_dict[str(img_id)] = {"boxes": [bb_coords[i].tolist()],
                                          "scores": [outputs[i].item()]}
            else:
                pred_dict[str(img_id)]["boxes"].append(bb_coords[i].tolist())
                pred_dict[str(img_id)]["scores"].append(outputs[i].item())

    return np.mean(loss_hist), pred_dict