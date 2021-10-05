import typing

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import numpy as np
from torch.nn import Module
from torch import sigmoid
import torch
from typing import Union


def train_one_epoch(model: Module, dataloader: DataLoader, criterion,
                    optimizer: Optimizer, device: str) -> typing.Tuple[
    np.ndarray, dict]:
    """
    Train the given model for one epoch.
    :param model: model to train
    :type model: Module
    :param dataloader: data loader to load batches
    :type dataloader: DataLoader
    :param criterion: function for calculating the loss
    :type criterion:
    :param optimizer: optimizer to use
    :type optimizer: Optimizer
    :param device: device to train the model on, should be either "cuda" or "cpu"
    :type device: str
    :return: average loss over all batches, dictionary containing the
    prediction scores ["scores"], bounding boxes ["bboxes"],
    and ground truth labels ["lables"]
    :rtype: typing.Tuple[np.ndarray, dict]
    """
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
        outputs = sigmoid(outputs)
        update_prediction_dict(pred_dict, outputs, labels,
                               bb_coords, ids)

    return np.mean(loss_hist), pred_dict


@torch.no_grad()
def val_one_epoch(model: Module, dataloader: DataLoader, criterion,
                  device: str) -> typing.Tuple[typing.Union[np.ndarray, None],
                                               dict]:
    """
    Validate the given model for one epoch.
    :param model: model to evaluate
    :type model: Module
    :param dataloader: data loader for batch loading
    :type dataloader: DataLoader
    :param criterion: loss function
    :type criterion: 
    :param device: device to train the model on, should be either "cpu" or "cuda"
    :type device: str
    :return: average loss over all batches or None, if criterion isnt 
    specified, dictionary containing the
    prediction scores ["scores"], bounding boxes ["bboxes"],
    and ground truth labels ["lables"]
    :rtype: (np.ndarray, dict) if criterion is specified, otherwise (None, dict)
    """
    model.eval()
    model = model.to(device)
    loss_hist = []
    pred_dict = {}
    for imgs, labels, bb_coords, ids in tqdm(dataloader, desc="Validation"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        if criterion:
            loss = criterion(outputs, labels.unsqueeze(1))
            loss_hist.append(loss.item())

        outputs = sigmoid(outputs)
        # logging
        update_prediction_dict(pred_dict, outputs, labels,
                               bb_coords, ids)
    if criterion:
        return np.mean(loss_hist), pred_dict
    else:
        return None, pred_dict


def update_prediction_dict(pred_dict: dict,
                           outputs: Union[np.ndarray, torch.Tensor],
                           labels: Union[np.ndarray, torch.Tensor],
                           bb_coords: Union[np.ndarray, torch.Tensor],
                           ids: Union[np.ndarray, torch.Tensor]):
    """
    Update the given prediction dictionary based on the given parameters.
    Prediction dictionary is build as follows:
    {"img_id": info} with info beeing another dictionary {"boxes": [list of
    bounding box coordinates in the image], "scores": [list of predicted
    scores], "labels": [list of ground truth labels]}
    :param pred_dict: prediction dictionary to update
    :type pred_dict: dict
    :param outputs: scores predicted by the labels
    :type outputs: Union[np.ndarray, torch.Tensor]
    :param labels: ground truth labels
    :type labels: Union[np.ndarray, torch.Tensor]
    :param bb_coords: coordinates of the bounding boxes
    :type bb_coords: Union[np.ndarray, torch.Tensor]
    :param ids: img ids
    :type ids: Union[np.ndarray, torch.Tensor]
    :return: updated prediction dictionary
    :rtype: dict
    """
    for i, img_id in enumerate(ids):
        img_id = img_id.item()
        if str(img_id) not in pred_dict.keys():
            pred_dict[str(img_id)] = {"boxes": [bb_coords[i].tolist()],
                                      "scores": [outputs[i].item()],
                                      "labels": [labels[i].item()]}
        else:
            pred_dict[str(img_id)]["boxes"].append(bb_coords[i].tolist())
            pred_dict[str(img_id)]["scores"].append(outputs[i].item())
            pred_dict[str(img_id)]["labels"].append(labels[i].item())
