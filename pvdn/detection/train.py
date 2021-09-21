import argparse
import os
from warnings import warn
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, \
    RandomResizedCrop, Compose, ToTensor, ToPILImage
from tensorboardX import SummaryWriter

from pvdn import BoundingBoxDataset
from pvdn.detection.model.single_flow_classifier import Classifier
from pvdn.detection.data.transforms import RandomGamma
from pvdn.detection.model.proposals import DynamicBlobDetector
from pvdn.detection.engine import train_one_epoch, val_one_epoch
from pvdn.metrics.bboxes import BoundingBoxEvaluator, evaluate


def setup_output_dir(output_dir: str):
    """
    Setup the output directory.
    :param output_dir: path where the data is written to
    :type output_dir: str
    """
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, "checkpoints"))
        os.mkdir(os.path.join(output_dir, "predictions"))
        os.mkdir(os.path.join(output_dir, "best"))


def device_available(device: str) -> bool:
    """
    Check if the given device is available to use.
    If it is not available, a warning is given.
    :param device: device on that the training should run
    :type device: str
    :return: True, if the device is available, False otherwise
    :rtype: bool
    """
    if "cuda" in device and not torch.cuda.is_available():
        warn("CUDA device cannot be found.")
        return False
    return True


def load_data_set(set_path: str,
                  transformations: Compose) -> BoundingBoxDataset:
    """
    Load the data set.
    :param set_path: path to base dir or to preprocessed ".pt" file
    :type set_path: str
    :param transformations: list of transformations to apply to the data set
    :type transformations: list
    :return: loaded data set
    :rtype: BoundingBoxDataset
    """
    if os.path.splitext(set_path)[1] == ".pt":
        set_ = torch.load(set_path)
    else:
        set_ = BoundingBoxDataset(set_path, transform=transformations)
    return set_


def compute_class_weights(set_: BoundingBoxDataset,
                          verbose: bool = True) -> float:
    """
    Compute the class_weights that are used for BCEDigitsLoss
    :param set_: the set the class weights should be computed for
    :type set_: BoundingBoxDataset
    :param verbose: if True, output information about class imbalance and
    class weights, defaults True
    :type verbose: bool
    :return: calculated class weights
    :rtype: float
    """
    cls_train = set_.compute_class_imbalance()
    weight_true_labels = cls_train['true'] / len(set_)
    weight_false_labels = cls_train['false'] / len(set_)
    class_weights = weight_false_labels / weight_true_labels
    if verbose:
        print(f"Training class imbalance:\tTrue = {cls_train['true']}\t"
              f"False = {cls_train['false']}")
        print(f"Training class weights:\t\t{class_weights}")
    return class_weights


def load_from_model_path(model_path: str, model: Classifier,
                         optimizer: optim.Optimizer,
                         scheduler, device: str):
    """
    Load model, and if possible optimizer and scheduler from the model_path
    :param model_path: path to the ".pt" file where the model, (optimizer,
    and scheduler) are stored
    :type model_path: str
    :param model: model to train
    :type model: Classifier
    :param optimizer: optimizer used for training
    :type optimizer: optim.Optimizer
    :param scheduler: scheduler used for training
    :type scheduler: one of the schedulers of optim.lr_scheduler
    :param device: the device the model should be loaded to ("cpu" or "cuda")
    :type device: str
    """
    ckp = torch.load(model_path, map_location=torch.device(device))
    print("Initialized model from checkpoint.")
    if "model" not in ckp.keys():
        model.load_state_dict(torch.load(model_path,
                                         map_location=torch.device(device)))
    else:
        model.load_state_dict(ckp["model"])
        if "optimizer" in ckp.keys():
            print("Initialized optimizer from checkpoint.")
            optimizer.load_state_dict(ckp["optimizer"])
        if "scheduler" in ckp.keys():
            print("Initialized scheduler from checkpoint.")
            scheduler.load_state_dict(ckp["scheduler"])


def save_epoch(ckp: dict, output_dir: str, epoch: int, val_predictions: dict):
    """
    Save the current epoch
    :param ckp: dictionary storing the state dictionary of the model,
    optimizer, and scheduler
    :type ckp: dict
    :param output_dir: path to the directory where the checkpoint should be
    saved to
    :type output_dir: str
    :param epoch: current epoch
    :type epoch: int
    :param val_predictions: predictions of the current epoch
    :type val_predictions: dict
    """
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(os.path.join(output_dir, "checkpoints")):
        os.mkdir(os.path.join(output_dir, "checkpoints"))
    if not os.path.isdir(os.path.join(output_dir, "predictions")):
        os.mkdir(os.path.join(output_dir, "prediction"))

    torch.save(ckp, os.path.join(output_dir, "checkpoints",
                                 f"ckp_epoch_{epoch}.pt"))
    with open(os.path.join(output_dir, "predictions",
                           f"val_predictions_epoch_"
                           f"{epoch}.json"), "w") as f:
        json.dump(val_predictions, f)


def update_best_models(ckp: dict, best_metrics: dict, best_epochs: dict,
                       val_metrics: dict, val_predictions: dict,
                       epoch: int, output_dir: str):
    """
    Update the dictionary storing the information about the best models.
    Saves the model if one metric is better than the best one before this
    epoch.
    :param ckp: checkpoint containing model, optimizer and scheduler recent
    state
    :type ckp: dict
    :param best_metrics: recent best metrics
    :type best_metrics: dict
    :param best_epochs: recent best epochs
    :type best_epochs: dict
    :param val_metrics: valuation metrics of the recent epoch
    :type val_metrics: dict
    :param val_predictions: predicions of the recent epoch
    :type val_predictions: dict
    :param epoch: recent epoch
    :type epoch: int
    :param output_dir: directory where to save the files to
    :type output_dir: str
    """
    output_dir = os.path.abspath(output_dir)
    for k, v in val_metrics.items():
        if "loss" in k:
            if v < best_metrics[k]:
                best_metrics[k] = v
                best_epochs[k] = epoch
                torch.save(ckp, os.path.join(output_dir, "best",
                                             f"ckp_best_{k}.pt"))
                with open(os.path.join(output_dir, "best",
                                       f"val_predictions_best_"
                                       f"{k}.json"), "w") as f:
                    json.dump(val_predictions, f)

        else:
            if v > best_metrics[k]:
                best_metrics[k] = v
                best_epochs[k] = epoch
                torch.save(ckp, os.path.join(output_dir, "checkpoints",
                                             f"best_{k}.pt"))


def print_end_summary(epochs: int, output_dir: str, best_epochs: dict,
                      best_metrics: dict):
    """
    Outputs the final summary to the terminal.
    :param epochs: number of epochs the model has been trained
    :type epochs: int
    :param output_dir: directory where the results are written to
    :type output_dir: str
    :param best_epochs: dictionary with metrics as keys, and epoch number of
    the model that has the best performance in it as value
    :type best_epochs: dict
    :param best_metrics: dictionary storing the best result for each metric
    :type best_metrics: dict
    """
    print(f"\nTraining completed successfully for {epochs} epochs.\n"
          f"Logs written to {output_dir}.")
    best_str = [f"{k}: {v:4f} (epoch {best_epochs[k]})" for k, v in
                best_metrics.items()]
    print(f"-------------- Best --------------")
    for s in best_str:
        print(s)


def train(train_data: str, val_data: str, epochs: int, lr: float, bs: int,
          conf_thresh: float = 0.5, output_dir: str = None,
          model_path: str = None, device: str = "cuda", n_workers: int = 16,
          save_epochs: bool = False):
    """
    Train the model that is presented in the IROS paper 2021.
    :param train_data: path to the base directory of the training set
    :type train_data: str
    :param val_data: path to the base directory of the validation set
    :type val_data: str
    :param epochs: number of epocjhs to train
    :type epochs: int
    :param lr: initial learning rate
    :type lr: float
    :param bs: batch size for the data loader
    :type bs: int
    :param conf_thresh: confidence threshold at which to classify a bounding
    box a positive
    :type conf_thresh: float
    :param output_dir: director where the results will be saved to
    :type output_dir: str
    :param model_path: if model path is specified, the model at the given
    path is loaded and the training continues. The model can also contain a
    scheduler and an optimizer
    :type model_path: str
    :param device: device to train on, "cuda" or "cpu"
    :type device: str
    :param n_workers: number of workers for the data loader
    :type n_workers: int
    :param save_epochs: if True, the model and the predictions are stored for each epoch
    :type save_epochs: bool
    """
    # seed for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    setup_output_dir(output_dir)

    # set up logging
    writer = SummaryWriter(logdir=output_dir)

    # check device
    if not device_available(device):
        device = "cpu"
    print(f"Device:\t{device}")

    # set up data
    print("\nSetting up data...")
    transforms = Compose([
        ToPILImage(),
        RandomHorizontalFlip(),
        RandomRotation(20),
        RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1. / 1.5,
                                                             1.5 / 1.)),
        RandomGamma(),
        ToTensor()
    ])

    # load sets
    train_set = load_data_set(train_data, transforms)
    val_set = load_data_set(val_data, transforms)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # TODO: BoundingBoxEvaluator cannot process ".pt" file, remove option
    #  that train_data/val_data can be ".pt" file
    train_evaluator = BoundingBoxEvaluator(data_dir=train_data)
    val_evaluator = BoundingBoxEvaluator(data_dir=val_data)

    # calculate class imbalance
    class_weights = compute_class_weights(train_set, verbose=True)
    class_weights = torch.tensor(class_weights).to(device)

    print("\nInitializing loaders...")
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=bs,
                                               shuffle=True,
                                               num_workers=n_workers)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=bs,
                                             shuffle=False,
                                             num_workers=n_workers)

    model = Classifier()
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     factor=0.9,
                                                     patience=5,
                                                     verbose=True)

    if model_path:
        load_from_model_path(model_path, model, optimizer, scheduler, device)

    # train
    print("Starting training...")
    best_metrics = None
    best_epochs = None
    for epoch in range(epochs):
        # off set by 1
        epoch += 1
        print(f"\nEpoch {epoch} / {epochs}")

        # Training
        train_loss, train_predictions = train_one_epoch(model=model,
                                                        dataloader=train_loader,
                                                        criterion=criterion,
                                                        optimizer=optimizer,
                                                        device=device)
        train_metrics = evaluate(train_evaluator, conf_thresh,
                                 train_predictions)
        train_metrics["loss"] = round(train_loss, ndigits=6)
        train_metrics_str = [f"{k}: {v:4f}" for k, v in train_metrics.items()]
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)

        # Evaluation
        val_loss, val_predictions = val_one_epoch(model=model,
                                                  dataloader=val_loader,
                                                  criterion=criterion,
                                                  device=device)
        val_metrics = evaluate(val_evaluator, conf_thresh,
                               val_predictions)
        val_metrics["loss"] = round(val_loss, ndigits=6)
        val_metrics_str = [f"{k}: {v:4f}" for k, v in val_metrics.items()]
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        print(f"------ Summary epoch {epoch} -------")
        print(f"Training:\t{'   '.join(train_metrics_str)}")
        print(f"Validation:\t{'   '.join(val_metrics_str)}")

        scheduler.step(val_loss)

        ckp = {
            "optim": optimizer.state_dict(),
            "model": model.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        }

        if best_metrics is None:
            best_metrics = val_metrics
            best_epochs = {k: 1 for k in best_metrics.keys()}
            for k in best_metrics.keys():
                torch.save(ckp, os.path.join(output_dir, "best",
                                             f"ckp_best_{k}.pt"))
        else:
            update_best_models(ckp, best_metrics, best_epochs, val_metrics,
                               val_predictions, epoch, output_dir)

        if save_epochs:
            save_epoch(ckp, output_dir, epoch, val_predictions)

    print_end_summary(epochs, output_dir, best_epochs, best_metrics)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str,
                        help="Either the path to a preprocessed *.pt file or to the root dir "
                             "for the BoundingBoxDataset.",
                        default="/raid/Datasets/EODAN/kaggle/day/train")
    parser.add_argument("--val_data", type=str,
                        help="Either the path to a preprocessed *.pt file or to the root dir "
                             "for the BoundingBoxDataset.",
                        default="/raid/Datasets/EODAN/kaggle/day/val")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Epochs to train.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of workers to use for "
                             "dataloader.")
    parser.add_argument("--output_dir", type=str, default="runs",
                        help="Folder where the results will be stored.")
    parser.add_argument("--model_path", type=str,
                        default="weights_pretrained.pt",
                        help="Restart the training from a given model path.")
    parser.add_argument("--save_epochs", action="store_true",
                        help="Flag to set if a checkpoint "
                             "is supposed to be saved each "
                             "epoch")
    parser.add_argument("--device", choices=("cuda", "cpu"), type=str,
                        default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="Confidence threshold at which to classify positive.")
    return parser


if __name__ == "__main__":
    parser_ = create_parser()
    args = parser_.parse_args()

    train(args.train_data, args.val_data, args.epochs, args.lr, args.batch_size,
          args.conf_thresh,
          args.output_dir, args.model_path, n_workers=args.workers,
          save_epochs=args.save_epochs,
          device=args.device)
