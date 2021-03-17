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
from pvdn.detection.engine import train_one_epoch, val_one_epoch
from pvdn.metrics.bboxes import BoundingBoxEvaluator


def train(train_data, val_data, epochs, lr, bs, conf_thresh=0.5, output_dir=None, model_path=None, \
          device="cuda", n_workers=16, save_epochs=False):

    # seed for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    # set up output path
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, "checkpoints"))
        os.mkdir(os.path.join(output_dir, "predictions"))
        os.mkdir(os.path.join(output_dir, "best"))

    # set up logging
    writer = SummaryWriter(logdir=output_dir)

    # check device
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"
        warn("CUDA device cannot be found.")
    print(f"Device:\t{device}")

    # set up data
    print("\nSetting up data...")
    transforms = Compose([
        ToPILImage(),
        RandomHorizontalFlip(),
        RandomRotation(20),
        RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1. / 1.5, 1.5 / 1.)),
        RandomGamma(),
        ToTensor()
    ])

    if os.path.splitext(args.train_data)[1] == ".pt":
        trainset = torch.load(train_data)
    else:
        trainset = BoundingBoxDataset(train_data, transform=transforms)
        torch.multiprocessing.set_sharing_strategy('file_system')

    if os.path.splitext(args.val_data)[1] == ".pt":
        testset = torch.load(val_data)
    else:
        testset = BoundingBoxDataset(val_data, transform=ToTensor())
        torch.multiprocessing.set_sharing_strategy('file_system')

    train_evaluator = BoundingBoxEvaluator(data_dir=train_data)
    val_evaluator = BoundingBoxEvaluator(data_dir=val_data)

    # calculate class imbalance
    cls_train = trainset.compute_class_imbalance()
    print(f"Training class imbalance:\tTrue = {cls_train['true']}\t"
          f"False = {cls_train['false']}")
    weight_true_labels = cls_train['true'] / len(trainset)
    weight_false_labels = cls_train['false'] / len(trainset)
    class_weights = weight_false_labels / weight_true_labels
    class_weights = torch.tensor(class_weights).to(device)
    print(f"Training class weights:\t\t{class_weights.item()}")

    print("\nInitializing loaders...")
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=bs,
                                              shuffle=True,
                                              num_workers=n_workers)
    testloader = torch.utils.data.DataLoader(testset,
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
        ckp = torch.load(model_path)
        print("Initialized model from checkpoint.")
        if not "model" in ckp.keys():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(ckp["model"])
            if "optimizer" in ckp.keys():
                print("Initialized optimizer from checkpoint.")
                optimizer.load_state_dict(ckp["optimizer"])
            if "scheduler" in ckp.keys():
                print("Initialized scheduler from checkpoint.")
                scheduler.load_state_dict()

    # train
    print("Starting training...")
    best_metrics = None
    for epoch in range(epochs):
        print(f"\nEpoch {epoch} / {epochs}")

        train_loss, train_predictions = train_one_epoch(model=model, dataloader=trainloader,
                                                        criterion=criterion, optimizer=optimizer,
                                                        device=device)
        train_evaluator.load_results_from_dict(train_predictions)
        train_metrics = train_evaluator.evaluate(conf_thresh=conf_thresh)
        train_metrics["loss"] = round(train_loss, ndigits=6)
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        train_metrics_str = [f"{k}: {v:4f}" for k, v in train_metrics.items()]

        val_loss, val_predictions = val_one_epoch(model=model, dataloader=testloader,
                                                  criterion=criterion, device=device)
        val_evaluator.load_results_from_dict(val_predictions)
        val_metrics = val_evaluator.evaluate(conf_thresh=conf_thresh)
        val_metrics["loss"] = round(val_loss, ndigits=6)
        for k, v in val_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        val_metrics_str = [f"{k}: {v:4f}" for k, v in val_metrics.items()]

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
            best_epochs = {k: 0 for k in best_metrics.keys()}
        else:
            for k, v in val_metrics.items():
                if "loss" in k:
                    if v < best_metrics[k]:
                        best_metrics[k] = v
                        best_epochs[k] = epoch
                        torch.save(ckp, os.path.join(output_dir, "best", f"ckp_best_{k}.pt"))
                        with open(os.path.join(output_dir, "best", f"val_predictions_best_"
                                  f"{k}.json"), "w") as f:
                            json.dump(val_predictions, f)

                else:
                    if v > best_metrics[k]:
                        best_metrics[k] = v
                        best_epochs[k] = epoch
                        torch.save(ckp, os.path.join(output_dir, "checkpoints", f"best_{k}.pt"))

        if save_epochs:
            torch.save(ckp, os.path.join(output_dir, "checkpoints", f"ckp_epoch_{epoch}.pt"))
            with open(os.path.join(output_dir, "predictions", f"val_predictions_epoch_"
                      f"{epoch}.json"), "w") as f:
                json.dump(val_predictions, f)

    print(f"\nTraining completed successfully for {epochs} epochs.\n"
          f"Logs written to {output_dir}.")
    best_str = [f"{k}: {v:4f} (epoch {best_epochs[k]})" for k, v in best_metrics.items()]
    print(f"-------------- Best --------------")
    for s in best_str:
        print(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str,
                        help="Either the path to a preprocessed *.pt file or to the root dir "
                             "for the BoundingBoxDataset.",
                        default="/raid/Datasets/EODAN/kaggle/day/train")
    parser.add_argument("--val_data", type=str,
                        help="Either the path to a preprocessed *.pt file or to the root dir "
                             "for the BoundingBoxDataset.",
                        default="/raid/Datasets/EODAN/kaggle/day/val")
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs to train.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers to use for "
                                                                "dataloader.")
    parser.add_argument("--output_dir", type=str, default="runs",
                        help="Folder where the results will be stored.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Restart the training from a given model path.")
    parser.add_argument("--save_epochs", action="store_true", help="Flag to set if a checkpoint "
                                                                   "is supposed to be saved each "
                                                                   "epoch")
    parser.add_argument("--device", choices=("cuda", "cpu"), type=str, default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="Confidence threshold at which to classify positive.")
    args = parser.parse_args()

    args.model_path = "weights_pretrained.pt"
    train(args.train_data, args.val_data, args.epochs, args.lr, args.batch_size, args.conf_thresh,
          args.output_dir, args.model_path, n_workers=args.workers, save_epochs=args.save_epochs,
          device=args.device)
