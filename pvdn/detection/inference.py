import cv2
import torch
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from ptflops import get_model_complexity_info
from pypapi import papi_high
from pypapi import events as papi_events
from pypapi.exceptions import PapiNoEventError
import argparse

from pvdn.detection.model.proposals import DynamicBlobDetector
from pvdn.detection.model.single_flow_classifier import Classifier
from pvdn import PVDNDataset
from pvdn.detection.utils.misc import crop_bboxes


class Runner:
    def __init__(self, yaml: str, weights: str,
                 device: str = "cuda", bbox_size: tuple = (64, 64)):
        """
        :param yaml: path to the blob detector config (.yaml)
        :param weights: path to the model weights (.pt or .pth)
        :param device: 'cuda' or 'cpu'
        :param bbox_size: Size of the bounding boxes used for classification (height, width)
        """
        # setup blob detector
        self.detector = DynamicBlobDetector.from_yaml(yaml)

        # setup model
        self.model = Classifier()
        weights = torch.load(weights, map_location=device)
        if type(weights) == dict:
            weights = weights["model"]
        self.model.load_state_dict(weights)
        self.model = self.model.to(device)

        # further parameters
        self.device = device
        self.bbox_size = bbox_size

        self.model_flops, self.model_params = self.analyze_model(
            self.model, (1, self.bbox_size[0], self.bbox_size[1])
        )

        self._counter_available = True
        try:
            papi_high.start_counters([papi_events.PAPI_DP_OPS])
            papi_high.stop_counters()
        except PapiNoEventError:
            print("CPU architecture does not support the FLOP counter necessary to "
                  "get the complexity information for the bounding box generation "
                  "process. Thus, only the model complexity will be reported.\n")
            self._counter_available = False

    @staticmethod
    def analyze_model(model, input_size):
        macs, params = get_model_complexity_info(model,
                                                 input_size,
                                                 as_strings=False,
                                                 print_per_layer_stat=False,
                                                 verbose=False)
        flops = macs * 2
        return flops, params

    def infer(self, img: np.ndarray, return_stats=False):
        """
        Runs the whole pipeline for one image including following steps:
            1. Blob detector (finds the bounding boxes in the image)
            2. Classifier (classifies the bounding boxes true or false)
        :param img: Input image as a np.ndarray of the shape [h, w, c] and type uint8
        :param return_stats: Flag to return the inference statistics such as flops and
            runtime.
        :return: class probabilities for each bounding box found in the image
        """
        start = timer()
        if self._counter_available:
            papi_high.start_counters([papi_events.PAPI_DP_OPS])

        # create bounding boxes
        proposals = self.detector.propose(img)

        # crop bounding boxes out of image
        cropped = crop_bboxes(img, proposals)
        bboxes = []

        # resize them all to constant size
        for box in cropped:
            bboxes.append(
                cv2.resize(box, dsize=self.bbox_size, interpolation=cv2.INTER_LINEAR)
            )
        bboxes = np.array(bboxes)
        blob_runtime = timer() - start

        blob_flops = 0
        if self._counter_available:
            blob_flops = papi_high.read_counters()[0]
            papi_high.stop_counters()

        # shift from uint8 to float
        bboxes = bboxes.astype(float) / 255.0
        bboxes = torch.from_numpy(bboxes).unsqueeze(1).float()
        bboxes = bboxes.to(self.device)
        start = timer()

        # classify each bounding box
        cls_probs = []
        if len(bboxes > 0):
            cls_probs = self.model(bboxes)
        model_runtime = timer() - start

        model_stats = {
            "flops": self.model_flops * len(bboxes),
            "runtime": model_runtime,
            "params": self.model_params
        }

        blob_stats = {
            "flops": blob_flops,
            "runtime": blob_runtime
        }

        if return_stats:
            return proposals, cls_probs, model_stats, blob_stats
        return proposals, cls_probs


def analyze_performance(data: str, yaml: str, weights: str, device: str = "cuda",
                        bbox_size: tuple = (64, 64)) -> tuple:
    """
    TODO: docstring
    """

    dataset = PVDNDataset(path=data)
    input_size = tuple(dataset[0][0].shape)
    pipeline = Runner(yaml=yaml, weights=weights, device=device, bbox_size=bbox_size)

    model_hist = {
        "flops": [],
        "runtime": [],
        "params": []
    }
    blob_hist = {
        "flops": [],
        "runtime": []
    }
    hist = {
        "n_boxes": [],
        "runtime": [],
        "flops": []
    }
    for img, _, _ in tqdm(dataset):
        bboxes, cls_probs, model_stats, blob_stats = pipeline.infer(img, True)

        for k, v in model_stats.items():
            model_hist[k].append(v)
        for k, v in blob_stats.items():
            blob_hist[k].append(v)

        hist["n_boxes"].append(len(bboxes))
        hist["runtime"].append(model_stats["runtime"] + blob_stats["runtime"])
        hist["flops"].append(model_stats["flops"] + blob_stats["flops"])

    # average
    print("Statistics:\n\n"
          "Model:\n"
          f"\tAvg. Runtime:\t{round(np.mean(model_hist['runtime'])*1000, 2)} +- "
          f"{round(np.std(model_hist['runtime'])*1000, 2)} ms\n"
          f"\tParameters:\t\t{np.mean(model_hist['params'])} +- "
          f"{np.round(np.std(model_hist['params']))}\n"
          f"\tFLOPs:\t\t\t{np.round(np.mean(model_hist['flops']))} +- "
          f"{np.round(np.std(model_hist['flops']))}\n")

    print("Blob Detector:\n"
          f"\tAvg. Runtime:\t{round(np.mean(blob_hist['runtime']) * 1000, 2)} +- "
          f"{round(np.std(blob_hist['runtime']) * 1000, 2)} ms\n"
          f"\tFLOPs:\t\t\t{np.round(np.mean(blob_hist['flops']))} +- "
          f"{np.round(np.std(blob_hist['flops']))}\n"
          )

    print("Total:\n"
          f"\tAvg. Runtime:\t{round(np.mean(hist['runtime']) * 1000, 2)} +- "
          f"{round(np.std(hist['runtime']) * 1000, 2)} ms\n"
          f"\tFLOPs:\t\t\t{np.round(np.mean(hist['flops']))} +- "
          f"{np.round(np.std(hist['flops']))}\n"
          )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str,
                        help="/path/to/dataset/PVDN/day/test")
    parser.add_argument("-y", "--yaml", type=str,
                        help="/path/to/BlobDetectorParameters.yaml")
    parser.add_argument("-w", "--weights", type=str,
                        help="/path/to/weights_pretrained.pt")
    opts = parser.parse_args()

    analyze_performance(data=opts.data,
                        yaml=opts.yaml,
                        weights=opts.weights)
