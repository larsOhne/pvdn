import os
import json
import numpy as np
from tqdm import tqdm
import os
from warnings import warn

from pvdn import PVDNDataset


class BoundingBoxEvaluator():
    """
    Class which helps to evaluate the results on the PVDN dataset when predicting bounding boxes.
    """
    def __init__(self, dataset: PVDNDataset):
        """
        :param dataset: PVDNDataset object containing the ground truth annotations.
        """
        if not isinstance(dataset, PVDNDataset):
            raise TypeError(f"Dataset has to be of type {PVDNDataset}, not {type(dataset)}.")
        self._predictions = None
        self.gt_kps = {}
        self._nbr_images = len(dataset.img_idx)
        print("Indexing keypoint annotations...")
        for i in dataset.img_idx:
            kp_path = os.path.join(dataset.keypoints_path, "{:06d}.json".format(i))
            with open(kp_path, "r") as kp_file:
                kp_dict = json.load(kp_file)
                kps = []
                for vehicle_annot in kp_dict["annotations"]:
                    kps += [inst["pos"] for inst in vehicle_annot["instances"]]
                self.gt_kps[i] = kps
        print("...done!\n")

    def load_results_from_file(self, path: str) -> None:
        """
        Results are loaded from .json file with the structure:
        {"image_id": [[x1, y1, x2, y2], [...], ...], ...}
        :param path: path to the .json results file.
        """
        if not os.path.splitext(path)[-1]:
            raise AttributeError(f"File extension should be .json not {os.path.splitext(path)[-1]}")
        with open(path, "r") as f:
            self._predictions = json.load(f)
        if len(self._predictions.items()) == 0:
            warn(f"Results are empty. You might want to check your results file ({path}).")

    def load_results_from_dict(self, results: dict) -> None:
        """
        Results are loaded from a dict with the structure:
        {"image_id": [[x1, y1, x2, y2], [...], ...], ...}
        :param results: Dictionary containing the results.
        """
        self._predictions = results
        if len(self._predictions.items()) == 0:
            warn(f"Results are empty. You might want to check your results dict.")

    def _kp_in_box(self, kp: [np.ndarray, list], box: [np.ndarray, list], tolerance: float = 0.0) \
            -> bool:
        """
        Checks if a keypoint lies within a bounding box. The height & width of the bounding box
        can be extended by a tolerance factor.
        :param kp: Keypoint which has to be checked in form of an array [x, y] -> np.ndarray
        :param box: Bounding box to be checked in form of an array [x1, y1, x2, y2] -> np.ndarray
        :param tolerance: Scale factor to extend the bounding box width and height
        """
        tolerance /= 2
        height = (box[3] - box[1]) * tolerance
        width = (box[0] - box[2]) * tolerance
        diff = np.array([-width, -height, width, height])
        box += diff
        return box[0] <= kp[0] <= box[2] and box[1] <= kp[1] <= box[3]

    @staticmethod
    def _get_nbr_of_kps_in_box(box: [np.ndarray, list], kps: [np.ndarray, list]) -> int:
        """
        Calculates the number of keypoints in a specific bounding box.
        :param box: bounding box as an array of the form [x1, y1, x2, y2] -> np.ndarray
        :param kps: all keypoints in the image as an of shape [nbr_of_kps, 2],
            where each keypoint is of the form [x, y] -> np.ndarray
        """
        if len(kps) == 0:
            return 0

        return np.count_nonzero(
            np.bitwise_and(np.bitwise_and(box[0] <= kps[:, 0], kps[:, 0] <= box[2]), np.bitwise_and(
                box[1] <= kps[:, 1], kps[:, 1] <= box[3]))
        )

    @staticmethod
    def _get_nbr_of_boxes_containing_kp(kp: np.ndarray, boxes: np.ndarray) -> int:
        """
        Calculates the number of boxes spanning over a specific keypoint.
        :param kp: keypoint as an array of the form [x, y] -> np.ndarray
        :param boxes: bounding boxes as a np.ndarray of shape [nbr_of_boxes, 4], where each
            bounding box is of the form [x1, y1, x2, y2].
        """
        if len(boxes) == 0:
            return 0

        return np.count_nonzero(
            np.bitwise_and(np.bitwise_and(boxes[:, 0] <= kp[0], kp[0] <= boxes[:, 2]),
                           np.bitwise_and(boxes[:, 1] <= kp[1], kp[1] <= boxes[:, 3]))
        )

    def evaluate(self, verbose=False) -> dict:
        """
        Evaluates the results based on the cool PVDN metric for bounding box prediction. If an
        image from the dataset is not present in the result dict it is treated as if there was no
        prediction for this image.
        :param verbose: flag to show details on the evaluation process. -> bool
        :return: dict with the keys
            "precision": TP / (TP + FP)
            "recall": TP / (TP + FN)
            "f1_score": TP / (TP + 0.5 * (FP + FN))
            "box_quality": estimate for the quality of the predicted boxes
        """
        self._total_scores = {"tps": 0, "boxes": 0, "kps": 0, "fps": 0, "fns": 0}

        if self._predictions is None:
            raise ValueError("You need to load the results first by calling the "
                             "load_results_from_file or load_results_from_dict function.")

        no_key_counter = 0
        box_quality = []
        kp_quality = []
        for id, kps in tqdm(self.gt_kps.items(), disable=not verbose):
            kps = np.array(kps)

            img_scores = {"tps": 0, "boxes": 0, "kps": 0, "fps": 0, "fns": 0}

            if str(id) in self._predictions.keys():
                pred_boxes = np.array(self._predictions[str(id)])
            else:
                pred_boxes = []
                no_key_counter += 1

            # first check every box
            # penalize if one box contains more than one kp
            for box in pred_boxes:
                nbr_kps_in_box = self._get_nbr_of_kps_in_box(box=box, kps=kps)

                # if there is no kp in the box it is a false positive, otherwise true positive
                if nbr_kps_in_box == 0:
                    img_scores["fps"] += 1
                else:
                    # goal: exactly one kp in each box
                    # box quality is lower the more kps there are in one box
                    box_quality.append(1 / nbr_kps_in_box)

            # second check every kp
            # penalize if one kp lies in several boxes
            for kp in kps:
                nbr_boxes_containing_kp = self._get_nbr_of_boxes_containing_kp(kp=kp,
                                                                               boxes=pred_boxes)

                # if there is no box containing the kp it is a false negative
                if nbr_boxes_containing_kp == 0:
                    img_scores["fns"] += 1
                else:
                    img_scores["tps"] += 1
                    # goal: each kp lies in only one box
                    # quality is lower the more boxes the kp lies in
                    kp_quality.append(1 / nbr_boxes_containing_kp)

            self._total_scores = {k: v + self._total_scores[k] for k, v in img_scores.items()}

        nds = 3
        # now we have TPs, FPs, and FNs and can calculate precision & recall
        precision = round(self._total_scores["tps"] / (self._total_scores["tps"] +
                                                      self._total_scores["fps"]), ndigits=nds)
        recall = round(self._total_scores["tps"] / (self._total_scores["tps"] +
                    self._total_scores["fns"]), ndigits=nds)
        f1_score = round(self._total_scores["tps"] / (self._total_scores["tps"] + 0.5 * (
                self._total_scores["fps"] + self._total_scores["fns"])), ndigits=nds)

        box_quality = round(np.mean(box_quality), ndigits=nds)
        kp_quality = round(np.mean(kp_quality), ndigits=nds)
        box_quality_combined = round(box_quality * kp_quality, ndigits=nds)

        if verbose:
            print(f"Could not find {no_key_counter} of {self._nbr_images} images.")
            print("----------------------------------")
            print(f"Precision:\t\t{precision}\n"
                  f"Recall:\t\t\t{recall}\n"
                  f"F1 Score:\t\t{f1_score}\n"
                  f"Box Quality:\t{box_quality_combined}")
            print("----------------------------------")

        return {"precision": precision, "recall": recall, "f1_score": f1_score,
                "box_quality": box_quality_combined}


def evaluate_single(src, dataset_path):
    dataset = PVDNDataset(path=dataset_path)
    evaluator = BoundingBoxEvaluator(dataset=dataset)
    evaluator.load_results_from_file(src)
    evaluator.evaluate(verbose=True)


def find_best(src, file_pattern, dataset_path):
    dataset = PVDNDataset(path=dataset_path)
    evaluator = BoundingBoxEvaluator(dataset=dataset)

    best = {"precision": {"model": None, "value": 0}, "recall": {"model": None, "value": 0},
            "f1_score": {"model": None, "value": 0}, "box_quality": {"model": None, "value": 0}}

    for pred_file in sorted(os.listdir(src)):
        if file_pattern in pred_file:
            try:
                pred_path = os.path.join(src, pred_file)
                evaluator.load_results_from_file(pred_path)
                yolov5s_results = evaluator.evaluate(verbose=False)
                for k in yolov5s_results.keys():
                    if yolov5s_results[k] > best[k]["value"]:
                        best[k]["value"] = yolov5s_results[k]
                        best[k]["model"] = pred_path
            except:
                continue

    for k in best.keys():
        print(k)
        print(best[k]["model"])
        evaluator.load_results_from_file(best[k]["model"])
        evaluator.evaluate(verbose=True)
        print()


if __name__ == "__main__":

    src = "/media/lukas/empty/EODAN_Dataset/results/custom/test"
    pattern = ".json"
    # dataset_path = "/media/lukas/empty/EODAN_Dataset/day/val"
    dataset_path = "/media/lukas/empty/EODAN_Dataset/day/test"
    # find_best(src, pattern, dataset_path)
    evaluate_single("/media/lukas/empty/EODAN_Dataset/results/gt_test_results.json",
                    dataset_path)