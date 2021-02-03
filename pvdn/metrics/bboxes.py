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
        self._predictions = None
        self.gt_kps = {}
        for i in tqdm(dataset.img_idx):
            kp_path = os.path.join(dataset.keypoints_path, "{:06d}.json".format(i))
            with open(kp_path, "r") as kp_file:
                kp_dict = json.load(kp_file)
                kps = []
                for vehicle_annot in kp_dict["annotations"]:
                    kps += [inst["pos"] for inst in vehicle_annot["instances"]]
                self.gt_kps[i] = kps

    def load_results_from_file(self, path: str) -> None:
        """
        Results are loaded from .json file with the structure:
        {"image_id": [[x1, y1, x2, y2], [...], ...], ...}
        :param path: path to the .json results file.
        """
        with open(path, "r") as f:
            self._predictions = json.load(f)
        if len(self._predictions.items()) == 0:
            warn(f"Results are empty. You might want to check your results file ({path}).")

    def load_results_from_dict(self, results: dict) -> None:
        """
        Results are loaded from a dict with the structure:
        {"image_id": [[x1, y1, x2, y2], [...], ...], ...}
        :param results: dict containing the results.
        """
        self._predictions = results
        if len(self._predictions.items()) == 0:
            warn(f"Results are empty. You might want to check your results dict.")

    def _kp_in_box(self, kp: np.ndarray, box: np.ndarray, tolerance: float=0.0) -> bool:
        """
        Checks if a keypoint lies within a bounding box. The height & width of the bounding box
        can be extended by a tolerance factor.
        """
        tolerance /= 2
        height = (box[3] - box[1]) * tolerance
        width = (box[0] - box[2]) * tolerance
        diff = np.array([-width, -height, width, height])
        box += diff
        return box[0] <= kp[0] <= box[2] and box[1] <= kp[1] <= box[3]

    @staticmethod
    def _get_nbr_of_kps_in_box(box: np.ndarray, kps: np.ndarray) -> int:
        """
        Calculates the number of keypoints in a specific bounding box.
        :param box: bounding box as a np.ndarray of the form [x1, y1, x2, y2].
        :param kps: all keypoints in the image as a np.ndarray of shape [nbr_of_kps, 2],
            where each keypoint is of the form [x, y].
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
        :param kp: keypoint as a np.ndarray of the form [x, y].
        :param boxes: bounding boxes as a np.ndarray of shape [nbr_of_boxes, 4], where each
            bounding box is of the form [x1, y1, x2, y2].
        """
        if len(boxes) == 0:
            return 0

        return np.count_nonzero(
            np.bitwise_and(np.bitwise_and(boxes[:, 0] <= kp[0], kp[0] <= boxes[:, 2]),
                           np.bitwise_and(boxes[:, 1] <= kp[1], kp[1] <= boxes[:, 3]))
        )

    def evaluate(self) -> dict:
        """
        Evaluates the results based on the cool PVDN metric for bounding box prediction.
        :return: dict with the keys
            "precision": TP / (TP + FP)
            "recall": TP / (TP + FN)
            "f1_score": TP / (TP + 0.5 * (FP + FN))
        """
        self._total_scores = {"boxes": 0, "kps": 0, "fps": 0, "fns": 0}

        if self._predictions is None:
            raise ValueError("You need to load the results first by calling the "
                             "load_results_from_file or load_results_from_dict function.")

        for id, kps in tqdm(self.gt_kps.items()):
            if len(kps) != 0:
                img_scores = {"boxes": 0, "kps": 0, "fps": 0, "fns": 0}

                try:
                    pred_boxes = np.array(self._predictions[str(id)])
                except KeyError:
                    tqdm.write(f"Missing image id {id} in results. Skipping this one.")
                    continue

                kps = np.array(kps)

                # first check every box
                # penalize if one box contains more than one kp
                for box in pred_boxes:
                    nbr_kps_in_box = self._get_nbr_of_kps_in_box(box=box, kps=kps)

                    # if there is no kp in the box it is a false positive, otherwise true positive
                    if nbr_kps_in_box == 0:
                        img_scores["fps"] += 1
                    else:
                        img_scores["boxes"] += 1 / nbr_kps_in_box

                # normalize by number of boxes in the image
                if len(pred_boxes) != 0:
                    img_scores["boxes"] /= len(pred_boxes)
                    img_scores["fps"] /= len(pred_boxes)

                # second check every kp
                # penalize if one kp lies in several boxes
                for kp in kps:
                    nbr_boxes_containing_kp = self._get_nbr_of_boxes_containing_kp(kp=kp,
                                                                                   boxes=pred_boxes)

                    # if there is no box containing the kp it is a false negative, otherwise box
                    # counts as true positive
                    if nbr_boxes_containing_kp == 0:
                        img_scores["fns"] += 1
                    else:
                        img_scores["kps"] += 1 / nbr_boxes_containing_kp

                # normalize by number of kps in the image
                if len(kps) != 0:
                    img_scores["kps"] /= len(kps)
                    img_scores["fns"] /= len(kps)

                self._total_scores = {k: v + self._total_scores[k] for k, v in img_scores.items()}

        # combine boxes & kps scores to get a estimate for the true positives
        self._total_scores["tps"] = (self._total_scores["boxes"] + self._total_scores["kps"]) / 2

        # now we have TPs, FPs, and FNs and can calculate precision & recall
        precision = self._total_scores["tps"] / (self._total_scores["tps"] + self._total_scores[
            "fps"])
        recall = self._total_scores["tps"] / (self._total_scores["tps"] +
                    self._total_scores["fns"])
        f1_score = self._total_scores["tps"] / (self._total_scores["tps"] + 0.5 * (
                self._total_scores["fps"] + self._total_scores["fns"]))

        # TODO: maybe combine the scores smarter
        return {"precision": precision, "recall": recall, "f1_score": f1_score}
