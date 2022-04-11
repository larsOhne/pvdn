import os
import json
import numpy as np
from tqdm import tqdm
import os
from warnings import warn
import argparse
from sklearn.metrics import precision_recall_fscore_support

from pvdn import PVDNDataset


class BoundingBoxEvaluator():
    """
    Class which helps to evaluate the results on the PVDN dataset when predicting bounding boxes.
    """

    # TODO: Add coco metric
    def __init__(self, data_dir: str):
        """
        :param data_dir: Path to dataset; must be serializable by PVDNDataset class -> str
        """
        data_dir = os.path.abspath(data_dir)
        dataset = PVDNDataset(data_dir)
        self._predictions = None
        self.gt_kps = {}
        self._nbr_images = len(dataset.img_idx)

        # indexing keypoint annotations
        for i in dataset.img_idx:
            kp_path = os.path.join(dataset.keypoints_path,
                                   "{:06d}.json".format(i))
            with open(kp_path, "r") as kp_file:
                kp_dict = json.load(kp_file)
                kps = []
                for vehicle_annot in kp_dict["annotations"]:
                    kps += [inst["pos"] for inst in vehicle_annot["instances"]]
                self.gt_kps[i] = kps

    def load_results_from_file(self, path: str) -> None:
        """
        Results are loaded from .json file with the structure:
        {"image_id":
            {
            "boxes": [nbr_boxes, 4]},
            "scores": [nbr_boxes]
            }
        }
        Each bounding box is saved as [x1, y1, x2, y2].
        Note that "scores" has to contain the probabilities predicted by the model
        in the range between 0 and 1.
        :param path: path to the .json results file.
        """
        if not os.path.splitext(path)[-1]:
            raise AttributeError(
                f"File extension should be .json not {os.path.splitext(path)[-1]}")
        with open(path, "r") as f:
            self._predictions = json.load(f)
        if len(self._predictions.items()) == 0:
            warn(
                f"Results are empty. You might want to check your results file ({path}).")

    def load_results_from_dict(self, results: dict) -> None:
        """
        Results are loaded from a dict with the structure:
        {"image_id":
            {
            "boxes": [nbr_boxes, 4]},
            "scores": [nbr_boxes]
            }
        }
        Each bounding box is saved as [x1, y1, x2, y2].
        Note that "scores" has to contain the probabilities predicted by the model
        in the range between 0 and 1.
        :param results: Dictionary containing the results.
        """
        self._predictions = results
        if len(self._predictions.items()) == 0:
            warn(
                f"Results are empty. You might want to check your results dict.")

    @staticmethod
    def _kp_in_box(kp: [np.ndarray, list], box: [np.ndarray, list],
                   tolerance: float = 0.0) \
            -> bool:
        """
        Checks if a keypoint lies within a bounding box. The height & width of the bounding box
        can be extended by a tolerance factor.
        :param kp: Keypoint which has to be checked in form of an array [x, y] -> np.ndarray
        :param box: Bounding box to be checked in form of an array [x1, y1, x2, y2] -> np.ndarray
        :param tolerance: Scale factor to extend the bounding box width and height
        :return: bool whether the keypoint lies within the bounding box or not.
        """
        tolerance /= 2
        height = (box[3] - box[1]) * tolerance
        width = (box[0] - box[2]) * tolerance
        diff = np.array([-width, -height, width, height])
        box += diff
        return box[0] <= kp[0] <= box[2] and box[1] <= kp[1] <= box[3]

    @staticmethod
    def _get_nbr_of_kps_in_box(box: [np.ndarray, list],
                               kps: [np.ndarray, list]) -> int:
        """
        Calculates the number of keypoints in a specific bounding box.
        :param box: bounding box as an array of the form [x1, y1, x2, y2] -> np.ndarray
        :param kps: all keypoints in the image as an of shape [nbr_of_kps, 2],
            where each keypoint is of the form [x, y] -> np.ndarray
        """
        if len(kps) == 0:
            return 0

        return np.count_nonzero(
            np.bitwise_and(
                np.bitwise_and(box[0] <= kps[:, 0], kps[:, 0] <= box[2]),
                np.bitwise_and(
                    box[1] <= kps[:, 1], kps[:, 1] <= box[3]))
        )

    @staticmethod
    def _get_nbr_of_boxes_containing_kp(kp: np.ndarray,
                                        boxes: np.ndarray) -> int:
        """
        Calculates the number of boxes spanning over a specific keypoint.
        :param kp: keypoint as an array of the form [x, y] -> np.ndarray
        :param boxes: bounding boxes as a np.ndarray of shape [nbr_of_boxes, 4], where each
            bounding box is of the form [x1, y1, x2, y2].
        """
        if len(boxes) == 0:
            return 0

        return np.count_nonzero(
            np.bitwise_and(
                np.bitwise_and(boxes[:, 0] <= kp[0], kp[0] <= boxes[:, 2]),
                np.bitwise_and(boxes[:, 1] <= kp[1], kp[1] <= boxes[:, 3]))
        )

    def quality_check(self, conf_thresh=0.5) -> dict:
        """
        Checks if a bounding box contains more than one keypoint and if a keypoint has an enclosing bounding box
        :return: lists of ids of the images where the check failed
        """
        if self._predictions is None:
            raise ValueError(
                "You need to load the results first by calling the "
                "load_results_from_file or load_results_from_dict function.")
        try:
            filtered_predictions = {k: np.delete(np.array(v["boxes"]),
                                                 np.where(np.array(v[
                                                                       "scores"]) <= conf_thresh),
                                                 axis=0)
                                    for k, v in self._predictions.items()}
        except:
            filtered_predictions = self._predictions.copy()
        nbr_kps_in_box_list = []
        kp_in_box_list = []

        # Prepares keypoints and bounding_boxes and calculates the number of keypoints in a bounding_box for every image
        for id, kps in tqdm(self.gt_kps.items()):
            kps = np.array(kps)

            if str(id) in filtered_predictions.keys():
                pred_boxes = np.array(filtered_predictions[str(id)])
            else:
                pred_boxes = []
            for box in pred_boxes:
                get_nbr_of_kps_in_box = self._get_nbr_of_kps_in_box(box=box,
                                                                    kps=kps)
                if get_nbr_of_kps_in_box > 1:
                    if not nbr_kps_in_box_list or nbr_kps_in_box_list[0] != id:
                        nbr_kps_in_box_list.insert(0, id)

        # Prepares keypoints and bounding boxes and returns if a keypoint has a enclosing bounding box
        for id, kps in tqdm(self.gt_kps.items()):
            kps = np.array(kps)
            for kp in kps:
                kp_in_box = False

                if str(id) in filtered_predictions.keys():
                    pred_boxes = np.array(filtered_predictions[str(id)])
                else:
                    pred_boxes = []
                for box in pred_boxes:
                    if self._kp_in_box(kp=kp, box=box):
                        kp_in_box = True
                if not kp_in_box:
                    kp_in_box_list.append(id)

        return {"nbr_kps_in_box_list": nbr_kps_in_box_list,
                "kp_in_box_list": kp_in_box_list}

    def evaluate(self, conf_thresh=0.5, verbose=False) -> dict:
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
            "qk": keypoint-based quality of bbox
            "qk_std": standard deviation of qk
            "qb": bounding box-based quality of bbox
            "qb_std": standard deviation of qb
        For a detailed explanation of the metrics please read
            https://arxiv.org/abs/2105.13236
        """
        self._total_scores = {"tps": 0, "fps": 0, "fns": 0}

        if self._predictions is None:
            raise ValueError(
                "You need to load the results first by calling the "
                "load_results_from_file or load_results_from_dict function.")
        try:
            filtered_predictions = {k: np.delete(np.array(v["boxes"]),
                                                 np.where(np.array(v[
                                                                       "scores"]) <= conf_thresh),
                                                 axis=0)
                                    for k, v in self._predictions.items()}
        except:
            filtered_predictions = self._predictions.copy()

        no_key_counter = 0
        box_quality_hist = []
        kp_quality_hist = []
        for id, kps in tqdm(self.gt_kps.items(), disable=not verbose,
                            desc="Evaluating bounding "
                                 "box metrics"):
            kps = np.array(kps)

            if str(id) in filtered_predictions.keys():
                pred_boxes = np.array(filtered_predictions[str(id)])
            else:
                pred_boxes = []
                no_key_counter += 1

            img_scores, kp_quality, box_quality = \
                self.evaluate_single_image(kps, pred_boxes)

            kp_quality_hist += kp_quality
            box_quality_hist += box_quality

            self._total_scores = {k: v + self._total_scores[k] for k, v in
                                  img_scores.items()}

        nds = 4
        # now we have TPs, FPs, and FNs and can calculate precision & recall
        try:
            precision = round(
                self._total_scores["tps"] / (self._total_scores["tps"] +
                                             self._total_scores["fps"]),
                ndigits=nds)
        except ZeroDivisionError:
            precision = -1

        try:
            recall = round(
                self._total_scores["tps"] / (self._total_scores["tps"] +
                                             self._total_scores["fns"]),
                ndigits=nds)
        except ZeroDivisionError:
            recall = -1

        try:
            f1_score = round(
                self._total_scores["tps"] / (self._total_scores["tps"] + 0.5 * (
                        self._total_scores["fps"] + self._total_scores["fns"])),
                ndigits=nds)
        except ZeroDivisionError:
            f1_score = -1

        box_quality = round(np.mean(box_quality_hist), ndigits=nds)
        box_quality_std = round(np.std(box_quality_hist), ndigits=nds)

        kp_quality = round(np.mean(kp_quality_hist), ndigits=nds)
        kp_quality_std = round(np.std(kp_quality_hist), ndigits=nds)

        box_quality_combined = round(box_quality * kp_quality, ndigits=nds)

        if verbose:
            print(
                f"Could not find {no_key_counter} of {self._nbr_images} images.")
            print("----------------------------------")
            print(f"Precision:\t\t{precision}\n"
                  f"Recall:\t\t\t{recall}\n"
                  f"F1 Score:\t\t{f1_score}\n"
                  f"Box Quality:\t\t{box_quality_combined}\n"
                  f"q_k:\t\t\t{kp_quality} +- {kp_quality_std}\n"
                  f"q_b:\t\t\t{box_quality} +- {box_quality_std}")
            print("----------------------------------")

        return {"precision": precision, "recall": recall, "f1_score": f1_score,
                "box_quality": box_quality_combined, "qk": kp_quality, "qk_std":
                    kp_quality_std, "qb": box_quality,
                "qb_std": box_quality_std}
    
    @staticmethod
    def evaluate_single_image(kps: np.ndarray, pred_boxes):
        """
        Calculates metrics for a list of keypoints and predicted bounding boxes for 
            an image.
        :param kps: numpy array of shape [num_keypoints, 2] containing the groundtruth 
            keypoint coordinates as [x, y].
        :param pred_boxes: numpy array of shape [num_boxes, 4] containing the 
            predicted bounding boxes to be evaluated as [x1, y1, x2, y2].
        :return:
            img_scores: dict containing
                "tps": (int) true positives
                "fps": (int) false positives
                "fns": (int) false negatives

            kp_quality: list of quality measures 1

            box_qualtiy: list of quality measures 2
        """
        img_scores = {"tps": 0, "fps": 0, "fns": 0}
        kp_quality, box_quality = [], []

        for box in pred_boxes:
            nbr_kps_in_box = BoundingBoxEvaluator._get_nbr_of_kps_in_box(box=box, 
                                                                         kps=kps)

            # if there is no kp in the box it is a false positive, otherwise true positive
            if nbr_kps_in_box == 0:
                img_scores["fps"] += 1
            else:
                # goal: exactly one kp in each box
                # box quality is lower the more kps there are in one box
                kp_quality.append(1 / nbr_kps_in_box)

        # second check every kp
        # penalize if one kp lies in several boxes
        for kp in kps:
            nbr_boxes_containing_kp = \
                BoundingBoxEvaluator._get_nbr_of_boxes_containing_kp(
                    kp=kp,boxes=pred_boxes
                )

            # if there is no box containing the kp it is a false negative
            if nbr_boxes_containing_kp == 0:
                img_scores["fns"] += 1
            else:
                img_scores["tps"] += 1
                # goal: each kp lies in only one box
                # quality is lower the more boxes the kp lies in
                box_quality.append(1 / nbr_boxes_containing_kp)
        
        return img_scores, kp_quality, box_quality

def evaluate_single(src, dataset_path):
    evaluator = BoundingBoxEvaluator(data_dir=dataset_path)
    evaluator.load_results_from_file(src)
    evaluator.evaluate(verbose=True)


def evaluate(evaluator: BoundingBoxEvaluator, conf_thresh: float,
             predictions: dict, verbose: bool = True) -> dict:
    """
    Calculate the metric presented in the paper using the given evaluator.
    :param evaluator: evaluator that should be used to evaluate the predictions
    :type evaluator: BoundingBoxEvaluator
    :param conf_thresh: threshold used for classification
    :type conf_thresh: float
    :param predictions: prediction dictionary
    :type predictions: dict
    :param verbose: whether to output information on screen, default True
    :type verbose: bool
    :return: dictionary containing the metrics of the paper
    :rtype: dict
    """
    evaluator.load_results_from_dict(predictions)
    metrics = evaluator.evaluate(conf_thresh=conf_thresh, verbose=verbose)
    return metrics


def evaluate_basic_metric(prediction_dict: dict, conf_thres: float,
                          verbose: bool = False) -> dict:
    """
    Evaluate the model *NOT* given the paper metric but the main definition
    of precision, recall, and f1-score.
    :param prediction_dict: results to evaluate
    :type prediction_dict: dict
    :param conf_thres: if the score stored in prediction_dict["score"] is at
    least as large as conf_thres the bounding box is classified as relevant
    :type conf_thres: float
    :param verbose: set to True, if output should be written to screen
    :type verbose: bool
    :return: dictionary whose keys are the metrics and whose values are the
    corresponding performance results
    :rtype: dict
    """
    labels_gt = None
    labels_pred = None
    for img_id, info in prediction_dict.items():
        boolean_pred = np.array(info["scores"]) >= conf_thres
        y_pred = np.array(boolean_pred, dtype=np.uint8)
        if labels_pred is None:
            labels_pred = y_pred
            labels_gt = info["labels"]
        else:
            labels_pred = np.concatenate((labels_pred, y_pred))
            labels_gt = np.concatenate((labels_gt, info["labels"]))

    metrics = compute_basic_metric(y_test=labels_gt, y_pred=labels_pred,
                                   verbose=verbose)
    metrics = {k: (v.tolist() if k.startswith("per_class") else v)
               for k, v in metrics.items()}
    return metrics


def compute_basic_metric(y_test: np.ndarray, y_pred: np.ndarray,
                         verbose: bool = True) -> dict:
    """
    Compute the basic metrics, *NOT* the paper metrics.
    :param y_test: ground truth labels
    :type y_test: np.ndarray
    :param y_pred: predicted labels
    :type y_pred: np.ndarry
    :param verbose: set to True, if output should be written to screen
    :type verbose: bool
    :return: dictionary whose keys are the metrics and whose values are the
    corresponding performance results
    :rtype: dict
    """
    micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="micro")

    macro_precision, macro_recall, macro_f1_score, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro")

    precision_per_class, recall_per_class, f1_score_per_class, _ = precision_recall_fscore_support(
        y_test,
        y_pred)

    if verbose:
        print("----------------------------------")
        print(f"Micro:\t\tPrecision: {micro_precision}\tRecall: "
              f"{micro_recall}\tF1: {micro_f1_score}\n"
              f"Macro:\t\tPrecision: {macro_precision}\tRecall: "
              f"{macro_recall}\tF1: {macro_f1_score}\n"
              f"Per class:\tPrecision: {precision_per_class}\tRecall: "
              f"{recall_per_class}\tF1: {f1_score_per_class}\n")
        print("----------------------------------")

    return {"micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1_score": micro_f1_score,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1_score": macro_f1_score,
            "per_class_precision": precision_per_class,
            "per_class_recall": recall_per_class,
            "per_class_f1_score": f1_score_per_class
            }


def find_best(src, file_pattern, dataset_path):
    dataset = PVDNDataset(path=dataset_path)
    evaluator = BoundingBoxEvaluator(dataset=dataset)

    best = {"precision": {"model": None, "value": 0},
            "recall": {"model": None, "value": 0},
            "f1_score": {"model": None, "value": 0},
            "box_quality": {"model": None, "value": 0}}

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
