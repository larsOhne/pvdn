import json
import numpy as np
from tqdm import tqdm
import os
from warnings import warn


def gt_to_results_format(gt_bbox_dir):
    """
    Converts the ground truth bounding box annotation files to the format used for the custom
    evaluation metric.
    :param gt_bbox_dir: Directory path of the stored bounding box annotation files -> str
    :return: Dictionary containing the bounding boxes in the custom format, where each key is the
        unique image id containing a list of bounding boxes of the shape [nbr_boxes, 4]
    """
    path = os.path.abspath(gt_bbox_dir)
    box_files = os.listdir(path)
    output_dict = {}
    for file in tqdm(box_files):
        id = file.split(".")[0].lstrip("0")
        with open(os.path.join(path, file), "r") as f:
            annots = json.load(f)
        boxes = np.array(annots["bounding_boxes"])
        labels = np.array(annots["labels"])
        boxes = np.delete(boxes, np.where(labels == 0), axis=0)

        output_dict[str(id)] = boxes.tolist()
    return output_dict


def coco_to_results_format(coco_path: str, output_path: str, conf_thresh: float = 0.5):
    """
    Converts the results saved in the coco detection results format to the format used in the
    custom bounding box evaluation metric.
    :param coco_path: Path of the coco .json file -> str
    :param output_path: Path under which the final results file has to be saved. -> str
    :param conf_thresh: Confidence threshold at which a prediction is supposed to be interpreted
    as valid -> float
    """
    if not os.path.splitext(output_path)[-1] == ".json":
        raise AttributeError(f"The output path extension should be .json, not "
                             f"{os.path.splitext(output_path)[-1]}.")
    if not os.path.isfile(coco_path):
        raise FileNotFoundError(f"File {coco_path} does not exist.")
    if not os.path.splitext(coco_path)[-1] == ".json":
        raise AttributeError(f"The coco prediction file should have the extension .json, not "
                             f"{os.path.splitext(coco_path)[-1]}.")

    coco_path = os.path.abspath(coco_path)
    with open(coco_path, "r") as f:
        preds = json.load(f)
    eodan_format = {}
    fault_counter = True
    for pred in tqdm(preds):
        fault_counter = False
        k = pred["image_id"]
        if pred["score"] > conf_thresh:
            bbox = np.array(pred["bbox"])
            bbox[2:] += bbox[:2]
            bbox[0] = (bbox[0] / 960) * 1280
            bbox[2] = (bbox[2] / 960) * 1280
            if k in eodan_format.keys():
                eodan_format[k].append(bbox.tolist())
            else:
                eodan_format[k] = [bbox.tolist()]
        else:
            if not k in eodan_format.keys():
                eodan_format[k] = []

    if fault_counter:
        warn(f"There were no predictions found in the provided file {coco_path}.")

    with open(os.path.abspath(output_path), "w") as f:
        json.dump(eodan_format, f)


if __name__ == "__main__":
    # data_dir = "/media/lukas/empty/EODAN_Dataset/results/custom/predictions"
    # print(f"Converting for data in {data_dir}...")
    # for yolo_file in os.listdir(data_dir):
    #     if ".json" in yolo_file:
    #         data_path = os.path.join(data_dir, yolo_file)
    #         epoch = yolo_file.split(".")[0]
    #         output_path = os.path.join(data_dir, f"{epoch}_eodan.json")
    #         yolo_to_eodan(data_path, output_path)
    # print("Done!")

    out = gt_to_results_format(path="/media/lukas/empty/EODAN_Dataset/day/test/labels"
                                    "/bounding_boxes")
    with open("gt_test_results.json", "w") as f:
        json.dump(out, f)
