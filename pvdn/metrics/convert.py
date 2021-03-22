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
        output_dict[str(id)] = {}
        output_dict[str(id)]["boxes"] = boxes.tolist()
        output_dict[str(id)]["scores"] = labels.tolist()

    return output_dict


def coco_to_results_format(coco_path: str, output_path: str = None,
                           coco_img_size: tuple = (960, 960)) -> None:
    """
    Converts the results saved in the coco detection results format to the format used in the
    custom bounding box evaluation metric.
    :param coco_path: Path of the coco .json file -> str
    :param output_path: Path under which the final results file has to be saved. If no path is
        provided, the result format is returned as a dictionary. -> str
    :param coco_img_size: Size of the images which were used to generate the results in the coco
        file. -> tuple
    :return: None, if the output path is specified. If the output path is not specified, the
        result format is returned as a dictionary.
    """
    if output_path:
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
    pvdn_format = {}
    fault_counter = True
    h_orig, w_orig = coco_img_size
    for pred in tqdm(preds):
        fault_counter = False
        k = str(pred["image_id"])
        bbox = np.array(pred["bbox"])
        bbox[2:] += bbox[:2]
        bbox[0] = (bbox[0] / w_orig) * 1280
        bbox[2] = (bbox[2] / w_orig) * 1280
        bbox[1] = (bbox[1] / h_orig) * 960
        bbox[3] = (bbox[3] / h_orig) * 960
        if k in pvdn_format.keys():
            pvdn_format[k]["boxes"].append(bbox.tolist())
            pvdn_format[k]["scores"].append(pred["score"])
        else:
            pvdn_format[k] = {"boxes": [bbox.tolist()], "scores": [pred["score"]]}

    if fault_counter:
        warn(f"There were no predictions found in the provided file {coco_path}.")

    if output_path:
        with open(os.path.abspath(output_path), "w") as f:
            json.dump(pvdn_format, f)
    else:
        return pvdn_format


def result_to_coco_format(results: dict) -> list:
    """
    TODO: provide parameter description
    """
    coco_preds = []
    for id, items in results.items():
        for bbox, score in zip(items["boxes"], items["scores"]):
            bbox = np.array(bbox)
            bbox[2:] = bbox[2:] - bbox[:2]
            coco_preds.append({"image_id": id, "category_id": 0,
                               "bbox": bbox.tolist(), "score": score})
    return coco_preds


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

    # out = gt_to_results_format(gt_bbox_dir="/media/lukas/empty/EODAN_Dataset/day/test/labels"
    #                                 "/bounding_boxes")
    # with open("/media/lukas/empty/EODAN_Dataset/results/gt_test_results.json", "w") as f:
    #     json.dump(out, f)

    preds = "/media/lukas/empty/EODAN_Dataset/results/yolov5x/test/119_predictions.json"
    coco_to_results_format(preds,
                           "/media/lukas/empty/EODAN_Dataset/results/yolov5x/test/119_eodan.json")