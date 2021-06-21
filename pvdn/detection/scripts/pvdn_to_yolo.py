import os
import yaml
import json
import shutil
import numpy as np
from tqdm import tqdm
import argparse
import cv2
from warnings import warn


def swap_file_structure(target_dir: str, source_dir: str, img_size: int = 960):
    """
    This function will take the PVDN dataset file structure and create the file & annotation
    structure required by yolov5.
    :param target_dir: Directory where the new yolo file structure is supposed to be created.
    :param source_dir: Base directory of the original PVDN dataset.
    :param img_size: The final size of the image to be fed into the yolo network. The image will
        have to be square, so it will have the size img_size x img_size. The default value is 960.
    """
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # paranoid checks
    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"{source_dir} is not a directory. Please check.")
    if not os.path.isdir(target_dir):
        raise NotADirectoryError(f"{target_dir} is not a directory. Please check.")

    # check if the bounding box annotations have actually been created before
    if not os.path.isdir(os.path.join(source_dir, "train/labels/bounding_boxes")) \
        or not os.path.isdir(os.path.join(source_dir, "test/labels/bounding_boxes"))\
        or not os.path.isdir(os.path.join(source_dir, "val/labels/bounding_boxes")):
        raise FileNotFoundError("The bounding box annotations could not be found. "
                                "Please check if you have generated them yet. You "
                                "can generate them by using the "
                                "generate_bounding_boxes() method from the "
                                "BoundingBoxDataset class in pvdn/bboxes.py.")


    num_classes = 1
    names = ['instance']

    overview = {
        "train": os.path.join(target_dir, "train"),
        "val": os.path.join(target_dir, "val"),
        "test": os.path.join(target_dir, "test"),
        "nc": num_classes,
        "names": names
    }

    # create .yaml file required for yolo training
    yaml_dir = os.path.join(target_dir, 'pvdn.yaml')
    print(f"Creating yolo .yaml file at {yaml_dir}...")
    with open(yaml_dir, "w") as f:
        yaml.dump(overview, f, default_flow_style=None)

    # doing conversion for each split
    splits = ("train", "test", "val")
    for split in splits:

        # checking & setting up paths
        target_path = os.path.join(target_dir, split)
        source_path = os.path.join(source_dir, split)
        if not os.path.isdir(source_path):
            warn(f"{source_path} does not exist or is not a directory. Skipping the {split} split.")
            continue
        os.makedirs(target_path, exist_ok=True)

        print(f"Copying {split} images to {target_path}.")
        scenes_dir = os.path.join(source_path, "images")
        scenes = os.listdir(scenes_dir)
        for scene in tqdm(scenes, desc=f"Running through scenes of the {split} split"):
            images = os.listdir(os.path.join(scenes_dir, scene))
            for img in images:
                # resize image to be square (img_size x img_size)
                im = cv2.imread(os.path.join(scenes_dir, scene, img), 0)
                h_orig, w_orig = im.shape
                im = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA)

                # save image to new location
                cv2.imwrite(os.path.join(target_path, img), im)
                if not os.path.exists(os.path.join(target_path, img)):
                    shutil.copy(os.path.join(scenes_dir, scene, img), target_path)

                # create annotation file
                annot_file = img.split(".")[0] + ".json"
                with open(os.path.join(source_dir, split, "labels", "bounding_boxes",
                                       annot_file), "r") as f:
                    annot = json.load(f)

                annot["bounding_boxes"] = np.array(annot["bounding_boxes"])
                annot["labels"] = np.array(annot["labels"])
                deletes = np.where(annot["labels"] == 0)
                annot["bounding_boxes"] = np.delete(annot["bounding_boxes"], deletes, axis=0)
                annot["labels"] = np.delete(annot["labels"], deletes)

                yolo_file = img.split(".")[0] + ".txt"
                if os.path.exists(os.path.join(target_path, yolo_file)):
                    os.remove(os.path.join(target_path, yolo_file))
                if len(annot["labels"]) > 0:
                    with open(os.path.join(target_path, yolo_file), "w") as f:
                        for box, label in zip(annot["bounding_boxes"], annot["labels"]):
                            box = np.array(box)
                            new_box = box.copy()
                            new_box[:2] += (box[2:] - box[:2]) / 2
                            new_box[2:] -= box[:2]
                            new_box[0] /= w_orig
                            new_box[2] /= w_orig
                            new_box[1] /= h_orig
                            new_box[3] /= h_orig
                            line = [int(label)-1] + new_box.tolist()
                            line = [str(e) for e in line]
                            line = " ".join(line)
                            f.write(line)
                            f.write("\n")

    print("Finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-source_dir", type=str, help="Source dir of the EODAN dataset",
                        default="/raid/Datasets/EODAN/kaggle/day")
    parser.add_argument("-target_dir", type=str, help="Target dir of the new Yolo format "
                        "dataset.", default="/raid/Datasets/EODAN/yolo/day")
    parser.add_argument("-img_size", type=int, help="Final yolo image size (image will be square).")
    args = parser.parse_args()

    swap_file_structure(source_dir=args.source_dir, target_dir=args.target_dir)
