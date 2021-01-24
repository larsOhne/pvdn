# system libraries
import os
import json
import shutil
from copy import deepcopy
from datetime import datetime

# Typing
from enum import IntEnum
from typing import Tuple, List, Any

# pytorch
from torch.utils.data import Dataset

# image loading
import cv2

# local imports
from pvdn.meta import CameraConfiguration, Category, ImageInformation, Annotation, SequenceInformation
from pvdn.keypoints import Vehicle

class PVDNDataset(Dataset):
    """ Base-Class to read and handle the PVDN dataset.

    The class expects the dataset in the same format as specified here:
    https://www.kaggle.com/saralajew/provident-vehicle-detection-at-night-pvdn
    """

    def __init__(self, path: str, filters: List[Any] = [], transform: List[Any] = None,
                 read_annots: bool = True, load_images: bool = True, keypoints_path: str = None):
        super(PVDNDataset, self).__init__()

        # init not affected instance vars
        self.load_images = load_images

        # check if all required folders exist and create paths for them
        self.filters = filters
        self.base_path = path
        self.images_path = os.path.join(path, "images")
        self.labels_path = os.path.join(path, "labels")
        assert os.path.exists(path) and os.path.isdir(path)
        assert os.path.exists(self.images_path)
        assert os.path.exists(self.labels_path)

        # setup image transforms
        self.transform = transform

        # read sequence-data
        self.sequences_file = os.path.join(self.labels_path, "sequences.json")
        with open(self.sequences_file, "r") as s_file:
            seqs_dict = json.load(s_file)

        self.sequences = [SequenceInformation(**seq_dict) for seq_dict in seqs_dict["sequences"]]

        # read image annotations
        self.annotation_path = os.path.join(self.labels_path, "image_annotations.json")
        with open(self.annotation_path, "r") as a_file:
            annots_dict = json.load(a_file)

        # add path for keypoints
        self.keypoints_path = keypoints_path if keypoints_path else os.path.join(self.labels_path, "keypoints")

        assert os.path.exists(self.keypoints_path), "This dataset contains no keypoints"

        # parse dataset information
        self.info = annots_dict["info"]

        # parse licenses
        self.licences = annots_dict["licences"]

        # get all camera configurations
        self.cam_configs = [CameraConfiguration(cid=i, **cam_dict)
                            for i, cam_dict in enumerate(annots_dict["camera_configurations"])]

        # get all possible categories
        self.categories = [Category(**cat_dict) for cat_dict in annots_dict["categories"]]

        # read all image definitions and store them in list
        self.img_infos = {}
        for image_dict in annots_dict["images"]:
            camera_config = self.cam_configs[image_dict["camera_configuration"]]
            image_info = ImageInformation(camera_config=camera_config, **image_dict)
            self.img_infos[image_info.id] = image_info

        # create an additional array to hold all index mappings
        self.img_idx = list(self.img_infos.keys())

        # add annotations to all images
        if read_annots:
            for annot_dict in annots_dict["annotations"]:
                image_info = self.img_infos[annot_dict["image_id"]]
                annot = Annotation(**annot_dict)
                image_info.add_annotation(annot)

        # assign sequence to images
        for sequence in self.sequences:
            for image_id in sequence.image_ids:
                image_info = self.img_infos[image_id]
                image_info.sequence = sequence

        # filter images
        if len(filters) > 0:
            for image_id in self.img_idx:
                image_info = self.img_infos[image_id]
                valid = True
                for f in self.filters:
                    valid = valid and f.evaluate(image_info)
                    if not valid:
                        break
                if not valid:
                    image_info.sequence.image_ids.remove(image_id)
                    del self.img_infos[image_id]
                else:
                    # make the sequence of image used
                    image_info.sequence.used = True

            self.img_idx = list(self.img_infos.keys())
        else:  # still set all sequences to "used"
            for seq in self.sequences:
                seq.used = True

    def print_summary(self):
        """
        prints dataset statistics to console
        :return:
        """
        print("########################################")
        print("Summary for PVDN dataset:")
        print("########################################")

        for key in self.info:
            print("{}:\t {}".format(key, self.info[key]))

        print("\n\n")
        print("total number of images: \t {}".format(len(self)))

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, idx) -> Tuple[Any, ImageInformation, List[Vehicle]]:

        # retrieve metadata
        info = self.img_infos[self.img_idx[idx]]
        sequence_info = info.sequence

        if self.load_images:
            image_path = os.path.join(self.images_path, sequence_info.directory, info.file_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if self.transform is not None:
                img = self.transform.transform(img)
        else:
            img = None

        # determine path where keypoints should be stored
        kp_path = os.path.join(self.keypoints_path, "{:06d}.json".format(info.id))

        # check if keypoints are available and read them if necessary
        vehicles = []
        kp_dict = None
        if os.path.exists(kp_path):
            # read file
            with open(kp_path, "r") as kp_file:
                kp_dict = json.load(kp_file)

                # iterate over all vehicles
                for vehicle_annot in kp_dict["annotations"]:
                    vehicle = Vehicle()
                    vehicle.from_dict(vehicle_annot)
                    vehicles.append(vehicle)

        return img, info, vehicles
