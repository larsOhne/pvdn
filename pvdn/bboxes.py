from pvdn import PVDNDataset
from pvdn.detection.model.proposals import DynamicBlobDetector, Detector
from pvdn.detection.utils.misc import rescale_boxes

import os
import shutil
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


class BoundingBoxDataset(PVDNDataset):
    def __init__(self, path, filters: list = [], transform=None,
                 read_annots=True, load_images=True,
                 keypoints_path: str = None, bounding_box_path: str = None,
                 blob_detector: object = None, sup_mult: int = 2,
                 box_size: tuple = (64, 64),
                 norm_minmax: bool = True) -> None:
        """
        :param path: Path to the dataset directory of the specific illumination cycle and split.
        :param filters: List of filters to be applied to the dataset. See the filters directory
            of this package for available filters.
        :param transform: Transforms to be applied to the data.
        :param read_annots: Flag to set if the annotations to the images are supposed to be read.
        :param load_images: Flag to set if the images are supposed to be loaded.
        :param keypoints_path: Path where the keypoint annotations are stored. If the PVDN
            dataset structure is used, this parameter can be ignored and left at default None.
        :param bounding_box_path: Path where the keypoint annotations are stored. If the
            bounnding boxes have been created using the proposed script and file architecture,
            this parameter can be ignored and left at default None.
        :param blob_detector: Algorithm to be used for the region proposal part. If None,
            the blob detector proposed in the original paper is used. If you want to create your
            own, please refer to the output format used in the DynamicBlobDetector class.
        :param sup_mult: Support multiplier for extending the region around the proposed bounding
            boxes.
        :param box_size: Size to which each proposed bounding box is rescaled before fed into the
            classifier.
        :param norm_minmax: Flag to set if the pixel values in each bounding box are supposed to
            be normalized between 0 and 1.
        """
        super().__init__(path, filters, None, read_annots, load_images,
                         keypoints_path)

        self.bounding_box_path = bounding_box_path if bounding_box_path else \
            os.path.join(self.labels_path, "bounding_boxes")

        # blob detector setting dataset
        if blob_detector is not None:
            if not isinstance(blob_detector, Detector):
                raise TypeError(f"blob_detector has to be of type {Detector}.")

        self.blob_detector = blob_detector if blob_detector else \
            DynamicBlobDetector(k=0.55, w=26, padding=9, dev_thresh=0.01,
                                nms_distance=2, small_scale=None,
                                considered_region=None)
        self.sup_mult = sup_mult
        self.box_size = box_size
        self.bounding_box_idx = []
        self.bounding_boxes = []
        self.labels = []
        self.bb_transform = transform
        self.norm_minmax = norm_minmax
        self.init_bounding_boxes()

    def generate_bounding_boxes(self, verbose=False) -> None:
        """Generates the bounding boxes and stores the annotation files.
        :param verbose: Flag to show progress bar. Default false.
        """
        if not os.path.exists(self.bounding_box_path):
            os.mkdir(self.bounding_box_path)
            print(f"Created path {self.bounding_box_path}")

        for idx, id in tqdm(enumerate(self.img_idx), disable=not verbose):
            # print('idx: {}, id: {}'.format(idx, id))
            img, info, vehicles = super().__getitem__(idx)

            # extract information
            bounding_boxes = self.blob_detector.propose(np.array(img))

            if bounding_boxes:
                # extract keypoints from vehicles
                points = [inst.position for vehic in vehicles for inst in
                          vehic.instances]
                # assign labels
                labels = self.label_boxes(bounding_boxes, points)
                labels = list(np.array(labels).astype(float))
            else:
                labels = []

            annotation_path = os.path.join(self.bounding_box_path,
                                           "{:06d}.json".format(id))

            # write annotations
            with open(annotation_path, 'w') as f:
                annotation = {'bounding_boxes': bounding_boxes,
                              'labels': labels}
                json.dump(annotation, f)

    def plot_scenes(self, scene_ids: list, preds: dict, output_dir: str,
                    conf_thresh: float = 0.5):
        """
        TODO: function & parameter description
        """
        if not isinstance(scene_ids, list):
            scene_ids = [int(scene_ids)]
        else:
            scene_ids = [int(i) for i in scene_ids]

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for scene in self.sequences:
            if scene.sid in scene_ids:
                scene_out_dir = os.path.join(output_dir, scene.directory)
                os.mkdir(scene_out_dir)
                scene_ids.remove(scene.sid)
                for img_id in scene.image_ids:
                    img_path = os.path.join(self.images_path, scene.directory,
                                            f"{img_id:06d}.png")
                    # load image as grayscale
                    img = cv2.imread(img_path)
                    try:
                        bboxes = np.array(preds[str(img_id)]["boxes"])
                        scores = np.array(preds[str(img_id)]["scores"])
                        bboxes = np.delete(bboxes,
                                           np.where(scores < conf_thresh),
                                           axis=0)
                        bboxes = bboxes.reshape(-1, 4)
                        # plot the boxes
                        img = self.plot_bboxes(img, bboxes)
                    except KeyError:
                        pass
                    # save
                    cv2.imwrite(
                        os.path.join(scene_out_dir, f"{img_id:06d}.png"), img)

    def plot_bboxes(self, img, bboxes):
        """
        TODO: function & parameter description
        """
        bboxes = np.array(bboxes, dtype=int)
        for bbox in bboxes:
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                color=(0, 255, 0),
                                thickness=3)
        return img

    def plot(self, idx, extract=True, path=None) -> list:
        """
        Plots the results for a given index.
        TODO: Do not use this function yet!!
        TODO: Provide parameter description
        :param idx: index of the image to be plotted -> int
        :param extract:
        :param path:
        :param external_img:
        :return:
        """

        img, info, vehicles = PVDNDataset.__getitem__(self, idx)

        path = self.bounding_box_path if path is None else path
        points = points = [inst.position for vehic in vehicles for inst in
                           vehic.instances]
        # extract information
        if extract:
            bounding_boxes = self.blob_detector.propose(img)
            labels = self.label_boxes(bounding_boxes, points)
        else:
            i = 0
            try:
                start_idx = self.bounding_box_idx.index((idx, 0))
                # search for all bounding_boxes of the image
                while self.bounding_box_idx[start_idx][0] == \
                        self.bounding_box_idx[start_idx + i][0]:
                    i += 1
                bounding_boxes = self.bounding_boxes[start_idx:start_idx + i]
                labels = self.labels[start_idx:start_idx + i]
            except ValueError:
                bounding_boxes = []
                labels = []

        # plot
        plt.clf()
        plt.figure(dpi=300)
        plt.axis('off')
        ax = plt.gca()

        for point in points:
            ax.add_patch(plt.Circle(tuple(point), radius=2))

        img = np.array(img) / 255.
        img_min = np.min(img)
        img_max = np.max(img)
        img = (img - img_min) / (img_max - img_min)
        img = (img * 255).astype('uint8')

        # plot_with_bbs(img, bbs=bounding_boxes,
        #               plotter=ax, bb_labels=labels)
        plt.savefig(os.path.join(path, "{:06d}.png".format(idx)))

        plt.close()

        # test that dict structure is correct
        for key, value in enumerate(self.bounding_box_idx):
            if value[0] == idx:
                print("bounding box {} at position {} with label {}".format(
                    value[1], self.bounding_boxes[key], self.labels[key]))

        return bounding_boxes

    @staticmethod
    def label_boxes(boxes, kps) -> list:
        """
        Evaluates the bounding box labels based on the key points.
        :param boxes: array of bounding boxes of shape [nbr_boxes, 4] -> list
        :param kps: array of keypoints of shape [nbr_kps, 2] -> list
        :return: array of labels per bounding box of shape [nbr_boxes] -> list
        """
        boxes = np.array(boxes)
        points = np.array(kps)
        labels = np.zeros((boxes.shape[0],)).astype(bool)

        if len(boxes) != 0:
            for point in points:
                left = boxes[:, 0] <= point[0]
                right = point[0] <= boxes[:, 2]
                top = boxes[:, 1] <= point[1]
                bottom = point[1] <= boxes[:, 3]
                labels = np.logical_or(
                    labels,
                    np.logical_and(
                        np.logical_and(
                            np.logical_and(left, right), top), bottom))

        return labels

    def init_bounding_boxes(self) -> None:
        try:
            for img_idx, id in enumerate(self.img_idx):
                # loading box annotation file
                annotation_path = os.path.join(self.bounding_box_path,
                                               "{:06d}.json".format(id))
                with open(annotation_path, 'r') as f:
                    annotations = json.load(f)

                for bb_idx in range(len(annotations['labels'])):
                    # key is the continuous index starting at 0 that indexes
                    #    all bounding boxes
                    # img_idx is the continuous index starting at 0 that
                    #    refers to the corresponding image;
                    #    self.img_idx[img_idx] returns the id of the image
                    # bb_idx is the continuous index starting at 0 to the
                    #    bounding box id in the corresponding image
                    self.bounding_box_idx.append((img_idx, bb_idx))
                    self.bounding_boxes.append(
                        annotations['bounding_boxes'][bb_idx])
                    self.labels.append(annotations['labels'][bb_idx])

        except FileNotFoundError:
            print('Annotation(s) file(s) not found. Call '
                  '`generate_bounding_boxes` to extract the bounding box '
                  'information. Until that, not all class methods are '
                  'available (e.g., `__get_item__()` and `__len()__`). If '
                  'bounding boxes are available, call '
                  '`init_bounding_boxes` to re-initialize the object '
                  'structure.')

    def _crop_image(self, img: np.ndarray, bb_coords: np.ndarray) -> np.ndarray:
        """
        Crop a sub image from the given image specified by the given
        coordinates. The support multiplier is used to increase the cropped
        area.
        :param img: the subimage should be cropped from this image
        :type img: numpy.ndarray with shape (w,h)
        :param bb_coords: coordinates [x1, y1, x2, y2] of the subimage
        :type bb_coords: numpy.ndarray with shape (4,)
        :return: the cropped image
        :rtype: numpy.ndarray
        """
        h, w = img.shape

        bb_w, bb_h = bb_coords[2] - bb_coords[0], bb_coords[3] - bb_coords[1]

        center_x = bb_coords[0] + (bb_coords[2] - bb_coords[0]) // 2
        center_y = bb_coords[1] + (bb_coords[3] - bb_coords[1]) // 2

        # increase cropping area by support multiplier
        crop_bb = [max(center_x - bb_w * self.sup_mult, 0),
                   max(center_y - bb_h * self.sup_mult, 0),
                   min(center_x + bb_w * self.sup_mult, w),
                   min(center_y + bb_h * self.sup_mult, h)]
        crop_bb = np.array(crop_bb).astype('uint32')
        cropped_bb = img[crop_bb[1]:crop_bb[3], crop_bb[0]:crop_bb[2]]
        return cropped_bb

    def __getitem__(self, idx):
        """
        TODO: param description
        :param idx:
        :return: float img between 0-1
        """
        # load the data
        img_idx, bb_idx = self.bounding_box_idx[idx]
        img, info, vehicles = super().__getitem__(img_idx)
        bb_coords = np.array(self.bounding_boxes[idx])
        label = self.labels[idx]

        cropped_bbox = self._crop_image(img, bb_coords)

        # resize cropped box to constant size
        feature = cv2.resize(cropped_bbox,
                             self.box_size, interpolation=cv2.INTER_LINEAR)
        feature = np.array(feature).astype('float32') / 255
        if self.norm_minmax:
            cv2.normalize(src=feature, dst=feature, norm_type=cv2.NORM_MINMAX)
            feature = np.clip(feature, a_min=0.0, a_max=1.0)
        feature = np.expand_dims(feature, -1)

        # apply transforms if neccessary
        if self.bb_transform is not None:
            if isinstance(self.bb_transform, list):
                for trans in self.bb_transform:
                    feature = trans(feature)
            else:
                feature = self.bb_transform(feature)

        max_intensity = feature.max()
        min_intensity = feature.min()
        difference = np.maximum(max_intensity - min_intensity, 1.e-12)
        feature = (feature - min_intensity) / difference
        return feature, label, bb_coords, info.id

    def __len__(self):
        return len(self.labels)

    def load_image(self, idx):
        """
        TODO: Provide function & parameter description.
        :param index:
        :return:
        """
        # retrieve metadata
        info = self.img_infos[self.img_idx[idx]]
        sequence_info = info.sequence
        image_path = os.path.join(self.images_path, sequence_info.directory,
                                  info.file_name)
        # load actual image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return img, self.bounding_boxes[idx], self.labels[idx]

    def compute_class_imbalance(self):
        true = np.sum(self.labels).astype(np.u)
        false = len(self.labels) - true
        return {"true": true, "false": false}
