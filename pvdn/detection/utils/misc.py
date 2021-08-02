import numpy as np
from typing import Union


def rescale_boxes(boxes, scale_x, scale_y):
    return [[box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y] for box in boxes]


def crop_bboxes(img: np.ndarray, bboxes: Union[list, np.ndarray]) -> list:
    """
    Crops several bounding boxes out of an image.
    :param img: image of shape [h, w]
    :param bboxes: array of bounding boxes of shape [n_boxes, 4] where each box is
        like [x1, y1, x2, y2]
    :return: list (same length as bboxes) of bounding boxes cropped out of the
        image, where each element is a numpy.ndarray of the shape as specified by
        the bounding box coordinates.
    """
    features = []
    for box in bboxes:
        features.append(
            img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        )
    return features
