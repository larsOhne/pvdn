import numpy as np
from copy import deepcopy
import cv2


def normalized_intersection(bbox1, bbox2):
    """
    Intersection that is normalized with respect to the first bbox.
    :param bbox1: bounding boxes in the format [left, top, right, bottom]
        and shape [4]; dtype has to be np.ndarray or list
    :param bbox2: bounding boxes in the format [left, top, right, bottom]
        and shape [4]; dtype has to be np.ndarray or list
    :return: normalized intersection value
    """

    assert bbox1[0] < bbox1[2]
    assert bbox1[1] < bbox1[3]
    assert bbox2[0] < bbox2[2]
    assert bbox2[1] < bbox2[3]

    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area
    bb1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

    i_value = intersection_area / float(bb1_area)

    assert i_value >= 0.0
    assert i_value <= 1.0

    return i_value


def bbox_xywh_to_ltrb(bbox_xywh):
    """Converts a xywh bbox to a ltrb bbox.

    :param bbox_xywh: bounding boxes in the format [center_x, center_y, width,
        height] and shape [4]; dtype has to be np.ndarray or list
    :return: bbox in the format [left, top, right, bottom]; dtype np.ndarray
        of shape [4]
    """

    l = bbox_xywh[0] - bbox_xywh[2] / 2
    t = bbox_xywh[1] - bbox_xywh[3] / 2
    r = bbox_xywh[0] + bbox_xywh[2] / 2
    b = bbox_xywh[1] + bbox_xywh[3] / 2

    return np.array([l, t, r, b])


def bbox_ltrb_to_xywh(bbox_ltrb):
    """Converts a ltrb bbox to a xywh bbox.

    :param bbox_ltrb: bounding boxes in the format [left, top, right, bottom]
        and shape [4]; dtype has to be np.ndarray or list
    :return: bbox in the format [center_x, center_y, width, height];
        dtype np.ndarray of shape [4]
    """

    x = 0.5 * (bbox_ltrb[0] + bbox_ltrb[2])
    y = 0.5 * (bbox_ltrb[1] + bbox_ltrb[3])
    w = bbox_ltrb[2] - bbox_ltrb[0]
    h = bbox_ltrb[3] - bbox_ltrb[1]

    return np.array([x, y, w, h])


class Tracker(object):
    """Tracks bboxes based on a normalized intersection measure. The
        position of tracked objects is estimated by a alpha-beta filter 
        (momentum method). Predictions are smoothed by a moving average. The 
        tracker deals with occlusions.
    """
    def __init__(self, intersection_threshold=0.1, padding_factor=1.5,
                 max_count_lost=3, prediction_threshold=0.5,
                 visibility_threshold=5, alpha_bbox_center=0.5,
                 alpha_bbox_size=0.3, alpha_prediction=0.33,
                 alpha_distance=0.8, box_area=(0, 0, 1279, 959),
                 min_prediction=0.1):
        """
        :param intersection_threshold: threshold after which objects between
            different frames will be considered the same [in pixels]
        :param padding_factor: padding that will be applied to the bboxes
            before intersection computation
        :param max_count_lost: maximal number that a object can be occluded
            before it is removed (e.g., 3 means it can be occluded three times)
        :param prediction_threshold: threshold before an object is
            considered to be of true class
        :param visibility_threshold: objects with an visibility of at least
            this threshold can be focus objects
        :param alpha_bbox_center: learning rate for the bbox center (like
            gradient descent)
        :param alpha_bbox_size: learning rate for the bbox size (like
            gradient descent)
        :param alpha_prediction: alpha of moving mean to smooth the prediction
        :param alpha_distance: learning rate for the distance (like gradient
            descent)
        :param box_area: region in which bboxes are plausible (image size)
        :param min_prediction: minimal prediction score of an object before
            it is considered for tracking
        """
        if padding_factor <= 0:
            raise ValueError("padding_factor must be greater than zero. "
                             "You provided {}.".format(padding_factor))

        if not (intersection_threshold >= 0 and intersection_threshold <= 1):
            raise ValueError("intersection_threshold must be an element of "
                             "[0, 1]. "
                             "You provided {}.".format(intersection_threshold))

        if not (alpha_bbox_center >= 0 and alpha_bbox_center <= 1):
            raise ValueError("alpha_bbox_center must be an element of [0, 1]. "
                             "You provided {}.".format(alpha_bbox_center))

        if not (alpha_bbox_size >= 0 and alpha_bbox_size <= 1):
            raise ValueError("alpha_bbox_size must be an element of [0, 1]. "
                             "You provided {}.".format(alpha_bbox_size))

        if not (alpha_prediction >= 0 and alpha_prediction <= 1):
            raise ValueError("alpha_prediction must be an element of [0, 1]. "
                             "You provided {}.".format(alpha_prediction))

        if not (alpha_distance >= 0 and alpha_distance <= 1):
            raise ValueError("alpha_distance must be an element of [0, 1]. "
                             "You provided {}.".format(alpha_distance))

        self.tracked_objects = []
        self.intersection_threshold = intersection_threshold
        self.padding_factor = padding_factor
        self.max_count_lost = max_count_lost
        self.prediction_threshold = prediction_threshold
        self.visibility_threshold = visibility_threshold
        self.alpha_bbox_center = alpha_bbox_center
        self.alpha_bbox_size = alpha_bbox_size
        self.alpha_prediction = alpha_prediction
        self.alpha_distance = alpha_distance
        self.box_area = box_area
        self.min_prediction = min_prediction

    def __call__(self, bboxes, predictions, distances=None):
        """Basic method to trigger the tracker. Pass the objects detected at
        time t to this method.

        :param bboxes: bounding boxes in the format [left, top, right,
            bottom] and shape [#boxes, 4]; dtype can be list or np.ndarray
        :param predictions: confidence scores in the shape [#boxes]; dtype can
            be list or np.ndarray
        :param distances: distances in the shape [#boxes]; dtype can
            be list or np.ndarray (input is optional!)
        :return: None if there is no object that has the focus and the focus
            object otherwise (the focus object is the object that fulfills
            the criteria to be forwarded to the MxB).
        """
        # type checks and conversions
        if type(bboxes) not in (list, np.ndarray):
            raise TypeError(
                "The type of the bboxes must be list or numpy.ndarry. "
                "Your input was {}.".format(type(bboxes)))

        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)

        if type(predictions) not in (list, np.ndarray):
            raise TypeError(
                "The type of the predictions must be list or numpy.ndarry. "
                "Your input was {}.".format(type(predictions)))

        if isinstance(predictions, list):
            predictions = np.array(predictions)

        if distances is not None:
            if type(distances) not in (list, np.ndarray):
                raise TypeError(
                    "The type of the distances must be list or numpy.ndarry. "
                    "Your input was {}.".format(type(distances)))

            if isinstance(distances, list):
                distances = np.array(distances)

        # shape checks
        if len(bboxes.shape) != 2:
            raise ValueError(
                "The shape of the bboxes must be [nbr_of_bboxes, 4]. "
                "Your input was {}.".format(bboxes.shape))

        if bboxes.shape[1] != 4:
            raise ValueError(
                "The shape of the bboxes must be [nbr_of_bboxes, 4]. "
                "Your input was {}.".format(bboxes.shape))

        if len(predictions.shape) != 1:
            # print(predictions)
            raise ValueError(
                "The shape of the predictions must be [nbr_of_bboxes]. "
                "Your input was {}.".format(predictions.shape))

        if distances is not None:
            if len(distances.shape) != 1:
                # print(predictions)
                raise ValueError(
                    "The shape of the distances must be [nbr_of_bboxes]. "
                    "Your input was {}.".format(distances.shape))

        if distances is not None:
            if not len(distances) == len(predictions) == bboxes.shape[0]:
                raise ValueError(
                    "The number of object entries must be the same for all "
                    "attributes: "
                    "'len(distances) == len(predictions) == bboxes.shape[0]'. "
                    "You passed len(distances)={}, len(predictions)={}, and "
                    "bboxes.shape[0]={}". format(len(distances),
                                                 len(predictions),
                                                 bboxes.shape[0]))
        else:
            if not len(predictions) == bboxes.shape[0]:
                raise ValueError(
                    "The number of object entries must be the same for all "
                    "attributes: "
                    "'len(predictions) == bboxes.shape[0]'. "
                    "You passed len(predictions)={} and "
                    "bboxes.shape[0]={}". format(len(predictions),
                                                 bboxes.shape[0]))

        tracked_objects, tracked_obj_is_found = \
            self.register_objects(bboxes, predictions, distances)

        occluded_objects = self.predict_occluded_objects(tracked_obj_is_found)

        self.tracked_objects = tracked_objects + occluded_objects

        self.tracked_objects = self.remove_lost_objects()

        return self.determine_focus_object()

    def register_objects(self, bboxes, predictions, distances):
        """Matches the given objects at time t with the previous objects
        at time t-1.

        :param bboxes: bounding boxes in the format [left, top, right,
            bottom] and shape [#boxes, 4]; dtype has to be np.ndarray
        :param predictions: confidence scores in the shape [#boxes]; dtype has
            to be np.ndarray
        :param distances: None or distances in the shape [#boxes]; dtype has
            to be np.ndarray
        :return: two arguments:
            1: object list of tracked and new objects,
            2: set of indices to objects at time t-1 that had a match with
            objects at time t.
        """
        new_objects = []
        tracked_obj_is_found = set()  # set to store indices of tracked objects

        for i, (bbox, prediction) in \
                enumerate(zip(bboxes, predictions)):
            if prediction >= self.min_prediction:
                if distances is None:
                    distance = None
                else:
                    distance = distances[i]

                # always new object if tracked objects is empty
                if len(self.tracked_objects) == 0:
                    new_objects.append(TrackedObject(
                        bbox, prediction, distance,
                        alpha_bbox_center=self.alpha_bbox_center,
                        alpha_bbox_size=self.alpha_bbox_size,
                        alpha_distance=self.alpha_distance,
                        alpha_prediction=self.alpha_prediction,
                        box_area=self.box_area))

                # determine neighbors (matches) for each bbox in the  previous
                # time step;
                else:
                    neighbors = []
                    i_values = []
                    for j, obj in enumerate(self.tracked_objects):

                        # increase predicted bbox size (!) by padding
                        padded_bbox = self.bbox_padding(deepcopy(bbox))

                        # Use future position of tracked objects for
                        # intersection computation;
                        # A value of one means the future bbox is
                        # fully covered by the padded box.
                        i_value = normalized_intersection(obj.future_bbox,
                                                          padded_bbox)
                        if i_value >= self.intersection_threshold:
                            neighbors.append(j)
                            i_values.append(i_value)

                    # new object
                    if len(neighbors) == 0:
                        new_objects.append(TrackedObject(
                            bbox, prediction, distance,
                            alpha_bbox_center=self.alpha_bbox_center,
                            alpha_bbox_size=self.alpha_bbox_size,
                            alpha_distance=self.alpha_distance,
                            alpha_prediction=self.alpha_prediction,
                            box_area=self.box_area))

                    # unique match predict next position
                    elif len(neighbors) == 1:
                        tracked_obj_is_found.add(neighbors[0])
                        obj = deepcopy(self.tracked_objects[neighbors[0]])
                        new_objects.append(obj(bbox, prediction, distance))

                    # if a box has more than one neighbor the boxes, we
                    # inherit the history of the box with the greatest i_value
                    else:
                        neighbor = self.determine_winning_object(neighbors,
                                                                 i_values)
                        tracked_obj_is_found.add(neighbor)
                        obj = deepcopy(self.tracked_objects[neighbor])
                        new_objects.append(obj(bbox, prediction, distance))

        return new_objects, tracked_obj_is_found

    def determine_winning_object(self, candidates, i_values):
        """Determine the winning object based on the ivalue. If both are
        equally good, the history counts. After that the confidence.

        :param candidates: indices to objects in the tracked object list
            that are candidates for the best match
        :param i_values: intersection values between the candidates and the
            bbox in question
        :return: index to the best candidate in the tracked object list
        """

        best_prediction = -1
        best_count_visibility = 0
        best_candidate = None
        best_i = -1

        for j, i in enumerate(candidates):
            count_visibility = self.tracked_objects[i].count_visibility
            prediction = self.tracked_objects[i].prediction
            i_value = i_values[j]
            if i_value > best_i:
                best_candidate = i
                best_i = i_value
                best_prediction = prediction
                best_count_visibility = count_visibility

            elif i_value == best_i:
                if count_visibility > best_count_visibility:
                    best_candidate = i
                    best_i = i_value
                    best_prediction = prediction
                    best_count_visibility = count_visibility

                elif count_visibility == best_count_visibility:
                    if prediction >= best_prediction:
                        best_candidate = i
                        best_i = i_value
                        best_prediction = prediction
                        best_count_visibility = count_visibility

        return best_candidate

    def remove_lost_objects(self):
        """Remove objects that are older than the max_count_lost.

        :return: cleaned object list
        """

        new_objects = []

        for obj in self.tracked_objects:
            if obj.count_lost <= self.max_count_lost:
                new_objects.append(obj)

        return new_objects

    def predict_occluded_objects(self, tracked_obj_is_found):
        """Predict the position of occluded objects.

        :param tracked_obj_is_found: set of indices of objects that have
            been tracked at current time step
        :return: object list with predicted objects
        """

        new_objects = []

        for i, obj in enumerate(self.tracked_objects):
            if i not in tracked_obj_is_found:
                # call default update of object to predict future position
                new_objects.append(obj())

        return new_objects

    def determine_focus_object(self):
        """The object that has the focus is the oldest object with a
        prediction above the threshold. If several objects have the same
        age, the highest prediction counts and after that first in first
        out.

        :return: object that has the focus"""

        focus_object = None
        best_count_visibility = 0
        best_candidate_prediction = -1

        for obj in reversed(self.tracked_objects):
            if obj.prediction > self.prediction_threshold:
                if obj.count_visibility > best_count_visibility:
                    focus_object = obj
                    best_count_visibility = focus_object.count_visibility
                    best_candidate_prediction = focus_object.prediction

                elif obj.count_visibility == best_count_visibility:
                    if obj.prediction > best_candidate_prediction:
                        focus_object = obj
                        best_count_visibility = \
                            focus_object.count_visibility
                        best_candidate_prediction = \
                            focus_object.prediction

        if focus_object is not None:
            if focus_object.count_visibility < self.visibility_threshold:
                focus_object = None

        return focus_object

    def reset(self):
        """Reset the object tracker.
        """
        self.tracked_objects = []

    def bbox_padding(self, bbox):
        """Applies padding to the given bbox by the padding factor.

        :param bbox: bounding boxes in the format [left, top, right,
            bottom] and shape [4]; dtype has to be np.ndarray
        :return: padded bbox
        """
        if self.padding_factor != 1:
            xywh = bbox_ltrb_to_xywh(bbox)
            xywh[2:] = xywh[2:] * self.padding_factor

            bbox = bbox_xywh_to_ltrb(xywh)

            bbox[0] = max(bbox[0], self.box_area[0])
            bbox[1] = max(bbox[1], self.box_area[1])
            bbox[2] = min(bbox[2], self.box_area[2])
            bbox[3] = min(bbox[3], self.box_area[3])

        return bbox

    def plot_tracked_objects(self, img):
        """Plots the tracked objects into the given image. The brightness of
        objects increases with their age and decreases with their los count.
        Objects are colored red if they are below the prediction threshold
        and green if they are above. Bboxes are tiled by their score (s=...),
        age (a=...), and lost count (l=...). The focus object is colored by
        a white bounding box

        :param img: cv2 image with three channels in BGR mode.
        :return: image with overlaid bboxes.
        """

        for obj in self.tracked_objects:
            text = "s={:1.2f}, a={}, l={}".format(obj.prediction,
                                                  obj.count_visibility,
                                                  obj.count_lost)

            # determine brightness depending on age and lost count.
            brightness = (1 - 1 / (0.5 + np.sqrt(obj.count_visibility))) * \
                         (1 - 0.7 * obj.count_lost / self.max_count_lost)

            if obj.prediction > self.prediction_threshold:
                color = (0, 255 * brightness, 0)
            else:
                color = (0, 0, 255 * brightness)

            img = cv2.rectangle(img,
                                (int(obj.bbox[0]), int(obj.bbox[1])),
                                (int(obj.bbox[2]), int(obj.bbox[3])),
                                color=color,
                                thickness=1)

            img = cv2.putText(img,
                              text,
                              (int(obj.bbox[0]), int(obj.bbox[1]) - 3),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              color=color,
                              fontScale=0.5)

        obj = self.determine_focus_object()
        if obj is not None:
            img = cv2.rectangle(img,
                                (int(obj.bbox[0]), int(obj.bbox[1])),
                                (int(obj.bbox[2]), int(obj.bbox[3])),
                                color=(255, 255, 255),
                                thickness=2)

        return img


class TrackedObject(object):
    """Tracked object class. The class includes the basic attributes and
    methods that are needed by an object tracker.
    """
    def __init__(self, bbox, prediction, distance=None,
                 count_visibility=1, count_lost=0, alpha_bbox_center=0.5,
                 alpha_bbox_size=0.3, alpha_prediction=0.5, alpha_distance=0.8,
                 box_area=(0, 0, 1279, 959)):
        """
        :param bbox: bounding boxes in the format [left, top, right,
            bottom] and shape [#boxes, 4]; dtype has to be np.ndarray
        :param prediction: confidence scores in the shape [#boxes]; dtype has
            to be np.ndarray
        :param distance: None or distances in the shape [#boxes]; dtype has
            to be np.ndarray
        :param count_visibility: counter for how long the object is already
            visible (age)
        :param count_lost: counter for how long the object is lost
        :param alpha_bbox_center: learning rate for the bbox center (like
            gradient descent)
        :param alpha_bbox_size: learning rate for the bbox size (like
            gradient descent)
        :param alpha_prediction: alpha of moving mean to smooth the prediction
        :param alpha_distance: learning rate for the distance (like gradient
            descent)
        :param box_area: region in which the bbox is plausible (image size)
        """
        self.bbox = bbox
        self.prediction = prediction
        self.distance = distance

        self.count_visibility = count_visibility
        self.count_lost = count_lost

        # learning rates for updates
        self.alpha_bbox_center = alpha_bbox_center  # difference based
        self.alpha_bbox_size = alpha_bbox_size
        self.alpha_prediction = alpha_prediction  # moving mean
        self.alpha_distance = alpha_distance  # difference based

        self.box_area = box_area

        # the bbox momentum is stored in the format [center_x, center_y,
        # width, height]
        self.momentum_bbox = None
        self.momentum_distance = None

        self.future_bbox = self.bbox  # used to track the position at time t+1

    def __call__(self, bbox=None, prediction=None, distance=None):
        """Predict the new position of an object. If a given, the predicted
        position is based on the detected position. Otherwise it is based on
        the stored momentum.

        :param bbox: bounding boxes in the format [left, top, right,
            bottom] and shape [#boxes, 4]; dtype has to be np.ndarray
            (optional)
        :param prediction: confidence scores in the shape [#boxes]; dtype has
            to be np.ndarray (optional)
        :param distance: None or distances in the shape [#boxes]; dtype has
            to be np.ndarray (optional)
        :return: the tracked object
        """

        # this is equivalent to the object is lost
        if bbox is None and prediction is None and distance is None:
            self.count_lost += 1

            # predict new values based on momentum
            if self.momentum_bbox is not None:
                # Don't predict the size, only the position. This avoids the
                # explosion of boxes.
                self.bbox = self.predict_bbox(predict_size=False)
                self.future_bbox = self.predict_bbox(predict_size=False)

            if self.momentum_distance is not None and \
                    self.distance is not None:
                self.distance = self.distance + \
                                self.alpha_distance * self.momentum_distance

        # object tracked -> was already visible at time t-1
        elif bbox is not None and prediction is not None:
            self.count_visibility += 1
            self.count_lost = 0

            self.update_momentum_bbox(bbox)
            self.bbox = self.predict_bbox()
            # predict future position at time t+1 (only the position!)
            self.future_bbox = self.predict_bbox(predict_size=False)

            # distance information available?
            if distance is not None:
                if self.distance is None:
                    self.distance = distance
                else:
                    self.momentum_distance = distance - self.distance
                    self.distance = \
                        self.distance + \
                        self.alpha_distance * self.momentum_distance

            self.prediction = self.alpha_prediction * prediction + \
                              (1 - self.alpha_prediction) * self.prediction

        else:
            raise ValueError("All inputs have to be None (object is "
                             "lost) or at least bbox and prediction has "
                             "to be given (object tracked)")

        # shrink to plausible region
        self.bbox[0] = max(self.bbox[0], self.box_area[0])
        self.bbox[1] = max(self.bbox[1], self.box_area[1])
        self.bbox[2] = min(self.bbox[2], self.box_area[2])
        self.bbox[3] = min(self.bbox[3], self.box_area[3])

        # empty bounding box (collapsed) --> mark for removal
        if self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]:
            self.count_lost = np.inf

        if self.distance is not None and self.distance < 0:
            self.distance = 0

        # preserve minimal size of future bbox
        if self.future_bbox[0] >= self.future_bbox[2] or \
                self.future_bbox[1] >= self.future_bbox[3]:
            xywh = bbox_ltrb_to_xywh(self.future_bbox)

            if xywh[2] <= 0:
                xywh[2] = 1

            if xywh[3] <= 0:
                xywh[3] = 1

            self.future_bbox = bbox_xywh_to_ltrb(xywh)

        return self

    def to_dict(self):
        """Convert the bbox, distance, and score information to a dict. The
        following keys are available:
            * `bbox`: bounding box position in the format [left, top,
                right, bottom] in floating point values
            * `distance`: distance information (can be None)
            * `score`: confidence value of the prediction

        :return: the dictionary with the information specified above.
        """
        return {'bbox': self.bbox,
                'distance': self.distance,
                'score': self.prediction}

    def predict_bbox(self, predict_size=True):
        """Predict the position of the object (bbox) based on the current
        position and the available momentum.

        :param predict_size: Boolean value to determine whether the size
            should be predicted
        :return: predicted bbox position in the format [left, top, right,
            bottom] and shape [#boxes, 4]; dtype np.ndarray
        """
        xywh = bbox_ltrb_to_xywh(self.bbox)

        xywh[:2] = xywh[:2] + self.momentum_bbox[:2] * self.alpha_bbox_center
        xywh[:2] = xywh[:2] + self.momentum_bbox[:2] * self.alpha_bbox_center

        if predict_size:
            xywh[2:] = xywh[2:] + self.momentum_bbox[2:] * self.alpha_bbox_size
            xywh[2:] = xywh[2:] + self.momentum_bbox[2:] * self.alpha_bbox_size

        return bbox_xywh_to_ltrb(xywh)

    def update_momentum_bbox(self, bbox):
        """Updates the bbox momentum based on the given bbox position.

        :param bbox: bounding boxes in the format [left, top, right,
            bottom] and shape [#boxes, 4]; dtype has to be np.ndarray
        """
        xywh_0 = bbox_ltrb_to_xywh(self.bbox)
        xywh_1 = bbox_ltrb_to_xywh(bbox)

        self.momentum_bbox = xywh_1 - xywh_0


if __name__ == '__main__':
    import os
    import json
    import argparse

    # Exemplary visualization of the tracker results
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str,
                        default="/pvdn_dataset/day/val/images/S00121",
                        help="Path to a sequence image folder.")
    parser.add_argument("--bounding_box_path", type=str,
                        default="/pvdn_dataset/day/val/labels/bounding_boxes",
                        help="Path to the corresponding bounding box folder.")
    args = parser.parse_args()

    tracker = Tracker()

    for img_file in sorted(os.listdir(args.image_path)):
        name, ext = os.path.splitext(img_file)
        annotations_path = os.path.join(args.bounding_box_path, name + '.json')

        if not os.path.exists(annotations_path):
            raise FileNotFoundError(annotations_path + " not found.")

        if ext.lower() == '.png':
            # check bounding box file existence
            if ext == '.png':
                with open(annotations_path, 'r') as f:
                    annotations = json.load(f)

                img = cv2.imread(os.path.join(args.image_path, img_file))
                cv2.namedWindow(name)

                bbs = []
                labels = []
                for idx in range(len(annotations['labels'])):
                    bb = annotations['bounding_boxes'][idx]
                    bbs.append(bb)
                    label = annotations['labels'][idx]

                    # distort labels a bit to have a variation
                    if label < 0.5:
                        label = 0 + np.random.rand() * 0.25
                    else:
                        label = 1 - np.random.rand() * 0.25
                    labels.append(label)

                # update tracker
                tracker(bbs, labels)

                img = tracker.plot_tracked_objects(img)
                print(len(tracker.tracked_objects))

                # open image window to check boxes
                while True:
                    cv2.imshow(name, img)
                    k = cv2.waitKey(20) & 0xFF
                    # close window if ESC is pressed
                    if k == 27:
                        break
                cv2.destroyAllWindows()
