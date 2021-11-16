import os
from typing import List

from pvdn import ImageInformation


class Filter:
    """
    Base class for dataset filtering
    """

    def evaluate(self, image_info: ImageInformation):
        """
        evaluates if an image represented by it's info passes this filter
        :param image_info:
        :return:
        """
        return True


class NegatedFilter(Filter):
    """
    Wraps another filter and negates it
    """

    def __init__(self, wrapped_filter: Filter):
        self.wrapped_filter = wrapped_filter

    def evaluate(self, image_info: ImageInformation):
        return not self.wrapped_filter.evaluate(image_info)


class SceneFilter(Filter):
    """ Filters specific scenes

    """
    scene_ids: List[int]

    def __init__(self, scene_ids: List[int] = []):
        """
        Initializes this filter.
        :param scene_ids: A List containing the sequences IDs that should not be filtered out.
        """
        self.scene_ids = scene_ids

    def evaluate(self, image_info: ImageInformation):
        return image_info.sequence.sid in self.scene_ids


class HasKeypointFilter(Filter):
    """ Checks if this data point contains any keypoint information.

    Scenes with no keypoint-annotations will be filtered out completely.
    """
    keypoints_path: str

    def __init__(self, keypoints_path: str = ""):
        """
        Initializes this filter :param keypoints_path: the path to the directory containing the keypoint annotations
        that should be considered for this task
        """
        self.keypoints_path = keypoints_path

    def evaluate(self, image_info: ImageInformation):
        # check if keypoint annotations exist.
        kp_path = os.path.join(self.keypoints_path, "{:06d}.json".format(image_info.id))

        return os.path.exists(kp_path)
