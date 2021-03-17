import numpy as np
import cv2
from skimage.transform.integral import integral_image

import pvdn.detection.utils.c_image_operations as img_ops
from pvdn.detection.utils.c_image_operations import Cblob


def _integral_image(img):
    """ compute integral image of img"""
    ii = integral_image(img)
    return ii


def integral(box, ii):
    """ integral of ii over box"""
    return ii[int(box[1]), int(box[0])] \
           + ii[int(box[3]), int(box[2])] \
           - ii[int(box[1]), int(box[2])] \
           - ii[int(box[3]), int(box[0])]


def area(box):
    """ area of box"""
    return (box[3] - box[1]) * (box[2] - box[0])


def mean(ii, box):
    """ mean of box with integral image ii """
    return integral(box, ii) / area(box)


class Detector():
    """ Base class for all detectors """
    def __init__(self):
        pass

    def propose(self, img):
        raise NotImplementedError("Every child of the class Detector needs to implement the "
                                  "propose function.")


class DynamicBlobDetector(Cblob, Detector):
    """ BlobDetector with dynamic thresholding and integral image
    optimization """
    def __init__(self, k: float = 0.06, w: int = 11, padding: int = 10, eps: float = 1e-3,
                 dev_thresh: float = 0.01, nms_distance: int = 20):
        """
        :param k: Scaling parameter in dynamic thresholding
        :param w: Window size in dynamic thresholding
        :param padding: Nbr of pixels to exclude at the image boundaries from proposal search
        :param eps: Small number for numerical stability in dynamic thresholding
        :param dev_thresh: Threshold which the deviation between maximum and minimum intensity
            within a bounding box needs to exceed in order to be proposed.
        :param nms_distance: distance until which to include points in flood fill algorithm
        """
        Detector.__init__(self)
        Cblob.__init__(self, k, w, eps)
        self.k = k
        self.w = w
        self.eps = eps
        self.padding = padding
        self.dev_thresh = dev_thresh
        self.nms_distance = nms_distance

    def propose(self, img):
        # filter image to remove high frequency noise
        img = cv2.GaussianBlur(img, (5, 3), 2)

        # create integral image
        ii = _integral_image(img)

        # dynamic binarization
        # bin_image = img_ops.binarize(img, k=self.k, window=self.w, eps=self.eps)
        bin_image = Cblob.binarize_in_c(self, img)

        h, w = img.shape
        assert h > self.padding and w > self.padding

        # proposals = img_ops.find_proposals(bin_image, padding=self.padding,
        #                                    nms_distance=self.nms_distance)
        proposals = Cblob.find_proposals_in_c(self, bin_image)

        # remove proposals with little to no intensity gradient
        filtered = []
        for proposal in proposals:
            _mean = mean(ii, proposal)
            snippet = img[int(proposal[1]):int(proposal[3]), int(proposal[0]):int(proposal[2])]
            deviation = np.abs(snippet - _mean).sum() / area(proposal)
            proposal_size = (proposal[2] - proposal[0]) * (proposal[3] - proposal[1])
            if deviation > self.dev_thresh and proposal_size < 2000:
                filtered.append(proposal)
        return filtered



