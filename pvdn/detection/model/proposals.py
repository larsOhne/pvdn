import numpy as np
import cv2
import yaml
import pvdn.detection.utils.c_image_operations as img_ops
from pvdn.detection.utils.c_image_operations import Cblob
from pvdn.detection.utils.misc import rescale_boxes
from pvdn.detection.utils.misc import rescale_boxes

# from skimage.morphology import white_tophat, disk, binary_opening
# from skimage.transform import rescale, resize
from scipy.ndimage import label, find_objects, binary_closing, binary_dilation
# from skimage.transform.integral import integral_image


def _integral_image(img):
    """ compute integral image of img"""
    ii = cv2.integral(img)
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

    def __init__(self, k: float = 0.55, w: int = 26, padding: int = 9, eps: float = 1e-3,
                 dev_thresh: float = 0.01, nms_distance: int = 2, small_scale: tuple = None,
                 considered_region: tuple = None):
        """
        :param k: Scaling parameter in dynamic thresholding
        :param w: Window size in dynamic thresholding
        :param padding: Nbr of pixels to exclude at the image boundaries from proposal search
        :param eps: Small number for numerical stability in dynamic thresholding
        :param dev_thresh: Threshold which the deviation between maximum and minimum intensity
            within a bounding box needs to exceed in order to be proposed.
        :param nms_distance: distance until which to include points in flood fill algorithm
        :param small_scale: Tuple (x,y) of size 2 to rescale the input image. If None, the default image size is used.
        :param considered_region: Tuple (xmin, ymin, xmax, ymax) of size 4 which clips the proposal of bounding
            boxes to the considered image region. If None, the whole image gets considered
        """
        Detector.__init__(self)
        Cblob.__init__(self, k, w, eps)
        self.k = k
        self.w = w
        self.eps = eps
        self.padding = padding
        self.dev_thresh = dev_thresh
        self.nms_distance = nms_distance
        self.small_scale = small_scale
        self.considered_region = considered_region

    @staticmethod
    def from_yaml(path: str):
        """ read input parameters for the DynamicBlobDetector from .yaml file """
        with open(path, 'r') as stream:
            file = yaml.load(stream)['DynamicBlobDetector']
        return DynamicBlobDetector(k=file['k'], w=file['w'], dev_thresh=file['dev_thresh'],
                                   nms_distance=file['nms_distance'], small_scale=file['small_scale'],
                                   considered_region=file['considered_region'])

    def propose(self, img):
        """
        :param img: grayscale input image of shape [h, w] and pixels in range 0-255 -> np.ndarray
        :return: array of proposed bounding boxes of shape [nbr_boxes, 4] -> list
        """
        proposals = []
        # clip to considered image region
        cr_l, cr_t = 0, 0
        cr_b, cr_r = img.shape
        if self.considered_region is not None:
            l, t, r, b = self.considered_region

            cr_l = l if l is not None else 0
            cr_t = t if t is not None else 0
            cr_r = r if r is not None else cr_r
            cr_b = b if b is not None else cr_b

            img = img[cr_t:cr_b, cr_l:cr_r]

        h, w = img.shape
        img = img.astype('float') / 255.
        if self.small_scale is not None:
            small_scale = tuple(self.small_scale[i] // 2 for i in range(len(self.small_scale)))
            scale_x, scale_y = (float(w) / float(small_scale[0])), (float(h) / float(small_scale[1]))
            resized = cv2.resize(img, small_scale, interpolation=cv2.INTER_LINEAR)
        else:
            scale = (w // 2, h // 2)
            scale_x, scale_y = (float(w) / float(scale[0])), \
                               (float(h) / float(scale[1]))
            resized = cv2.resize(img, scale,
                                 interpolation=cv2.INTER_LINEAR)
        cv2.normalize(resized, resized, 0, 1, cv2.NORM_MINMAX)

        # filter image to remove high frequency noise
        img = cv2.GaussianBlur(resized, (5, 3), 2)

        # create integral image
        ii = _integral_image(img)

        # dynamic binarization
        # bin_image = img_ops.binarize(img, k=self.k, window=self.w, eps=self.eps)
        bin_image = self.binarize_in_c(img)

        h, w = img.shape
        assert h > self.padding and w > self.padding

        # close gaps according to nms_distance
        # the structure element --> corresponds to Linf
        if self.nms_distance > 1:
            structure = np.ones((self.nms_distance, self.nms_distance))
            bin_image_dilation = binary_dilation(bin_image, structure=structure)
        else:
            bin_image_dilation = bin_image

        # the structure element --> corresponds to Linf
        structure = np.ones((3, 3))
        labeled_image = label(bin_image_dilation, structure=structure)[0]
        if self.nms_distance > 1:
            # shrink labeled_image to original blob sizes if dilation was applied
            labeled_image = bin_image * labeled_image
        bbs = find_objects(labeled_image)
        for bb in bbs:
            height = bb[0].stop - bb[0].start
            width = bb[1].stop - bb[1].start
            proposal_size = height * width
            if 10 < proposal_size < 2000 and width > 5:
                proposal = [bb[1].start, bb[0].start, bb[1].stop, bb[0].stop]

                _mean = mean(ii, proposal)
                snippet = img[proposal[1]:proposal[3],
                          proposal[0]:proposal[2]]
                deviation = np.abs(snippet - _mean).sum() / proposal_size
                if deviation > self.dev_thresh:
                    proposals.append(proposal)

        big_scale = rescale_boxes(proposals, scale_x, scale_y)
        return [[bb[0] + cr_l, bb[1] + cr_t, bb[2] + cr_l, bb[3] + cr_t] for bb in
                big_scale]
        # return proposals
