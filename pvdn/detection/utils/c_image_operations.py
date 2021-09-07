import ctypes
import cv2
from numpy.ctypeslib import ndpointer
import numpy as np
from skimage.transform.integral import integral_image
import os


class Cblob(object):
    """Helper class to provide a clean Python interface to the custom C extensions"""

    def __init__(self, k=0.06, w=11, eps=1e-3, n_p=None, nms_distance=2,
                 padding=5):
        """
        Constructur method.
        :param k: scaling parameter in dynamic thresholding
        :param w: window size in dynamic thresholding
        :param eps: small number for numerical stability in dynamic thresholding
        :param nms_distance: distance until which to include points in flood fill algorithm
        :param padding: nbr of pixels to exclude at the image boundaries from proposal search
        """
        # TODO: add parameter description
        self.k = k
        self.window = w
        self.window_area = w ** 2
        self.eps = eps
        self.n_p = n_p
        self.nms_distance = nms_distance
        self.padding = padding

        self._init_c_lib()

    def _init_c_lib(self):
        """Initialization of the C library and its functions to make them callable in Python"""
        # lib = ctypes.cdll.LoadLibrary("/home/sascha/eodan/blob_detector_with_classifier/utils/image_operations.so")
        lib_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "image_operations.so")
        lib = ctypes.cdll.LoadLibrary(lib_dir)

        # initialize binarize function
        self.__c_binarize = lib.binarize
        self.__c_binarize.restype = None
        self.__c_binarize.argtypes = [
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_ubyte, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_float]

        # initialize propose function
        self.__c_find_proposals = lib.find_proposals
        self.__c_find_proposals.restype = None
        self.__c_find_proposals.argtypes = [
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_ubyte, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    def binarize_in_c(self, img):
        """
        Wrapper function for the C extension binarize function.
        :param img: single channel input image (np.ndarray) of shape [h, w];
                    the image has to be of type np.float64 or np.float32 and in a range 0.0 - 1.0
        :return: binarized image (np.ndarray) of shape [h, w] and type np.uint8
                (without any morphological ops applied)
        """
        if not "float" in str(img.dtype):
            raise TypeError(
                "The input image is expected to be of type np.float32 or np.float64, "
                "but it is {}. Please check.".format(str(img.dtype)))

        h, w = img.shape[-2:]
        ii = integral_image(img)
        ii = ii.astype(np.float32)
        img = img.astype(np.float32)

        # initialize array to store binary image
        bin_img = np.zeros(shape=img.shape, dtype=np.uint8)

        self.__c_binarize(img, bin_img, ii, ctypes.c_int(h), ctypes.c_int(w),
                          ctypes.c_int(self.window),
                          ctypes.c_float(self.k), ctypes.c_float(self.eps))
        return bin_img

    def find_proposals_in_c(self, bin_img):
        """
        Wrapper function for the C extension find_proposals function (WITHOUT gradient filtering).
        :param bin_img: binary image (np.ndarray) of shape [h, w] and type np.uint8
        :return: bounding box proposals (np.ndarray) of shape [nbr_of_proposals, 4] and type np.int32
                NOTE: Here the gradient filtering has not been applied yet!
        """
        if not "uint8" in str(bin_img.dtype):
            raise TypeError(
                "The binary input image is expected to be of type np.uint8, "
                "but found {}. Please check.".format(str(bin_img.dtype)))

        h, w = bin_img.shape[-2:]

        # initialize the array with a size of 2000/4=500 maximal possible final proposals
        final_proposals = np.array([-1] * 2000, dtype=np.int32)

        # c function
        self.__c_find_proposals(final_proposals, bin_img, ctypes.c_int(h),
                                ctypes.c_int(w),
                                ctypes.c_int(self.padding),
                                ctypes.c_int(self.nms_distance))

        # delete all unused proposal spots in array and resize it to shape [nbr_of_proposals, 4]
        final_proposals = np.delete(final_proposals,
                                    np.where(final_proposals == -1))
        final_proposals = final_proposals.reshape((-1, 4))

        # to shift from pixel to index representation
        final_proposals -= 1

        return final_proposals
