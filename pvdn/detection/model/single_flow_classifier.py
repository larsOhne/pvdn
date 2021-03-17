import torch
import cv2
# from ptflops import get_model_complexity_info
from torch import nn
from torchvision.transforms.functional import to_tensor
import numpy as np


class IntermediateOutput(nn.Module):
    """seperate branch for intermediate classification output"""
    def __init__(self, in_features=128):
        super(IntermediateOutput, self).__init__()

        self.afc1 = nn.Linear(in_features, in_features // 2)
        self.arelu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.afc2 = nn.Linear(in_features // 2, in_features // 4)
        self.arelu2 = nn.ReLU()
        self.afc3 = nn.Linear(in_features // 4, 1)

    def forward(self, x):
        x = self.arelu1(self.afc1(x))
        x = self.dropout(x)
        x = self.arelu2(self.afc2(x))
        out = self.afc3(x)

        return out


class Classifier(nn.Module):
    """Class for classifier proposed in the original PVDN paper submitted at IROS 2021"""
    def __init__(self, f_size=64, support_multiplier=1, support_size=64):
        super(Classifier, self).__init__()
        self.f_size = f_size
        self.sup_size = support_size
        self.sup_mult = support_multiplier

        # reduce vertical dimension with strided convs
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))
        self.batch2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), bias=False)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.batch4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=(5, 5))
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), bias=False)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.AvgPool2d(5, 5)
        self.batch6 = nn.BatchNorm2d(256)

        self.int_out = IntermediateOutput(in_features=256)

    def process_image(self, img, boxes, device="cuda"):
        """
        Classifies a set of bounding boxes corresponding to the input image.
        :param img: input image of shape [h, w] and pixels in range 0-255 -> np.ndarray
        :param boxes: array of bounding boxes of shape [nbr_boxes, 4] -> list
        :param device: cuda or cpu -> str
        :return: torch.tensor of shape [nbr_boxes, 1] containing the predicted labels
            -> torch.tensor
        """
        self.eval()
        if len(boxes) < 1:
            return torch.zeros((0, 1))

        feature_tensor = torch.zeros((len(boxes), 1, 64, 64), device=device)
        h, w = img.shape
        for i, bb in enumerate(boxes):
            bb_w, bb_h = bb[2] - bb[0], bb[3] - bb[1]

            center_x = bb[0] + (bb[2] - bb[0]) // 2
            center_y = bb[1] + (bb[3] - bb[1]) // 2
            crop_bb = [max(center_x - bb_w * self.sup_mult, 0),
                       max(center_y - bb_h * self.sup_mult, 0),
                       min(center_x + bb_w * self.sup_mult, w),
                       min(center_y + bb_h * self.sup_mult, h)]

            crop_bb = np.array(crop_bb).astype('uint32')

            feature = cv2.resize(
                img[crop_bb[1]:crop_bb[3], crop_bb[0]:crop_bb[2]],
                (self.f_size, self.f_size))

            np_feature = np.array(feature).astype('float32') / 255

            max_intensity = np_feature.max()
            min_intensity = np_feature.min()
            difference = np.maximum(max_intensity - min_intensity, 1.e-12)
            np_feature = (np_feature - min_intensity) / difference

            feature_tensor[i] = to_tensor(np_feature)

        return self(feature_tensor).reshape(-1, 1)

    def forward(self, x):
        x = self.pool2(self.relu2(self.conv2(self.relu1(self.conv1(x)))))
        x = self.batch2(x)

        x = self.pool4(self.relu4(self.conv4(self.relu3(self.conv3(x)))))
        x = self.batch4(x)

        x = self.pool6(self.relu6(self.conv6(self.relu5(self.conv5(x)))))
        x = self.batch6(x)

        x = torch.flatten(x, 1)
        out = self.int_out(x)
        out = out.reshape(-1, 1)

        return out
