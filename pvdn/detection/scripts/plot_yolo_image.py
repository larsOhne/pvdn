import os
import cv2
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script plots the results from a"
                                                 "YoloV5 training for a specific "
                                                 "image and stores them to"
                                                 "a specified location.")
    parser.add_argument("-i", "--img", type=str, help="Path to image.")
    parser.add_argument("-a", "--annot", type=str, help="Path to annotation file "
                                                        ".txt for a specific image.")
    parser.add_argument("-o", "--out", type=str, help="Output directory.")
    parser.add_argument("-c", "--color", type=str, choices=("red", "green", "blue"),
                        help="Color of the bounding box (red, green, or blue).",
                        default="green")
    opts = parser.parse_args()

    if opts.color == "red":
        color = (0, 0, 255)
    elif opts.color == "green":
        color = (0, 255, 0)
    elif opts.color == "yellow":
        color = (255, 0, 0)

    # read image
    img = cv2.imread(opts.img)
    h, w, c = img.shape

    # load annotations & plot box
    with open(opts.annot, "r") as f:
        for line in f:
            line = [float(v) for v in line.split()]
            pt1 = (int((line[1] - line[3] / 2) * w),
                   int((line[2] - line[4] / 2) * h))
            pt2 = (int((line[1] + line[3] / 2) * w),
                   int((line[2] + line[4] / 2) * h))
            img = cv2.rectangle(img, pt1=pt1, pt2=pt2,
                                color=color, thickness=3)

    # save new img
    out_path = os.path.join(opts.out, opts.img.split("/")[-1])
    cv2.imwrite(out_path, img)
    print(f"Successfully saved to {out_path}")
