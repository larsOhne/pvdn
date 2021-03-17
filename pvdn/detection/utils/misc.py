


def rescale_boxes(boxes, scale_x, scale_y):
    return [[box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y] for box in boxes]