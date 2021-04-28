## Set up the environment

First, download the dataset from kaggle. 


To ensure you have all the requirements installed, a Dockerfile is provided in the base 
directory of this package. \
To train the model, build the image:
```
docker build --tag pvdn .
docker run -it --rm -v <path_to_dataset>:/dataset pvdn  # without cuda support
```
and start a container with CUDA support via:
```
docker run -it --rm --gpus all -v <path_to_dataset>:/dataset pvdn
``` 
and start a container without CUDA support via:
```
docker run -it --rm  -v <path_to_dataset>:/dataset pvdn
``` 
You will initially be located in the base directory of the pvdn package and your dataset directory will also be mounted into the container.

Note: If for some reason you do not want to use the Docker environment, keep in mind that you need to have OpenCV installed **and** manually compile  `image_operations.cpp` using the command `g++ -fpic -shared -o image_operations.so image_operations.cpp HeadLampObject.cpp`. Also, the project needs to be installed as a package in order to ensure proper linking between the different modules.

## Creating the bounding box annotations

In order to generate the bounding box annotations based on the keypoints, a blob detection algorithm is used to identify the interesting parts of the images and calculate the bounding boxes. Then, a bounding box is assigned the label "1" if it contains at least one ground truth keypoint.

To automatically generate the data in the compliant format, you can do the following:
```python
# start a python3 console
# example for generating the bounding boxes for the day/train split
# do this analogously for the other splits

from pvdn import BoundingBoxDataset

dataset = BoundingBoxDataset(path="path/to/day/train")
dataset.generate_bounding_boxes()

```

This will automatically create a folder bounding_boxes in the day/train/labels directory and store a .json file for each image of the form:
```
{
"bounding_boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
"labels": [0, 1, ...]
}
```

## Training the model
In order to train the model, simply navigate into the detection directory and run the `train.py` script:

```
cd pvdn/detection/
python3 train.py --train_data /path/train/ --val_data /path/val/ --epochs 500 --lr 0.01 --batch_size 64 --output_dir runs --device cuda
```

Note: The data has to be provided in the format as presented on kaggle.


In order to save the checkpoint and prediction results for each epoch, use the 
`--save_epochs` 
flag. To start the training from a checkpoint or initialize with given weights, provide the 
checkpoint path via the `--model_path` parameter.


## Testing the model

In order to test the model, simply navigate into the detction directory and run the `test.py` script:

```
cd pvdn/detection/
python3 test.py --test_data path/test/ --output_dir runs/test/ --model_path weights_pretrained.pt --device cuda --batch_size 64 --conf_thresh 0.5
```

Note: The data has to be provided in the format as presented on kaggle.


In order to save the prediction results also in the coco format, use the `--save_coco` flag. If you want to visualize and save the prediction of certain scenes (say scene 23 and scene 96), you can pass them simply as a list like this: `--plot_scenes 23 96`. It will save them to your specified output directory.


## Working with YoloV5

You can reproduce the results from the YoloV5s and YoloV5x networks. For that, you first need to 
convert the original PVDN dataset structure into the YoloV5 compatible structure. You can do 
this by using the script `pvdn_to_yolo.py`, located in the `pvdn/detection/scripts/` directory. 
To avoid dependency issues, you best do this also in the provided Docker environment.

Example:
```
cd pvdn/detection/scripts
python3 pvdn_to_yolo.py -source_dir /path/to/dataset/day 
    -target_dir /data/yolo_day -img_size 960
```

The `-img_size` parameter specifies the image size to which the images are resized. The YoloV5 
implementation expects the images to be square, so the images will be resized to *img_size x 
img_size*.

Next, you need to download the pretrained weights from this link:
```
https://drive.google.com/drive/folders/1DcajTkJKL3np81m6f7rfXddg2XYc0nF7?usp=sharing
```

Once you converted the file structure and downloaded the weights, you can use the train and test scripts provided in the 
original YoloV5 implementation [here](https://github.com/ultralytics/yolov5).

The YoloV5 training and test scripts generate the predictions in the coco format. You can 
convert them to the PVDN format by using the function `coco_to_results_format()` provided in 
`pvdn/metrics/convert.py` (please find the description of how to use the function in the code).
