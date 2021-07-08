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

You will initially be located in the base directory of the pvdn package and your dataset directory will also be mounted
into the container.

Note: If for some reason you do not want to use the Docker environment, keep in mind that you need to have OpenCV
installed **and** manually compile  `image_operations.cpp` using the
command `g++ -fpic -shared -o image_operations.so image_operations.cpp HeadLampObject.cpp`. Also, the project needs to
be installed as a package in order to ensure proper linking between the different modules.

## Creating the bounding box annotations

In order to generate the bounding box annotations based on the keypoints, a blob detection algorithm is used to identify
the interesting parts of the images and calculate the bounding boxes.
(If you want to set the blob detection parameters on your own, create a .yaml file in the style
of [BlobDetectorParameters.yaml](BlobDetectorParameters.yaml) and pass it in the following command to the
BoundingBoxDataset.)

Then, a bounding box is assigned the label "1" if it contains at least one ground truth keypoint.

To automatically generate the data in the compliant format, you can do the following:

```python
# start a python3 console
# example for generating the bounding boxes for the day/train split
# do this analogously for the other splits

from pvdn import BoundingBoxDataset
from pvdn.detection.model.proposals import DynamicBlobDetector

dataset = BoundingBoxDataset(path="path/to/day/train",
                             blob_detector=DynamicBlobDetector.from_yaml(
                                 "path/to/BlobDetectorParameters.yaml"))  # BlobDetector is optional
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


## Visualize tracking results

The results of the proposed tracker can be visualized by the main function of `model/tracker.py`. For that the path to a folder with a sequence of images and the correspoding annotations has to be provided:
```
cd pvdn/detection/model
python3 tracker.py --image_path /pvdn_dataset/day/val/images/S00121 --bounding_box_path /pvdn_dataset/day/val/labels/bounding_boxes
```
The main function opens for each image a cv2 figure. **The current figure must be closed by pressing ESC.** After that the next image of the sequence is visualized.
