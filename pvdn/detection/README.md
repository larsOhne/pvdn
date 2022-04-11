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

Note: If for some reason you do not want to use the Docker environment, keep in mind that the project needs to
be installed as a package in order to ensure proper linking between the different modules.

## Creating the bounding box annotations

In order to generate the bounding box annotations based on the keypoints, a blob detection algorithm is used to identify
the interesting parts of the images and calculate the bounding boxes.
(If you want to set the blob detection parameters on your own, create a .yaml file in the style
of [BlobDetectorParameters.yaml](BlobDetectorParameters.yaml) and pass it in the following command to the
BoundingBoxDataset.)

Then, a bounding box is assigned the label "1" if it contains at least one ground truth keypoint.

To automatically generate the data in the compliant format, you can do the following:

```
cd ~/pvdn/pvdn/detection/scripts
python3 generate_bbox_annotations.py --data_dir path/to/dataset/day/ --yaml BlobDetectorParameters.yaml
```

This will automatically create a folder bounding_boxes in the day/train/labels directory and store a .json file for each image of the form:
```
{
"bounding_boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
"labels": [0, 1, ...]
}
```

Note that you can also change the parameters in the BloDetectorParameters.yaml file 
in order to get a different behavior. The default parameters given here are the 
ones to reproduce the results stated in the [IROS paper](https://ieeexplore.ieee.org/abstract/document/9636162?casa_token=ssjoAGK3j5YAAAAA:BGvMEDmm11IUOuylGRecXOZ_yfGqTIgOFCwVapEP3xkpe7MXlDNFP75IT8mNOrvVZsjClWgfJe8XDg). The parameters in [BestBlobDetectorParameters.yaml](BestBlobDetectorParameters.yaml) are the ones to create the results from our [recent preprint paper](https://arxiv.org/abs/2107.11302).

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

In order to test the model, simply navigate into the detection directory and run the `test.py` script:

```
cd pvdn/detection/
python3 test.py --test_data path/test/ --output_dir runs/test/ --model_path weights_pretrained.pt --device cuda --batch_size 64 --conf_thresh 0.5
```

Note: The data has to be provided in the format as presented on kaggle.


In order to save the prediction results also in the coco format, use the `--save_coco` flag. If you want to visualize and save the prediction of certain scenes (say scene 23 and scene 96), you can pass them simply as a list like this: `--plot_scenes 23 96`. It will save them to your specified output directory.

**Disclaimer:** The image size here differs from the image size stated in the 
IROS2021 paper. 
This is because after publication a small bug was found which secretly downsized 
<<<<<<< HEAD
the images. The bug has now been fixed and the image size stated here is the one 
which actually has been used in the IROS2021 publication. If you want to use the weights of the model presented in our recent [preprint paper](https://arxiv.org/abs/2107.11302), you can use `--model_path weights_pretrained_optimized.pt`. Note that for that you should also have generated the bounding boxes with the proper parameters.

The first row of the table shows the results reported at IROS 2021 with the **actual image size**. The second row shows the results you would achieve if you use the image size actually reported in IROS 2021. If you use the parameters in [BlobDetectorParameters.yaml](BlobDetectorParameters.yaml) you get the results stated in the second row. If you want to reproduce the performance stated in the original IROS 2021 paper, you have to adjust the image size in [BlobDetectorParameters.yaml](BlobDetectorParameters.yaml) to `345x240`.

| Source | Image Size (WxH) |Precision | Recall | F1-Score | q | qk | qb |
| ------ | :-------: | :-------: | :----: | :------: | :-: | :-: | :-: |
| IROS 2021 (actual image size) | 345x240 | 0.88 | 0.54 | 0.67 | 0.40 | 0.40 +- 0.22 | 1.00 -+ 0.00 |
| IROS 2021 (reported image size) | 640x480 | 0.90 | 0.64 | 0.75 | 0.48 | 0.48 +- 0.26 | 1.00 -+ 0.00 |
=======
the images from 480x640 to 240x345. The bug has now been fixed and the image size stated here is the one 
which actually has been used in the IROS2021 publication.

| Source | Image Size (WxH) |Precision | Recall | F1-Score | q | qk | qb |
| ------ | :-------: | :-------: | :----: | :------: | :-: | :-: | :-: |
| IROS 2021 | **240x345** | 0.88 | 0.54 | 0.67 | 0.40 | 0.40 +/- 0.22 | 1.00 +/- 0.00 |
>>>>>>> main

## Evaluating runtime and computational requirements

You can check the runtime and computational requirements of the whole pipeline 
including bounding box generation and classification over a whole dataset split by 
using the `inference.py` script:

```
python3 inference.py --data /path/to/dataset/day/test --yaml BlobDetectorParameters.yaml --weights weights_pretrained.pt
```

The output will be displayed in the console.

**Note** that for counting the FLOPs for the blob detector, your CPU architecture 
has to support certain counters. If you think your counter is available but the 
script still says otherwise, have a look at the discussion here: https://stackoverflow.com/questions/32308175/papi-avail-no-events-available

## Visualize tracking results

The results of the proposed tracker can be visualized by the main function of `model/tracker.py`. For that the path to a folder with a sequence of images and the correspoding annotations has to be provided:
```
cd pvdn/detection/model
python3 tracker.py --image_path /pvdn_dataset/day/val/images/S00121 --bounding_box_path /pvdn_dataset/day/val/labels/bounding_boxes
```
The main function opens for each image a cv2 figure. **The current figure must be closed by pressing ESC.** After that the next image of the sequence is visualized.


## Working with YoloV5

**Disclaimer:** This How-To assumes that you are working in the provided docker 
environment, although it should be easily transferable towards other environments.

You can reproduce the results from the YoloV5s and YoloV5x networks. For that, you first need to 
convert the original PVDN dataset structure into the YoloV5 compatible structure. You can do 
this by using the script `pvdn_to_yolo.py`, located in the `pvdn/detection/scripts/` directory. 
To avoid dependency issues, you best do this also in the provided Docker environment.

Example:
```
cd pvdn/detection/scripts
python3 pvdn_to_yolo.py -source_dir /path/to/dataset/day -target_dir /data/yolo_day -img_size 960
```

The `-img_size` parameter specifies the image size to which the images are resized. The YoloV5 
implementation expects the images to be square, so the images will be resized to *img_size x 
img_size*.

Once you converted the file structure you can use the train and test scripts 
provided in the 
original YoloV5 implementation [here](https://github.com/ultralytics/yolov5). For the [IROS 2021 publication](https://ieeexplore.ieee.org/abstract/document/9636162?casa_token=ssjoAGK3j5YAAAAA:BGvMEDmm11IUOuylGRecXOZ_yfGqTIgOFCwVapEP3xkpe7MXlDNFP75IT8mNOrvVZsjClWgfJe8XDg), we
tested our code with the version **v5.0** and the commit `6187edcb53eb7982a23c5b0d3f1ab35d5d906ba6`. For the [preprint paper](https://arxiv.org/abs/2107.11302), we tested our code with the commit `b83e1a4adcf77ccafa72b22ade6cb3898ccb0e05`.

```
git clone https://github.com/ultralytics/yolov5.git ~/yolov5
cd ~/yolov5
git checkout 6187edcb53eb7982a23c5b0d3f1ab35d5d906ba6
```

Next, you can install all necessary dependencies in a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Now you are ready to use the `train.py` and `test.py` scripts. You can reproduce 
the training results as follows:

*YoloV5s*
```
python3 train.py --weights yolov5s.pt --data /data/yolo_day/pvdn.yaml --img-size 960 --hyp data/hyp.finetune.yaml --single-cls --workers 8 --name pvdn_yolov5s --batch-size 16 --epochs 200
```

*Yolov5x*
```
python3 train.py --weights yolov5x.pt --data /data/yolo_day/pvdn.yaml --img-size 960 --hyp data/hyp.finetune.yaml --single-cls --workers 8 --name pvdn_yolov5x --batch-size 16 --epochs 200
```

You can also download the pretrained weights from [this link](https://drive.google.com/drive/folders/1d1esRWFAElwNN8lqWvr_nec1trPyE1sY?usp=sharing) for the [results presented at IROS 2021](https://ieeexplore.ieee.org/abstract/document/9636162?casa_token=ssjoAGK3j5YAAAAA:BGvMEDmm11IUOuylGRecXOZ_yfGqTIgOFCwVapEP3xkpe7MXlDNFP75IT8mNOrvVZsjClWgfJe8XDg) or from [this link](https://drive.google.com/drive/folders/14feB0pvcDGX523m_bPMlYWpgKBnQs5pT?usp=sharing) for the results presented in [this preprint paper](https://arxiv.org/abs/2107.11302).

You can use the test script as follows. Note that you can either use the resulted 
weights from your own training or the ones you downloaded. In this example we will 
use the results from the downloaded pretrained weights.

*YoloV5s*
```
python3 test.py --task test --single-cls --verbose --save-json --save-txt --name pvdn_yolov5s --img 960 --conf-thres 0.5 --data /data/yolo_day/pvdn.yaml --batch-size 16 --weights weights_yolov5s_pretrained.pt
```

*Yolov5x*

```
python3 test.py --task test --single-cls --verbose --save-json --save-txt --name pvdn_yolov5x --img 960 --conf-thres 0.5 --data /data/yolo_day/pvdn.yaml --batch-size 16 --weights weights_yolov5x_pretrained.pt
```

The YoloV5 training and test scripts generate the predictions in the coco format. You can 
convert them to the PVDN format by using the script `evaluate_yolov5_results.py` 
provided in the *scripts* directory of this package:

```
# first deactivate the yolo venv
deactivate

cd ~/pvdn/pvdn/detection/scripts

python3 evaluate_yolov5_results.py --yolo_file 
~/yolov5/runs/test/pvdn_yolov5s/best_predictions.json --out_dir ../yolo_results/yolov5s --data_dir /path/to/pvdn/dataset/day/test

python3 evaluate_yolov5_results.py --yolo_file 
~/yolov5/runs/test/pvdn_yolov5x/best_predictions.json --out_dir ../yolo_results/yolov5x --data_dir /path/to/pvdn/dataset/day/test
```

Note that with the `--data_dir` option you need to specify the path of your original 
dataset in the PVDN format (**not** yolo format) which you used to originally 
create the yolo format.

Now your performance metrics are stored in `performance.json` in the via 
`--out_dir` specified directory.
