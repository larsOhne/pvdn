# PVDN
![picture](PVDN_Logo.png)
Pytorch implementation of the PVDN dataset.

## Scope
This repository contains code related to the usage of the PVDN dataset and is meant to be used together with the data-files available through [kaggle](https://www.kaggle.com/saralajew/provident-vehicle-detection-at-night-pvdn). Annotations and images are processed and packaged into the commonly used pytorch framework.

## Dataset
For advanced driver assistance systems, it is crucial to have information about oncoming vehicles as early as possible. At night, this task is especially difficult due to poor lighting conditions. For that, during nighttime, every vehicle uses headlamps to improve sight and therefore ensure safe driving. As humans, we intuitively assume oncoming vehicles before the vehicles are actually physically visible by detecting light reflections caused by their headlamps. With this dataset, we present a dataset containing 59746 annotated grayscale images out of 346 different scenes in a rural environment at night. In these images, all oncoming vehicles, their corresponding light objects (e. g., headlamps), and their respective light reflections (e. g., light reflections on guardrails) are labeled. With that, we are providing the first open-source dataset with comprehensive ground truth data to enable research into new methods of detecting oncoming vehicles based on the light reflections they cause, long before they are directly visible. We consider this as an essential step to further close the performance gap between current advanced driver assistance systems and human behavior. See the corresponding arXiv publication [2] for further details.

## References
[1] Emilio Oldenziel, Lars Ohnemus and Sascha Saralajew, **Provident Detection of Vehicles at Night**, 2020 IEEE Intelligent Vehicles Symposium (IV), Las Vegas, NV, USA, 2020, pp. 472-479, doi: 10.1109/IV47402.2020.9304752.

[2] Lars Ohnemus, Lukas Ewecker, Ebubekir Asan, Stefan Roos, Simon Isele, Jakob Ketterer, Leopold Müller, and Sascha Saralajew: **Provident Vehicle Detection at Night: The PVDN Dataset.** [arXiv:2012.15376](https://arxiv.org/abs/2012.15376), 2020.

## Citation
    @misc{ohnemus2020provident,
      title={Provident Vehicle Detection at Night: The PVDN Dataset}, 
      author={Lars Ohnemus and Lukas Ewecker and Ebubekir Asan and Stefan Roos and Simon Isele and Jakob Ketterer and Leopold Müller and Sascha Saralajew},
      year={2020},
      eprint={2012.15376},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
