
## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)
  
## Background
In recent years, automated pathology image processing technology has made significant progress, enabling efficient handling of specific tasks in a growing number of scenarios. Although the image processing workflows for various tasks are largely similar, non-specialist developers remain constrained by the high barrier to entry and limited time availability. Under these circumstances, it is challenging for them to keep pace with the rapid iteration of automated processing technologies and apply these advancements to their specialized fields.

To address this, we have developed an integrated image processing platform for pathology image analysis and deep learning application development. The platform incorporates functionalities such as patch extraction from Whole Slide Images (WSI) and YOLO-based object detection, enabling even non-specialist developers to perform automated pathology image processing conveniently and efficiently.

## Install
Make sure you have installed all the package that were list in requirements.txt
```
conda create -n Editing_auto_pathology python==3.10
pip install -r requirements.txt
conda activate Editing_auto_pathology
```
After downloading YOLOv5-7.0 from https://github.com/ultralytics/yolov5, place  Python script and the uploads folder in main into the root of the .\yolov5-7.0 directory.

## Step by step tutorial

### Testing dataset
This data can be downloaded from https://gtexportal.org/home/histologyPage




### Epidermis extraction
