
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
After downloading YOLOv5-7.0 from https://github.com/ultralytics/yolov5, place  Python script in main into the root of the ./yolov5-7.0 directory.

## Step by step tutorial

### Testing dataset
This data can be downloaded from https://gtexportal.org/home/histologyPage




### Access Online
Run the following command:
```
cd ./yolov5-7.0
streamlit run main.py
```

Access the application at http://localhost:8501.

### WSI to patch
The following example assumes that the whole slide images (WSIs) data is organized in well known standard formats (such as .svs, .ndpi, .tiff etc.) and stored in a folder named DATA_DIRECTORY.

```
    DATA_DIRECTORY/
        ├──slide_1.svs
        ├──slide_1.svs
        ├──slide_2.svs
        ├──slide_3.svs
        └── ...
        
```

Select the "H&E to patches" function from the sidebar. Place the WSI images you need to process into the upload box. After all images have been placed, click the "确定上传" button to upload the files to the server. Then, click the "开始切割" button to segment the uploaded WSI images into patches. Upon completion of the segmentation, you can download the results in a ZIP format.

The patches will be saved in ./save_dir.

    save_dir/
        ├──slide_1
            ├──1.png
            ├──2.png
            ├──3.png
            └──...
        ├──slide_2
            ├──1.png
            ├──2.png
            ├──3.png
            └──...
        ├──slide_3
            ├──1.png
            ├──2.png
            ├──3.png
            └──...
        ├──...
        ├──located_tiles1.png
        ├──located_tiles2.png
        ├──located_tiles3.png
        └── ...

The located_tiles.png file visualizes the sampling locations of the extracted tiles on the original WSI.

### Annotating Patches

This web page does not have an integrated image annotation function and requires manual annotation. In the sidebar's "Labeling" function, a link to the annotation website "Make Sense", https://www.makesense.ai/, is provided.
Since the YOLOv5 model will be used for object detection later, the annotations need to be exported in YOLOv5 format.

### Model Weight Training

After uploading the images and annotations, adjust the parameters such as the split ratios for the training set, validation set, and test set, as well as the random seed. Click the "确定划分数据集按钮" button, and the program will automatically partition the dataset.

Next, select image augmentation methods and other techniques for image enhancement and noise reduction. Modify the training parameters in the sidebar. Click the "开始训练" button to begin training your custom weights. Once training is complete, the weights will be saved on the server and can be selected for the object detection function in subsequent steps.
