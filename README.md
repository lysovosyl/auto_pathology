
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
Download YOLOv5-7.0 from: https://github.com/ultralytics/yolov5.
Copy Python script and PNG images to the root of the ./yolov5-7.0 folder.

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

Next, select image augmentation methods and other techniques for image enhancement and noise reduction. Modify the training parameters in the sidebar. Click the "开始训练" button to begin training your custom weights. Once training is complete, the best weights will be saved on the server and can be selected for the object detection function in subsequent steps.

Upon completion of training, key metrics including training curves, the confusion matrix, and the Precision-Recall (PR) curve will be displayed. Displaying the performance of the best weights, such as precision and recall, would provide a more intuitive understanding of the model's effectiveness, though the implementation of log parsing for these specific metrics is complex and has not been implemented.


### Object Detection on Patches

The following example assumes that the patch images data is organized in well known standard formats (such as .jpg, .png etc.) and stored in a folder named DATA_PATCHES

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
        └── ...

Select the "Detection" function from the sidebar and place the files you need to detect into the upload box. You can view the uploaded images. Adjust parameters in the sidebar and select the desired weights (distinguished by timestamp). Click "run detection" to perform object detection on the uploaded images. The webpage will display summarized results, and you can download a CSV file.

Below is an example table for slide_1.csv. 

| index | images |    xmin | ymin | xmax| ymax|    confidence |  | name| area|
|:------|     :---:      |-----:|           ---: |           ---: |           ---: |-----:|           ---: |           ---: |           ---: |
| 0     |  tile_12_level0_5896-4523-6616-5243.png    | 305.289856  | 374.140716552734     |  342.514923095703   |  463.612335205078 | 0.435933530330657 | 0     |  fillcle  |  3330.58701133914 |
| 1     |  tile_12_level0_5896-4523-6616-5243.png    | 15.428744316101  | 203.001434326171     |  105.341537475585   |  326.226593017578 | 0.239176213741302 | 0     |  fillcle  |  11079.5182054651 |
| 2     |  tile_12_level0_5896-4523-6616-5243.png    | 325.523345947265  | 33.3408126831054     |  443.434539794921   |  165.721862792968 | 0.195653393864631 | 0     |  fillcle  |  15609.2076612603 |

This table includes 10 columns.
- `index`: The serial number of the detection result.
- `images`: The name of the detected image.
- `xmin`: The lower bound of the x-coordinate of the detection bounding box.
- `ymin`: The lower bound of the y-coordinate of the detection bounding box.
- `xmax`: The upper bound of the x-coordinate of the detection bounding box.
- `ymax`: The upper bound of the y-coordinate of the detection bounding box.
- `confidence`: The confidence score of the detection result.
- `class_index`: The class index of the detected object.
- `name`: The class name of the detected object.
- `area`: The area of the detection bounding box.


## Please cite

 Submitted.

## Maintainer

Any questions, please contact [@Rongtao Ye](https://github.com/xumaosan)  [@XiaoNan Yu](https://github.com/shannon-you)

## Contributors

Thank you for the helps from Dr. Jiajian Zhou, Dr. Yusen Lin.

## License

[MIT](LICENSE) © Rongtao Ye @XiaoNan Yu
