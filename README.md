# Angle-Estimation-using-Detectron2-and-OpenCV

This repo provide an angle estimation framework using open source ML models and OpenCV library. 

# Installation

Git clone this repo using the command

`
git clone <url>
`

First install the conda env for detectron using the following commands

`
conda create env -n detectron2
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch (https://pytorch.org/get-started/locally/)
pip install cython
git clone detectron2 from the repo...if already installed skip this step
pip install opencv-python
conda install -c anaconda scikit-image
`

Then install pixellib dependencies as follows

`
pip3 install pycocotools
pip3 install pixellib
pip3 install pixellib --upgrade
`

Then in order to perform image segmentation on a video or image, run the main_1.py code. The code will not pnly perform
object detection using detectron2 but will also perform instance segmentation using PointRend as well as object extraction using pixellib.

`
python main_1.py
`

Once all the objects are extracted, label them manually.


then run the command

`
python main_ar_3.py
`

the reference images are stored in

`
/path/images/7-_6273_14178_1/images
`

Results have been recorded and can be viewed in the Results folder in this repo ---> https://github.com/shahaamirbader/Angle-Estimation-using-Detectron2-and-OpenCV/tree/main/Results


# References:

DETECTRON 2 

https://github.com/facebookresearch/detectron2

POINTREND

https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend

PIXELLIB

https://github.com/ayoolaolafenwa/PixelLib

COCO DATASET

https://cocodataset.org/#explore

CO3D DATASET

https://ai.meta.com/datasets/CO3D-dataset/

OPENCV LIBRARY 

https://opencv.org/
