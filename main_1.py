##Author: Aamir Bader Shah
##This code performs instance segmentation using Instseg.py file and also perfroms object extraction using pixellib library
from InstSeg import *
from detectron2.data import transforms as T

import cv2
from pixellib.torchbackend.instance import instanceSegmentation
detector = InstSeg() 

#detector.onVideo(instance_idx= 1, category_idx= 56, videoPath="/homelocal/videos/video_1.mp4")

detector.onimage(instance_idx= 4, category_idx= 51, imagepath= "/homelocal/images/cup/img_1.jpg" )

detector.bbox()

#detector.main()



ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")
results, output = ins.segmentImage("/homelocal/images/cup/img_1.jpg", show_bboxes=True, extract_segmented_objects=True,
save_extracted_objects=True, output_image_name="/homelocal/detectron2/output/output_image.jpg")
print(results["extracted_objects"])
