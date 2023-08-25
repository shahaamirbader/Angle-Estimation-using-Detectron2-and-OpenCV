##Author:Aamir Bader Shah
##This code performs instance segmentation using detectron2. The code is run in the main_1.py file
from matplotlib import figure
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend
import matplotlib.pyplot as plt
import time
import skimage.io
class InstSeg:
    def __init__(self):
        self.cfg = get_cfg()
        
        # Add PointRend-specific config
        point_rend.add_pointrend_config(self.cfg)
        # Load a config from file
        self.cfg.merge_from_file("projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
        self.cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco/164254221/model_final_736f5a.pkl"
        self.predictor = DefaultPredictor(self.cfg)
      
          
        
        # First we define a simple function to help us plot the intermediate representations.

  
      
    def plot_mask(self, mask, title="", point_coords=None, figsize=10, point_marker_size=5):
      '''
      Simple plotting tool to show intermediate mask predictions and points 
      where PointRend is applied.
      
      Args:
        mask (Tensor): mask prediction of shape HxW
        title (str): title for the plot
        point_coords ((Tensor, Tensor)): x and y point coordinates
        figsize (int): size of the figure to plot
        point_marker_size (int): marker size for points
      '''
      
      H, W = mask.shape
      plt.figure(figsize=(figsize, figsize))
      
      if title:
        title += ", "
      plt.title("{}resolution {}x{}".format(title, H, W), fontsize=30)
      plt.ylabel(H, fontsize=30)
      plt.xlabel(W, fontsize=30)
      plt.xticks([], [])
      plt.yticks([], [])
      plt.imshow(mask, interpolation="nearest", cmap=plt.get_cmap('gray'))
      if point_coords is not None:
        plt.scatter(x=point_coords[0], y=point_coords[1], color="red", s=point_marker_size, clip_on=True) 
      plt.xlim(-0.5, W - 0.5)
      plt.ylim(H - 0.5, - 0.5)
      plt.savefig('/homelocal/videos/output_image_1.jpg', bbox_inches='tight')
      plt.show()

    def onimage(self, instance_idx, category_idx, imagepath):
    
      self.im = cv2.imread(imagepath)
      self.im = cv2.resize(self.im, (0, 0), fx=0.5, fy=0.5)    

      predictions = self.predictor(self.im)
            
      viz = Visualizer(self.im[:,:,::-1], metadata= MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.SEGMENTATION)
            
      output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
      cv2.imshow("Result", output.get_image()[:,:,::-1])
      cv2.imwrite("/homelocal/images/frame_instances.jpg", output.get_image()[:,:,::-1] )
      cv2.waitKey(0)     
      # In this image we detect several objects but show only the first one.
      self.instance_idx = instance_idx
      # Mask predictions are class-specific, "plane" class has id 4.
      self.category_idx = category_idx
     
      
    
    def onVideo(self, instance_idx, category_idx, videoPath):
        cap = cv2.VideoCapture(videoPath)
        width = 400
        height = 350
        # used to record the time when we processed last frame
        prev_frame_time = 0
 
        # used to record the time at which we processed current frame
        new_frame_time = 0
        if (cap.isOpened()==False):
            print("Error opening the file...")
            return 
        
        (sucess, image) = cap.read()
        
        while sucess:
            image = cv2.resize(image, (width, height))
            predictions = self.predictor(image)
            viz = Visualizer(image[:,:,::-1], metadata= MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.SEGMENTATION)
            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
            # font which we will be using to display FPS
            font = cv2.FONT_HERSHEY_SIMPLEX
            # time when we finish processing for this frame
            new_frame_time = time.time()
        
            # Calculating the fps
        
            # fps will be number of frame processed in given time frame
            # since their will be most of time error of 0.001 second
            # we will be subtracting it to get more accurate result
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
        
            # converting the fps into integer
            fps = int(fps)
        
            # converting the fps to string so that we can display it on frame
            # by using putText function
            fps = str(fps)
            
            # putting the FPS count on the frame
            cv2.putText(image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        
            # displaying the frame with fps
            cv2.imshow("Result", output.get_image()[:,:,::-1])
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (sucess, image) = cap.read()
            cv2.imwrite("/homelocal/videos/frame_1.jpg", image)     # save frame as JPEG file
        self.im = cv2.imread("/homelocal/videos/frame_1.jpg")
        viz = Visualizer(image[:,:,::-1], metadata= MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.SEGMENTATION)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        
        cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.imwrite("/homelocal/videos/frame_instances.jpg", output.get_image()[:,:,::-1] )
        
        # In this image we detect several objects but show only the first one.
        self.instance_idx = instance_idx
        # Mask predictions are class-specific, "plane" class has id 4.
        self.category_idx = category_idx
    
    def bbox(self):
        
        from detectron2.data import transforms as T
        self.outputs = self.predictor(self.im)
        model = self.predictor.model
        outputs = self.outputs
        with torch.no_grad():
            # Prepare input image.
            height, width = self.im.shape[:2]
            im_transformed = T.ResizeShortestEdge(800, 1333).get_transform(self.im).apply_image(self.im)
            batched_inputs = [{"image": torch.as_tensor(im_transformed).permute(2, 0, 1)}]
            
            # Get bounding box predictions first to simplify the code.
            detected_instances = [x["instances"] for x in model.inference(batched_inputs)]
            [r.remove("pred_masks") for r in detected_instances]  # remove existing mask predictions
            pred_boxes = [x.pred_boxes for x in detected_instances] 
            classes = outputs['instances'].pred_classes
            print(len(classes), classes)
            # Run backbone.
            images = model.preprocess_image(batched_inputs)
            features = model.backbone(images.tensor)
            
            '''
            # Given the bounding boxes, run coarse mask prediction head.
            mask_coarse_logits = model.roi_heads.mask_head.coarse_head(model.roi_heads.mask_head._roi_pooler(features, pred_boxes))
            
            self.plot_mask(
                mask_coarse_logits[self.instance_idx, self.category_idx].to("cpu"),
                title="Coarse prediction"
            )
            

        # Prepare features maps to use later
        mask_features_list = [
        features[k] for k in model.roi_heads.mask_head.mask_point_in_features
        ]
        features_scales = [
        model.roi_heads.mask_head._feature_scales[k] 
        for k in model.roi_heads.mask_head.mask_point_in_features
        ]


        from detectron2.layers import interpolate
        from detectron2.projects.point_rend.mask_head import calculate_uncertainty
        from detectron2.projects.point_rend.point_features import (
            get_uncertain_point_coords_on_grid,
            point_sample,
            point_sample_fine_grained_features,
        )

        num_subdivision_steps = 5
        num_subdivision_points = 24 * 24


        with torch.no_grad():
            # We take predicted classes, whereas during real training ground truth classes are used.
            pred_classes = torch.cat([x.pred_classes for x in detected_instances])
            self.plot_mask(
                mask_coarse_logits[0, self.category_idx].to("cpu").numpy(), 
                title="Coarse prediction"
            )

            mask_logits = mask_coarse_logits
            for subdivions_step in range(num_subdivision_steps):
                # Upsample mask prediction
                mask_logits = interpolate(
                    mask_logits, scale_factor=2, mode="bilinear", align_corners=False
                )
                # If `num_subdivision_points` is larger or equalto to the
                # resolution of the next step, then we can skip this step
                H, W = mask_logits.shape[-2:]
                if (
                num_subdivision_points >= 4 * H * W
                and subdivions_step < num_subdivision_steps - 1
                ):
                    continue
                # Calculate uncertainty for all points on the upsampled regular grid
                uncertainty_map = calculate_uncertainty(mask_logits, pred_classes)
                # Select most `num_subdivision_points` uncertain points
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, 
                    num_subdivision_points
                )

                # Extract fine-grained and coarse features for the points
                fine_grained_features, _ = point_sample_fine_grained_features(
                mask_features_list, features_scales, pred_boxes, point_coords
                )
                coarse_features = point_sample(mask_coarse_logits, point_coords, align_corners=False)

                # Run PointRend head for these points
                point_logits = model.roi_heads.mask_head.point_head(fine_grained_features, coarse_features)
                
                # put mask point predictions to the right places on the upsampled grid.
                R, C, H, W = mask_logits.shape
                x = (point_indices[self.instance_idx] % W).to("cpu")
                y = (point_indices[self.instance_idx] // W).to("cpu")
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                mask_logits = (
                mask_logits.reshape(R, C, H * W)
                .scatter_(2, point_indices, point_logits)
                .view(R, C, H, W)
                )
                self.plot_mask(
                mask_logits[self.instance_idx, self.category_idx].to("cpu"), 
                title="Subdivision step: {}".format(subdivions_step + 1),
                point_coords=(x, y)
                )
                
            '''
              
        # This code has been copied from https://colab.research.google.com/drive/1isGPL5h5_cKoPPhVL9XhMokRtHDvmMVL#scrollTo=CIny-6sotDCL
