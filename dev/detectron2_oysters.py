import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

## Prepare the dataset

import os
import numpy as np
import json
from detectron2.structures import BoxMode

def get_oyster_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            # assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "val"]:
    DatasetCatalog.register("oyster_" + d, lambda d=d: get_oyster_dicts("/home/bsadrfa/behzad/projects/data_oyster/database/" + d))
    MetadataCatalog.get("oyster_" + d).set(thing_classes=["oyster"])
oyster_metadata = MetadataCatalog.get("oyster_train")

"""To verify the data loading is correct, let's visualize the annotations of randomly selected samples in the training set:"""

dataset_dicts = get_oyster_dicts("/home/bsadrfa/behzad/projects/data_oyster/database/train")
i = 1
for d in random.sample(dataset_dicts, 10):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=oyster_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    #cv2_imshow(vis.get_image()[:, :, ::-1])
    cv2.imwrite("mask_"+d["file_name"], vis.get_image()[:, :, ::-1])
    cv2.imwrite("/home/bsadrfa/behzad/projects/data_oyster/database/out/mask_{}.jpg".format(i), vis.get_image()[:, :, ::-1])
    i += 1

"""## Train!

Now, let's fine-tune a coco-pretrained R50-FPN Mask R-CNN model on the oyster dataset. It takes ~6 minutes to train 300 iterations on Colab's K80 GPU, or ~2 minutes on a P100 GPU.
"""

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("oyster_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (oyster)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

"""## Inference & evaluation using the trained model
Now, let's run inference with the trained model on the oyster validation dataset. First, let's create a predictor using the model we just trained:
"""

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("oyster_val", )
predictor = DefaultPredictor(cfg)

"""Then, we randomly select several samples to visualize the prediction results."""

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_oyster_dicts("/home/bsadrfa/behzad/projects/data_oyster/database/val")

i = 1
for d in random.sample(dataset_dicts, 5):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=oyster_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2_imshow(v.get_image()[:, :, ::-1])
    cv2.imwrite("sample_{}_out.jpg".format(i), v.get_image()[:, :, ::-1])
    cv2.imwrite("val_"+d["file_name"], v.get_image()[:, :, ::-1])
    cv2.imwrite("/home/bsadrfa/behzad/projects/data_oyster/database/out/{}.jpg".format(i), v.get_image()[:, :, ::-1])
    i += 1

"""We can also evaluate its performance using AP metric implemented in COCO API.
This gives an AP of ~70%. Not bad!
"""

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("oyster_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "oyster_val")
inference_on_dataset(trainer.model, val_loader, evaluator)
# another equivalent way is to use trainer.test

"""# Other types of builtin models"""

## Inference with a keypoint detection model
#cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
#predictor = DefaultPredictor(cfg)
#outputs = predictor(im)
#v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
##cv2_imshow(v.get_image()[:, :, ::-1])
#cv2.imwrite(d["v_keypoint_image_out.jpg", v.get_image()[:, :, ::-1])

## Inference with a panoptic segmentation model
#cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
#predictor = DefaultPredictor(cfg)
#panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
#v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
##cv2_imshow(v.get_image()[:, :, ::-1])
#cv2.imwrite(d["v_panoptic_image_out.jpg", v.get_image()[:, :, ::-1])

#"""# Run panoptic segmentation on a video"""

## This is the video we're going to process
#from IPython.display import YouTubeVideo, display
#video = YouTubeVideo("ll8TgCZ0plk", width=500)
#display(video)

## Install dependencies, download the video, and crop 5 seconds for processing
#!pip install youtube-dl
#!pip uninstall -y opencv-python opencv-contrib-python
#!apt install python3-opencv  # the one pre-installed have some issues
#!youtube-dl https://www.youtube.com/watch?v=ll8TgCZ0plk -f 22 -o video.mp4
#!ffmpeg -i video.mp4 -t 00:00:06 -c:v copy video-clip.mp4

## Run frame-by-frame inference demo on this video (takes 3-4 minutes)
## Using a model trained on COCO dataset
#!cd detectron2_repo && python demo/demo.py --config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml --video-input ../video-clip.mp4 --confidence-threshold 0.6 --output ../video-output.mkv \
  #--opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl

## Download the results
#from google.colab import files
#files.download('video-output.mkv')
