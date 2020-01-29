import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode

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


data_folder = "/home/bsadrfa/behzad/projects/data_oyster/database/"
save_folder = "/home/bsadrfa/behzad/projects/oyster/detectron2/dev/out/"
for d in ["train", "val"]:
    DatasetCatalog.register("oyster_" + d, lambda d=d: get_oyster_dicts(os.path.join(data_folder, d)))
    MetadataCatalog.get("oyster_" + d).set(thing_classes=["oyster"])
oyster_metadata = MetadataCatalog.get("oyster_train")

# To verify the data loading is correct,
# let's visualize the annotations of randomly selected samples in the training set
dataset_dicts = get_oyster_dicts(os.path.join(data_folder, "train"))
for d in random.sample(dataset_dicts, 10):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=oyster_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite(os.path.join(save_folder, "train_{}.jpg".format(d["image_id"])),
                vis.get_image()[:, :, ::-1])
# Train!
# Now, let's fine-tune a coco-pretrained R50-FPN Mask R-CNN model on the oyster dataset.
# It takes ~6 minutes to train 300 iterations on Colab's K80 GPU, or ~2 minutes on a P100 GPU.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("oyster_train",)
# cfg.DATASETS.TEST = ()
cfg.DATASETS.TEST = ("oyster_val",)
cfg.DATALOADER.NUM_WORKERS = 2
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
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
thresh_percent = 60
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_percent*.01   # set the testing threshold for this model
cfg.DATASETS.TEST = ("oyster_val", )
predictor = DefaultPredictor(cfg)

"""Then, we randomly select several samples to visualize the prediction results."""



dataset_dicts = get_oyster_dicts(os.path.join(data_folder, "val"))
for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    # Prediction
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=oyster_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE
                   # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(os.path.join(save_folder, "{}_predicted_{}.jpg".format(d["image_id"], thresh_percent)),
                             v.get_image()[:, :, ::-1])
    # Ground Truth
    img = cv2.imread(d["file_name"])
    cv2.imwrite(os.path.join(save_folder, "{}_no_mask.jpg".format(d["image_id"])), img)
    visualizer = Visualizer(img[:, :, ::-1], metadata=oyster_metadata, scale=0.8)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite(os.path.join(save_folder, "{}_ground_truth.jpg".format(d["image_id"])),
                vis.get_image()[:, :, ::-1])


"""We can also evaluate its performance using AP metric implemented in COCO API.
This gives an AP of ~70%. Not bad!
"""

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("oyster_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "oyster_val")
inference_on_dataset(trainer.model, val_loader, evaluator)
# another equivalent way is to use trainer.test
