import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json


# SIP-SEP class
#CLASS_NAMES = ["foreground"]
# TCAN-D classes
#CLASS_NAMES = ["rov", "plant", "animal_fish", "animal_starfish", "animal_shells", "animal_crab", "animal_eel", "animal_etc", "trash_clothing", "trash_pipe", "trash_bottle", "trash_bag", "trash_snack_wrapper", "trash_can", "trash_cup", "trash_container", "trash_unknown_instance", "trash_branch", "trash_wreckage", "trash_tarp", "trash_rope", "trash_net"]
# COCO-SEP classes
#CLASS_NAMES = ["person", "vehicles", "car", "chair", "furniture", "electronics", "home_appliances", "utensils", "food", "toys_sports_equipment", "animals", "misc_objects", "fashion_items"]
# COCO classes
#CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

PREDEFINED_SPLITS_DATASET = {
    "SIP-SEP_train": ("SIP-SEP_train/RGB", "SIP-SEP_train/SIP-SEP_train.json"),
    "SIP-SEP_val": ("SIP-SEP_val/RGB", "SIP-SEP_val/SIP-SEP_val.json"),
    "TCAN-D_train": ("TCAN-D_train/RGB", "TCAN-D_train/TCAN-D_train.json"),
    "TCAN-D_val": ("TCAN-D_val/RGB", "TCAN-D_val/TCAN-D_val.json"),
    "COCO-SEP_train": ("COCO-SEP_train/RGB", "COCO-SEP_train/COCO-SEP_train.json"),
    "COCO-SEP_val": ("COCO-SEP_val/RGB", "COCO-SEP_val/COCO-SEP_val.json"),
    "COBO-SEP_train": ("COBO-SEP_train/RGB", "COBO-SEP_train/COBO-SEP_train.json"),
    "COBO-SEP_val": ("COBO-SEP_val/RGB", "COBO-SEP_val/COBO-SEP_val.json"),
}

def register_dataset_instances(name, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")

def register_rgbd_datasets(root):
    for key,(image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key, json_file=os.path.join(root,json_file), image_root=os.path.join(root,image_root))

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

register_rgbd_datasets(_root)