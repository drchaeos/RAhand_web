import os
import random
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator

## parameter 설정
test_dir = '../RAhand_server/test_imgs'         # test image 들어간 directory
model_dir = '../RAhand_server/model'            # RA_hand.pth 들어간 directory
output_dir = '../RAhand_server/out'             # output파일들 저장될 directory


##
for d in ["train", "validation"]:
    register_coco_instances(f"RAhand_{d}", {}, f"./dataset/RA_hand/{d}.json", f"./dataset/RA_hand/images/")


MetadataCatalog.get("RAhand_train").set(thing_classes=[
    '1PP', '1DP', 'Hamate', 'Capitate', 'Trapezoid', 'Trapezium', 'Pisiform', 'Triquetrum', 'Lunate', 'Scaphoid',
    'Ulna', 'Radius', '1MC', '2MC', '3MC', '4MC', '5MC', '2PP', '3PP', '4PP', '5PP', '2IP', '3IP', '4IP', '5IP',
    '2DP', '3DP', '4DP', '5DP'])
RAhand_metadata = MetadataCatalog.get("RAhand_train")


cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("RAhand_train", )
cfg.DATASETS.TEST = ("RAhand_validation", )
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 5000
cfg.SOLVER.GAMMA = 0.8
cfg.SOLVER.STEPS = [1500, 2000, 2500, 3000, 3500, 4000]
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.CHECKPOINT_PERIOD = 2500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 13
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 29
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.TEST.EVAL_PERIOD = 100
cfg.OUTPUT_DIR = output_dir

cfg.MODEL.WEIGHTS = os.path.join(model_dir, "RA_hand.pth")  # 학습된 모델 들어가 있는 곳
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # custom testing threshold
predictor = DefaultPredictor(cfg)

test_dir = test_dir
test_list = os.listdir(test_dir)
test_list.sort()
except_list = []


for file in tqdm(test_list):
    filepath = os.path.join(test_dir, file)
    filename = os.path.splitext(file)[0]
    im = cv2.imread(filepath)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=RAhand_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.axis('off') 
    plt.savefig(f'{output_dir}/{filename}.jpg', dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.close()
