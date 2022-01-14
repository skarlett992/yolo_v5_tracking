import argparse

# подключение субмодуля [submodule "yolov5"] url = https://github.com/ultralytics/yolov5.git
import sys
import argparse
import os
from pathlib import Path

sys.path.insert(0, './yolov5')
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

parser = argparse.ArgumentParser()
parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
parser.add_argument('--source', type=str, help='source')  # file/folder
parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--evaluate', action='store_true', help='augmented inference')
parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
parser.add_argument('--visualize', action='store_true', help='visualize features')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')