#ЗАПУСК (указать в конфигураторе)
#--source Peoples.mp4
# --yolo_model weights/crowdhuman_yolov5m.pt
# --show-vid
# --save-vid

import torch
from src.detectors import detect
from src.argparser import parser


if __name__ == '__main__':

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
