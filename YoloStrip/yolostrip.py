import numpy as np
import torch
from typing import Any

from YoloStrip.models.common import DetectMultiBackend
from YoloStrip.utils.torch_utils import select_device, smart_inference_mode
from YoloStrip.utils.general import check_img_size, non_max_suppression, scale_boxes
from YoloStrip.utils.augmentations import letterbox


class YoloStrip():
    def __init__(self,
                 weights="yolov5x6.pt",
                 device="",
                 imgsz=640,
                 conf_thres = 0.25,
                 classes = None,
                 ) -> None:
        imgsz = [imgsz]
        imgsz *= 2 
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, self.stride)
        self.model.eval()
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else 1, 3, *self.imgsz))

        self.conf_thres = conf_thres
        self.classes = classes

    @smart_inference_mode()
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        assert isinstance(args[0], np.ndarray), "Input must be a NumPy array"
        im0 = args[0]
        im = letterbox(im0, self.imgsz, stride=self.stride)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im)
        pred = non_max_suppression(pred, self.conf_thres, 0.45, self.classes, False, max_det=1000)
        
        for det in pred:
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        return pred

