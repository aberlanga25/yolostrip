# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Common modules."""

import ast
import contextlib
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn


from YoloStrip.utils.general import (
    LOGGER,
    ROOT,
    check_suffix,
    check_version,
    yaml_load,
)


def autopad(k, p=None, d=1):
    """Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Applies a convolution, batch normalization, and activation function to an input tensor in a neural network."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Implements a depth-wise convolution layer with optional activation for efficient spatial filtering."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initializes a depth-wise convolution layer with optional activation; args: input channels (c1), output
        channels (c2), kernel size (k), stride (s), dilation (d), and activation flag (act).
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """A depth-wise transpose convolutional layer for upsampling in neural networks, particularly in YOLOv5 models."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initializes a depth-wise transpose convolutional layer for YOLOv5; args: input channels (c1), output channels
        (c2), kernel size (k), stride (s), input padding (p1), output padding (p2).
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    """Transformer layer with multihead attention and linear layers, optimized by removing LayerNorm."""

    def __init__(self, c, num_heads):
        """Initializes a transformer layer, sans LayerNorm for performance, with multihead attention and linear layers.

        See  as described in https://arxiv.org/abs/2010.11929.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Performs forward pass using MultiheadAttention and two linear transformations with residual connections."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    """A Transformer block for vision tasks with convolution, position embeddings, and Transformer layers."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initializes a Transformer block for vision tasks, adapting dimensions if necessary and stacking specified
        layers.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Processes input through an optional convolution, followed by Transformer layers and position embeddings for
        object detection.
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    """A bottleneck layer with optional shortcut and group convolution for efficient feature extraction."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP bottleneck layer for feature extraction with cross-stage partial connections and optional shortcuts."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP bottleneck with optional shortcuts; args: ch_in, ch_out, number of repeats, shortcut bool,
        groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward pass by applying layers, activation, and concatenation on input x, returning feature-
        enhanced output.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    """Implements a cross convolution layer with downsampling, expansion, and optional shortcut."""

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """Implements a CSP Bottleneck module with three convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """Extends the C3 module with cross-convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3x module with cross-convolutions, extending C3 with customizable channel dimensions, groups,
        and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    """C3 module with TransformerBlock for enhanced feature extraction in object detection models."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with TransformerBlock for enhanced feature extraction, accepts channel sizes, shortcut
        config, group, and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    """Extends the C3 module with an SPP layer for enhanced spatial feature extraction and customizable channels."""

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3 module with SPP layer for advanced spatial feature extraction, given channel sizes, kernel
        sizes, shortcut, group, and expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    """Implements a C3 module with Ghost Bottlenecks for efficient feature extraction in YOLOv5."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes YOLOv5's C3 module with Ghost Bottlenecks for efficient feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    """Implements Spatial Pyramid Pooling (SPP) for feature extraction, ref: https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initializes SPP layer with Spatial Pyramid Pooling, ref: https://arxiv.org/abs/1406.4729, args: c1 (input
        channels), c2 (output channels), k (kernel sizes).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Applies convolution and max pooling layers to the input tensor `x`, concatenates results, and returns output
        tensor.
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Implements a fast Spatial Pyramid Pooling (SPPF) layer for efficient feature extraction in YOLOv5 models."""

    def __init__(self, c1, c2, k=5):
        """Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    """Focuses spatial information into channel space using slicing and convolution for efficient feature extraction."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus module to concentrate width-height info into channel space with configurable convolution
        parameters.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Processes input through Focus mechanism, reshaping (b,c,w,h) to (b,4c,w/2,h/2) then applies convolution."""
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Implements Ghost Convolution for efficient feature extraction, see https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes GhostConv with in/out channels, kernel size, stride, groups, and activation; halves out channels
        for efficiency.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Performs forward pass, concatenating outputs of two convolutions on input `x`: shape (B,C,H,W)."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    """Efficient bottleneck layer using Ghost Convolutions, see https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck with ch_in `c1`, ch_out `c2`, kernel size `k`, stride `s`; see
        https://github.com/huawei-noah/ghostnet.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Processes input through conv and shortcut layers, returning their summed output."""
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    """Contracts spatial dimensions into channel dimensions for efficient processing in neural networks."""

    def __init__(self, gain=2):
        """Initializes a layer to contract spatial dimensions (width-height) into channels, e.g., input shape
        (1,64,80,80) to (1,256,40,40).
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor to expand channel dimensions by contracting spatial dimensions, yielding output shape
        `(b, c*s*s, h//s, w//s)`.
        """
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    """Expands spatial dimensions by redistributing channels, e.g., from (1,64,80,80) to (1,16,160,160)."""

    def __init__(self, gain=2):
        """Initializes the Expand module to increase spatial dimensions by redistributing channels, with an optional
        gain factor.

        Example: x(1,64,80,80) to x(1,16,160,160).
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor x to expand spatial dims by redistributing channels, requiring C / gain^2 == 0."""
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    """Concatenates tensors along a specified dimension for efficient tensor manipulation in neural networks."""

    def __init__(self, dimension=1):
        """Initializes a Concat module to concatenate tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along a specified dims; `x` is a list of tensors, `dimension` is an int."""
        return torch.cat(x, self.d)

def export_formats():
    r"""Returns a DataFrame of supported YOLOv5 model export formats and their properties.

    Returns:
        pandas.DataFrame: A DataFrame containing supported export formats and their properties. The DataFrame includes
            columns for format name, CLI argument suffix, file extension or directory name, and boolean flags indicating
            if the export format supports training and detection.

    Examples:
        ```python
        formats = export_formats()
        print(f"Supported export formats:\n{formats}")
        ```

    Notes:
        The DataFrame contains the following columns:
        - Format: The name of the model format (e.g., PyTorch, TorchScript, ONNX, etc.).
        - Include Argument: The argument to use with the export script to include this format.
        - File Suffix: File extension or directory name associated with the format.
        - Supports Training: Whether the format supports training.
        - Supports Detection: Whether the format supports detection.
    """
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


class DetectMultiBackend(nn.Module):
    """YOLOv5 MultiBackend class for inference on various backends including PyTorch, ONNX, TensorRT, and more."""

    def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        """Initializes DetectMultiBackend with support for various inference backends, including PyTorch and ONNX."""
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlpackage
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from YoloStrip.models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):  # dynamic
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_profile_shape(name, 0)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        elif saved_model:  # TF SavedModel
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta["stride"]), meta["names"]
        else:
            raise NotImplementedError(f"ERROR: {w} is not a supported format")

        # class names
        if "names" not in locals():
            names = yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}
        if names[0] == "n01440764" and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / "data/ImageNet.yaml")["names"]  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        """Performs YOLOv5 inference on input images with options for augmentation and visualization."""
        _b, _ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input["quantization"]
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if int8:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            if len(y) == 2 and len(y[1].shape) != 4:
                y = list(reversed(y))
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """Converts a NumPy array to a torch tensor, maintaining device compatibility."""
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """Performs a single inference warmup to initialize model weights, accepting an `imgsz` tuple for image size."""
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """Determines model type from file path or URL, supporting various export formats.

        Example: path='path/to/model.onnx' -> type=onnx
        """
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        #from export import export_formats
        from YoloStrip.utils.downloads import is_url

        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return [*types, triton]

    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        """Loads metadata from a YAML file, returning strides and names if the file exists, otherwise `None`."""
        if f.exists():
            d = yaml_load(f)
            return d["stride"], d["names"]  # assign stride, names
        return None, None