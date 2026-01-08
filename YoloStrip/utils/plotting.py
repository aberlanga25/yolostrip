import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from YoloStrip.utils.general import check_version, check_font

class Annotator:
    """Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image | np.ndarray): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype | ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
        skeleton (list[list[int]]): Skeleton structure for keypoints.
        limb_color (list[int]): Color palette for limbs.
        kpt_color (list[int]): Color palette for keypoints.
        dark_colors (set): Set of colors considered dark for text contrast.
        light_colors (set): Set of colors considered light for text contrast.

    Examples:
        >>> from ultralytics.utils.plotting import Annotator
        >>> im0 = cv2.imread("test.png")
        >>> annotator = Annotator(im0, line_width=10)
        >>> annotator.box_label([10, 10, 100, 100], "person", (255, 0, 0))
    """

    def __init__(
        self,
        im,
        line_width: int | None = None,
        font_size: int | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        if not input_is_pil:
            if im.shape[2] == 1:  # handle grayscale
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 2:  # handle 2-channel images
                im = np.ascontiguousarray(np.dstack((im, np.zeros_like(im[..., :1]))))
            elif im.shape[2] > 3:  # multispectral
                im = np.ascontiguousarray(im[..., :3])
        if self.pil:  # use PIL
            self.im = im if input_is_pil else Image.fromarray(im)  # stay in BGR since color palette is in BGR
            if self.im.mode not in {"RGB", "RGBA"}:  # multispectral
                self.im = self.im.convert("RGB")
            self.draw = ImageDraw.Draw(self.im, "RGBA")
            try:
                font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            # Deprecation fix for w, h = getsize(string) -> _, _, w, h = getbox(string)
            if check_version(pil_version, "9.2.0"):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # text width, height
        else:  # use cv2
            assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)  # font thickness
            self.sf = self.lw / 3  # font scale
        # Pose
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def get_txt_color(self, color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)) -> tuple:
        """Assign text color based on background color.

        Args:
            color (tuple, optional): The background color of the rectangle for text (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).

        Returns:
            (tuple): Text color for label.

        Examples:
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_txt_color(color=(104, 31, 17))  # return (255, 255, 255)
        """
        if color in self.dark_colors:
            return 104, 31, 17
        elif color in self.light_colors:
            return 255, 255, 255
        else:
            return txt_color

    def box_label(self, box, label: str = "", color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)):
        """Draw a bounding box on an image with a given label.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str, optional): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).

        Examples:
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.box_label(box=[10, 20, 30, 40], label="person")
        """
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, torch.Tensor):
            box = box.tolist()

        multi_points = isinstance(box[0], list)  # multiple points with shape (n, 2)
        p1 = [int(b) for b in box[0]] if multi_points else (int(box[0]), int(box[1]))
        if self.pil:
            self.draw.polygon(
                [tuple(b) for b in box], width=self.lw, outline=color
            ) if multi_points else self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = p1[1] >= h  # label fits outside box
                if p1[0] > self.im.size[0] - w:  # size is (w, h), check if label extend beyond right side of image
                    p1 = self.im.size[0] - w, p1[1]
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                # self.draw.text([box[0], box[1]], label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            cv2.polylines(
                self.im, [np.asarray(box, dtype=int)], True, color, self.lw
            ) if multi_points else cv2.rectangle(
                self.im, p1, (int(box[2]), int(box[3])), color, thickness=self.lw, lineType=cv2.LINE_AA
            )
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                h += 3  # add pixels to pad text
                outside = p1[1] >= h  # label fits outside box
                if p1[0] > self.im.shape[1] - w:  # shape is (h, w), check if label extend beyond right side of image
                    p1 = self.im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA,
                )

    def masks(self, masks, colors, im_gpu: torch.Tensor = None, alpha: float = 0.5, retina_masks: bool = False):
        """Plot masks on image.

        Args:
            masks (torch.Tensor | np.ndarray): Predicted masks with shape: [n, h, w]
            colors (list[list[int]]): Colors for predicted masks, [[r, g, b] * n]
            im_gpu (torch.Tensor | None): Image is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float, optional): Mask transparency: 0.0 fully transparent, 1.0 opaque.
            retina_masks (bool, optional): Whether to use high resolution masks or not.
        """
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        if im_gpu is None:
            assert isinstance(masks, np.ndarray), "`masks` must be a np.ndarray if `im_gpu` is not provided."
            overlay = self.im.copy()
            for i, mask in enumerate(masks):
                overlay[mask.astype(bool)] = colors[i]
            self.im = cv2.addWeighted(self.im, 1 - alpha, overlay, alpha, 0)
        else:
            assert isinstance(masks, torch.Tensor), "'masks' must be a torch.Tensor if 'im_gpu' is provided."
            if len(masks) == 0:
                self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
                return
            if im_gpu.device != masks.device:
                im_gpu = im_gpu.to(masks.device)

            ih, iw = self.im.shape[:2]
            if not retina_masks:
                # Use scale_masks to properly remove padding and upsample, convert bool to float first
                masks = ops.scale_masks(masks[None].float(), (ih, iw))[0] > 0.5
                # Convert original BGR image to RGB tensor
                im_gpu = (
                    torch.from_numpy(self.im).to(masks.device).permute(2, 0, 1).flip(0).contiguous().float() / 255.0
                )

            colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)
            colors = colors[:, None, None]  # shape(n,1,1,3)
            masks = masks.unsqueeze(3)  # shape(n,h,w,1)
            masks_color = masks * (colors * alpha)  # shape(n,h,w,3)
            inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
            mcs = masks_color.max(dim=0).values  # shape(h,w,3)

            im_gpu = im_gpu.flip(dims=[0]).permute(1, 2, 0).contiguous()  # shape(h,w,3)
            im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
            self.im[:] = (im_gpu * 255).byte().cpu().numpy()
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)

    def kpts(
        self,
        kpts,
        shape: tuple = (640, 640),
        radius: int | None = None,
        kpt_line: bool = True,
        conf_thres: float = 0.25,
        kpt_color: tuple | None = None,
    ):
        """Plot keypoints on the image.

        Args:
            kpts (torch.Tensor): Keypoints, shape [17, 3] (x, y, confidence).
            shape (tuple, optional): Image shape (h, w).
            radius (int, optional): Keypoint radius.
            kpt_line (bool, optional): Draw lines between keypoints.
            conf_thres (float, optional): Confidence threshold.
            kpt_color (tuple, optional): Keypoint color (B, G, R).

        Notes:
            - `kpt_line=True` currently only supports human pose plotting.
            - Modifies self.im in-place.
            - If self.pil is True, converts image to numpy array and back to PIL.
        """
        radius = radius if radius is not None else self.lw
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(kpts):
            color_k = kpt_color or (self.kpt_color[i].tolist() if is_pose else colors(i))
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < conf_thres or conf2 < conf_thres:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(
                    self.im,
                    pos1,
                    pos2,
                    kpt_color or self.limb_color[i].tolist(),
                    thickness=int(np.ceil(self.lw / 2)),
                    lineType=cv2.LINE_AA,
                )
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width: int = 1):
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text: str, txt_color: tuple = (255, 255, 255), anchor: str = "top", box_color: tuple = ()):
        """Add text to an image using PIL or cv2.

        Args:
            xy (list[int]): Top-left coordinates for text placement.
            text (str): Text to be drawn.
            txt_color (tuple, optional): Text color (R, G, B).
            anchor (str, optional): Text anchor position ('top' or 'bottom').
            box_color (tuple, optional): Box color (R, G, B, A) with optional alpha.
        """
        if self.pil:
            w, h = self.font.getsize(text)
            if anchor == "bottom":  # start y from font bottom
                xy[1] += 1 - h
            for line in text.split("\n"):
                if box_color:
                    # Draw rectangle for each line
                    w, h = self.font.getsize(line)
                    self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=box_color)
                self.draw.text(xy, line, fill=txt_color, font=self.font)
                xy[1] += h
        else:
            if box_color:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
                h += 3  # add pixels to pad text
                outside = xy[1] >= h  # label fits outside box
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, box_color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """Update `self.im` from a NumPy array or PIL image."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self, pil=False):
        """Return annotated image as array or PIL image."""
        im = np.asarray(self.im)  # self.im is in BGR
        return Image.fromarray(im[..., ::-1]) if pil else im

    def show(self, title: str | None = None):
        """Show the annotated image."""
        im = Image.fromarray(np.asarray(self.im)[..., ::-1])  # Convert BGR NumPy array to RGB PIL Image
        if IS_COLAB or IS_KAGGLE:  # cannot use IS_JUPYTER as it runs for all IPython environments
            try:
                display(im)  # noqa - display() function only available in ipython environments
            except ImportError as e:
                LOGGER.warning(f"Unable to display image in Jupyter notebooks: {e}")
        else:
            im.show(title=title)

    def save(self, filename: str = "image.jpg"):
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, np.asarray(self.im))

    @staticmethod
    def get_bbox_dimension(bbox: tuple | list):
        """Calculate the dimensions and area of a bounding box.

        Args:
            bbox (tuple | list): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).

        Returns:
            width (float): Width of the bounding box.
            height (float): Height of the bounding box.
            area (float): Area enclosed by the bounding box.

        Examples:
            >>> from ultralytics.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_bbox_dimension(bbox=[10, 20, 30, 40])
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height
