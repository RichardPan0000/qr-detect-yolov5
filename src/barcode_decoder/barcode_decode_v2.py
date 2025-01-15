import cv2
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version
from ultralytics.utils.checks import check_font, check_version, is_ascii
from collections import defaultdict
from qreader import QReader
from src.myutils.QReader_v2 import QReader_v2
from src.myutils.time_utils import method_using_time

# count_map=defaultdict(int)
# pic_count_map=dict()
# 使用 defaultdict 创建嵌套字典
pic_count_map = defaultdict(lambda: defaultdict(int))

idx=0

# 锐化处理
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# 放大ROI区域
def resize_roi(roi, scale=2):
    width = int(roi.shape[1] * scale)
    height = int(roi.shape[0] * scale)
    return cv2.resize(roi, (width, height), interpolation=cv2.INTER_CUBIC)

# 增强对比度
def increase_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# 灰度化
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'
class BarcodeAnnotator:
    """
    Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image or numpy array): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype or ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
        skeleton (List[List[int]]): Skeleton structure for keypoints.
        limb_color (List[int]): Color palette for limbs.
        kpt_color (List[int]): Color palette for keypoints.
    """

    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        if self.pil:  # use PIL
            self.im = im if input_is_pil else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
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

    def get_txt_color(self, color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Assign text color based on background color."""
        if color in self.dark_colors:
            return 104, 31, 17
        elif color in self.light_colors:
            return 255, 255, 255
        else:
            return txt_color
    def barcode_decode_and_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
        """
        Draws a bounding box to image with label.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            rotated (bool, optional): Variable used to check if task is OBB
        """
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil or not is_ascii(label):
            if rotated:
                p1 = box[0]
                self.draw.polygon([tuple(b) for b in box], width=self.lw, outline=color)  # PIL requires tuple box
            else:
                p1 = (box[0], box[1])
                self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = p1[1] >= h  # label fits outside box
                if p1[0] > self.im.size[0] - w:  # size is (w, h), check if label extend beyond right side of image
                    p1 = self.im.size[0] - w, p1[1]
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            if rotated:
                p1 = [int(b) for b in box[0]]
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)  # cv2 requires nparray box
            else:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
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

    @method_using_time
    def barcode_decode(self, box,pic_path):
        time1=time.time()
        box[0] = box[0] - 10
        box[1] = box[1] - 10
        box[2]=box[2]+10
        box[3] = box[3] + 10
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # roi=self.im[int(box[1]):int(box[1])+int(box[3]),int(box[0]):int(box[0])+int(box[2])] # h,w 如果是CV2的话，应该是h,w排列的
        roi = self.im[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0])]
        roi_area =(p2[0]-p1[0])*(p2[1]-p1[1])
        time2=time.time()-time1
        print('时间 time2',time2)
        if roi_area <=0:
            return '',None
        global  idx

        # qreader_reader = QReader()
        qreader_reader=QReader_v2()
        print('roi_area',roi_area)
        try :
            qreader_out,detections = qreader_reader.detect_and_decode(image=roi,return_detections=True) #  roi ndarray(178,173,3)
            # print('detections',detections)
            # ol={"confidence": 0.9517315030097961, 'bbox_xyxy': np.array([35.081, 18.645, 187.22, 187.7], dtype=np.float32),
            #     'bbox_xyxyn': None, 'cxcy': (111.15029907226562, 103.17047882080078), 'cxcyn': None,
            #     'wh': (152.13897705078125, 169.05043029785156), 'whn': None, 'polygon_xy': None, 'polygon_xyn': None,
            #     'quad_xy': None, 'quad_xyn': None, 'padded_quad_xy': None,
            #      'padded_quad_xyn': None, 'image_shape': (213, 215)}
            # detections6=[ol]
            # qreader_out2 = qreader_reader.decode(image=roi,detection_result=ol)
            # print('detections',detections)
            # print('qreader_out2',qreader_out2)
        except Exception as e:
            qreader_out=''
            print(e)
        time3=time.time()-time1-time2
        print('时间 time3',time3)
        # print(f'{idx} qreader_out',qreader_out)
        idx+=1
        if qreader_out:
            barcode_dict = {'box': box, 'barcode_content': qreader_out[0]}
            pic_count_map[pic_path.stem][qreader_out[0]] +=1
            # print(pic_count_map)
            time4=time.time()-time1-time2-time3
            print('时间time4',time4)
            quad_xy=detections[0]['quad_xy']
            quad_xy=quad_xy+[p1[0],p1[1]]
            return qreader_out[0],quad_xy
        else:
            return '',None

    @method_using_time
    def barcode_decode_v2(self, box, pic_path):
        time1 = time.time()
        box[0] = box[0] - 10
        box[1] = box[1] - 10
        box[2] = box[2] + 10
        box[3] = box[3] + 10
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # roi=self.im[int(box[1]):int(box[1])+int(box[3]),int(box[0]):int(box[0])+int(box[2])] # h,w 如果是CV2的话，应该是h,w排列的
        roi = self.im[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0])]
        roi_area = (p2[0] - p1[0]) * (p2[1] - p1[1])
        time2 = time.time() - time1
        print('时间 time2', time2)
        if roi_area <= 0:
            return '', None
        global idx

        qreader_reader = QReader()
        print('roi_area', roi_area)
        try:
            qreader_out, detections = qreader_reader.detect_and_decode(image=roi, return_detections=True)
            # print('detections',detections)
            # ol={"confidence": 0.9517315030097961, 'bbox_xyxy': np.array([35.081, 18.645, 187.22, 187.7], dtype=np.float32),
            #     'bbox_xyxyn': None, 'cxcy': (111.15029907226562, 103.17047882080078), 'cxcyn': None,
            #     'wh': (152.13897705078125, 169.05043029785156), 'whn': None, 'polygon_xy': None, 'polygon_xyn': None,
            #     'quad_xy': None, 'quad_xyn': None, 'padded_quad_xy': None,
            #      'padded_quad_xyn': None, 'image_shape': (213, 215)}
            # detections6=[ol]
            # qreader_out2 = qreader_reader.decode(image=roi,detection_result=ol)
            # print('detections',detections)
            # print('qreader_out2',qreader_out2)
        except Exception as e:
            qreader_out = ''
            print(e)
        time3 = time.time() - time1 - time2
        print('时间 time3', time3)
        # print(f'{idx} qreader_out',qreader_out)
        idx += 1
        if qreader_out:
            barcode_dict = {'box': box, 'barcode_content': qreader_out[0]}
            pic_count_map[pic_path.stem][qreader_out[0]] += 1
            # print(pic_count_map)
            time4 = time.time() - time1 - time2 - time3
            print('时间time4', time4)
            quad_xy = detections[0]['quad_xy']
            quad_xy = quad_xy + [p1[0], p1[1]]
            return qreader_out[0], quad_xy,time3
        else:
            return '', None

    @method_using_time
    def barcode_decode_batch(self, boxes:list, pic_path):
        roies_input=[]
        decode_str=[]  # 如果面积为0，则给''
        quad_xyes=[]
        quad_origin_p1_list=[]
        for box in boxes:
            time1 = time.time()
            box[0] = box[0] - 10
            box[1] = box[1] - 10
            box[2] = box[2] + 10
            box[3] = box[3] + 10
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            # roi=self.im[int(box[1]):int(box[1])+int(box[3]),int(box[0]):int(box[0])+int(box[2])] # h,w 如果是CV2的话，应该是h,w排列的
            roi = self.im[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0])]
            roies_input.append(roi)
            roi_area = (p2[0] - p1[0]) * (p2[1] - p1[1])
            time2 = time.time() - time1
            quad_origin_p1_list.append(p1)
            print('时间 time2', time2)

        t_b=time.time()
        qreader_reader = QReader_v2()
        print('模型加载时间',time.time()-t_b)
        print('roi_area', roi_area)
        try:
            batch_qreader_out, batch_detections = qreader_reader.detect_and_decode_batch(images=roies_input, return_detections=True)
            print('解码结果',batch_qreader_out)
        except Exception as e:
            print(e)
            return decode_str,quad_xyes

        time3 = time.time() - time1 - time2
        print('时间 time3', time3)
        if batch_qreader_out:
            for i,qreader_out in enumerate(batch_qreader_out):
                if qreader_out:
                    decode_str.append(qreader_out[0])
                else:
                    decode_str.append("")
        if  batch_detections:
            for detections,quad_origin_p1 in zip(batch_detections,quad_origin_p1_list):
                quad_xy = detections[0]['quad_xy']
                p_x,p_y=quad_origin_p1
                quad_xy = quad_xy + [p_x, p_y]
                quad_xyes.append(quad_xy)

        return decode_str,quad_xyes
