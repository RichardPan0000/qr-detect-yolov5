import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from barcode_decoder.barcode_decode_v2 import BarcodeAnnotator

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    non_max_suppression,
    print_args,
    scale_boxes,
)
from myutils.find_angle_vertex import find_right_angle_vertex,sort_boxes_by_center_angle
from utils.torch_utils import smart_inference_mode
from myutils.point_mapping import get_transformed_box, get_transformed_box_four, get_transformed_quad_xyes_four, \
    get_transformed_quad_xyes
# from myutils.find_which_box2 import TableAffineClass,MarkerAffineClass
from myutils.find_which_box_calling import TableAffineClass,MarkerAffineClass
from dataclasses import dataclass,asdict
'''
返回的东西:
    boxes: 转换后的或者是原始的，列表
    angle: 角度，如果无tile,就是一个都是90的列表
    table: 如果要求返回表格格式，则返回到底在哪个几个格子(row,col)
    table_attr: 如果要求返回表格格式，则返回表格有几行几列rows,cols,col_width,row_height,top_left
'''
# class ResponseData(BaseModel):
#     boxes:Optional[np.ndarray]=None
#     angles:Optional[np.ndarray]=None
#     table:Optional[dict]=None
#     table_attr:Optional[dict]=None

@dataclass
class ResponseData:
    def __init__(self):
        self.boxes=None
        self.angles=None
        self.table=None
        self.table_attr=None
        self.width_ret=None
        self.height_ret=None

    def to_json(self):
        return json.dumps(asdict(self))

    def add_attribute(self,name,value):
        setattr(self,name,value)


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device=None,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    tilt=False,  # 倾斜，判断是表格还是白纸那种类型
    use_config=False,  # 是否使用配置文件
    qr_anchor_config_path=None,  # 配置文件路径
    class_to_use=None,
    model=None
):
    """
    Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc.

    Args:
        weights (str | Path): Path to the model weights file or a Triton URL. Default is 'yolov5s.pt'.
        source (str | Path): Input source, which can be a file, directory, URL, glob pattern, screen capture, or webcam
            index. Default is 'data/images'.
        data (str | Path): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        imgsz (tuple[int, int]): Inference image size as a tuple (height, width). Default is (640, 640).
        conf_thres (float): Confidence threshold for detections. Default is 0.25.
        iou_thres (float): Intersection Over Union (IOU) threshold for non-max suppression. Default is 0.45.
        max_det (int): Maximum number of detections per image. Default is 1000.
        device (str): CUDA device identifier (e.g., '0' or '0,1,2,3') or 'cpu'. Default is an empty string, which uses the
            best available device.
        view_img (bool): If True, display inference results using OpenCV. Default is False.
        save_txt (bool): If True, save results in a text file. Default is False.
        save_csv (bool): If True, save results in a CSV file. Default is False.
        save_conf (bool): If True, include confidence scores in the saved results. Default is False.
        save_crop (bool): If True, save cropped prediction boxes. Default is False.
        nosave (bool): If True, do not save inference images or videos. Default is False.
        classes (list[int]): List of class indices to filter detections by. Default is None.
        agnostic_nms (bool): If True, perform class-agnostic non-max suppression. Default is False.
        augment (bool): If True, use augmented inference. Default is False.
        visualize (bool): If True, visualize feature maps. Default is False.
        update (bool): If True, update all models' weights. Default is False.
        project (str | Path): Directory to save results. Default is 'runs/detect'.
        name (str): Name of the current experiment; used to create a subdirectory within 'project'. Default is 'exp'.
        exist_ok (bool): If True, existing directories with the same name are reused instead of being incremented. Default is
            False.
        line_thickness (int): Thickness of bounding box lines in pixels. Default is 3.
        hide_labels (bool): If True, do not display labels on bounding boxes. Default is False.
        hide_conf (bool): If True, do not display confidence scores on bounding boxes. Default is False.
        half (bool): If True, use FP16 half-precision inference. Default is False.
        dnn (bool): If True, use OpenCV DNN backend for ONNX inference. Default is False.
        vid_stride (int): Stride for processing video frames, to skip frames between processing. Default is 1.

    Returns:
        None

    Examples:
        ```python
        from ultralytics import run

        # Run inference on an image
        run(source='data/images/example.jpg', weights='yolov5s.pt', device='0')

        # Run inference on a video with specific confidence threshold
        run(source='data/videos/example.mp4', weights='yolov5s.pt', conf_thres=0.4, device='0')
        ```
    """
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    t_1=time.time()
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print('模型加载时间',time.time()-t_1) # 差不多0.2s左右。
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        t_sii=time.time()
        # Inference
        with dt[1]:
            visualize = False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)

                    LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        print('时间 inference time',time.time()-t_sii) # 差不多0.01s左右。
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # pred = keep_one_non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)


        response_data=ResponseData()

        # Process predictions
        for i, det in enumerate(pred):  # per image

            # 记录三个标记的字典
            mark1_loc_list=[]
            mark2_loc_list=[]
            all_mark_loc_list=[]
            mark_clazz_list=[]
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            imc_bar=im0.copy()
            barcode_annotator=BarcodeAnnotator(imc_bar, line_width=line_thickness, example=str(names))
            topo_idx=0
            topo_dict=dict()
            qr_boxes=[]
            quad_xyes=[]
            multi_roi=[]
            t_single=0.00

            det_copy=det.clone()
            if len(det_copy):
                # Rescale boxes from img_size to im0 size
                det_copy[:, :4] = scale_boxes(im.shape[2:], det_copy[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det_copy):
                    c = int(cls)  # integer class
                    if c==1:
                        multi_roi.append((xyxy,p))


            '''
                存储多个码之后，尝试一把送进去看看效果，看看时间耗时在哪里。
            '''
            if len(multi_roi)>0:
                # # 先把多个码的box画出来

                multi_roi_input=[roi for roi,p in multi_roi ]
                multi_p=[p for roi,p in multi_roi ]
                start_t=time.time()
                decode_strs,quad_xyes=barcode_annotator.barcode_decode_batch(multi_roi_input,multi_p[0])
                print('batch_decode时间',time.time()-start_t)

            iddx=0
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    # todo 检测ROI 区域，进行解码
                    # print('xyxy:', xyxy)  # 左上，右下两个点坐标。这个坐标是还原过后
                    # 的
                    if c==1:
                        # decode_str,quad_xy,time3=barcode_annotator.barcode_decode_v2(xyxy,p)
                        time3=0
                        decode_str=decode_strs[iddx]
                        quad_xy=quad_xyes[iddx]
                        iddx+=1
                        t_single+=time3
                        multi_roi.append((xyxy,p))
                        if decode_str!='' and quad_xy is not None:
                            qr_boxes.append(xyxy)
                            # quad_xyes.append(quad_xy)
                    elif c==0:
                        mark1_loc_list.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])])
                        all_mark_loc_list.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])])
                        mark_clazz_list.append(c) # 标记类别
                        decode_str=''
                    else :
                        mark2_loc_list.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])])
                        all_mark_loc_list.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])])
                        mark_clazz_list.append(c) # 标记类别
                        decode_str=''

                    # if save_txt:  # Write to file
                    #     if save_format == 0:
                    #         coords = (
                    #             (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    #         )  # normalized xywh
                    #     else:
                    #         coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy
                    #     line = (cls, *coords, conf) if save_conf else (cls, *coords)  # label format
                    #     with open(f"{txt_path}.txt", "a") as f:
                    #         f.write(("%g " * len(line)).rstrip() % line + "\n")

                    # if save_img or save_crop or view_img:  # Add bbox to image
                    #     c = int(cls)  # integer class
                    #     label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        # label = None if hide_labels else (names[c])

                        # annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.box_label(xyxy, decode_str, color=colors(c, True))

                        # 打标,只针对能够解析出来的打标画框；并且标签写解析出来的label，单独存一张图片。
                        # 用一个字典来存 那些解析出来的box。
                        # barcode_annotator.barcode_decode_and_label(xyxy, label, color=colors(c, True))
                        # barcode_annotator.barcode_decode_and_label(xyxy, decode_str, color=colors(c, True))

                    # if save_crop:
                    #     save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)



            # 再进行记录

            # 记录三个标记的位置。处理三个boxes
            if len(mark1_loc_list)==3 and len(mark2_loc_list)<1:
                # class_to_use = opt.class_to_use
                if use_config and qr_anchor_config_path:
                    if class_to_use == "Marker":
                        boxAffineClass = MarkerAffineClass(qr_anchor_config_path)
                    else:
                        boxAffineClass = TableAffineClass(qr_anchor_config_path)
                else:
                    if class_to_use == "Marker":
                        boxAffineClass = MarkerAffineClass()
                    else:
                        boxAffineClass = TableAffineClass()

                angle_box_dict=find_right_angle_vertex(*mark1_loc_list)

                src_point=angle_box_dict.values()
                src_point=np.array(list(src_point),dtype=np.float32)
                src_point[:]=src_point[[1,2,0]]
                dst_point=np.array(boxAffineClass.get_dst_point_four(),dtype=np.float32)[1:4]  #就拿mark1,mark2,mark3 就行
                transformed_quad_xyes = get_transformed_quad_xyes(quad_xyes, src_point, dst_point)
                transformed_boxes=get_transformed_box(qr_boxes,src_point,dst_point)
                # response_data.boxes=transformed_boxes
                response_data.add_attribute('boxes',transformed_boxes)
                # 进行判断，到底使用什么处理方式

                if class_to_use == "Marker":
                    angle_rad,rec_centers,width_ret,height_ret=boxAffineClass.get_angle_via_quadxy(
                         transformed_quad_xyes, tilt=tilt
                    )
                    # response_data.angles=angle_rad
                    response_data.add_attribute('angles', angle_rad)
                    response_data.add_attribute('rec_centers', rec_centers)
                    response_data.add_attribute('width_ret', width_ret)
                    response_data.add_attribute('height_ret', height_ret)

                else:
                    table_box_dict=boxAffineClass.get_table_boxes(
                        transformed_boxes
                    )
                    # response_data.table=table_box_dict
                    response_data.add_attribute('table', table_box_dict)
            elif len(mark1_loc_list)==3 and len(mark2_loc_list)==1:
                # Use the class specified in the command-line argument
                # class_to_use = opt.class_to_use
                if use_config and qr_anchor_config_path:
                    if class_to_use == "Marker":
                        boxAffineClass = MarkerAffineClass(qr_anchor_config_path)
                    else:
                        boxAffineClass = TableAffineClass(qr_anchor_config_path)
                else:
                    if class_to_use == "Marker":
                        boxAffineClass = MarkerAffineClass()
                    else:
                        boxAffineClass = TableAffineClass()

                sorted_boxes = sort_boxes_by_center_angle(all_mark_loc_list, mark_clazz_list)
                src_point = np.array(list(sorted_boxes), dtype=np.float32)

                dst_point = np.array(boxAffineClass.get_dst_point_four(), dtype=np.float32)
                transformed_boxes = get_transformed_box_four(qr_boxes, src_point, dst_point)
                transformed_quad_xyes = get_transformed_quad_xyes_four(quad_xyes, src_point, dst_point)
                # response_data.boxes=transformed_boxes


                if class_to_use == "Marker":
                    # Use MarkerAffineClass visualization method
                    angle_rad,rec_centers,width_ret,height_ret=boxAffineClass.get_angle_via_quadxy(transformed_quad_xyes,tilt=tilt)
                    # response_data.angles=angle_rad
                    response_data.add_attribute('angles', angle_rad)
                    response_data.add_attribute('rec_centers', rec_centers)
                    response_data.add_attribute('width_ret', width_ret)
                    response_data.add_attribute('height_ret', height_ret)
                else:
                    # Use TableAffineClass visualization method
                    table_box_dict=boxAffineClass.get_table_boxes(
                        transformed_boxes
                    )
                    # response_data.table=table_box_dict
                    response_data.add_attribute('table', table_box_dict)
            else:
                # If no anchors, assume it's a plain paper form and return
                pass

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    angles=response_data.angles
    if isinstance(angles,np.ndarray):
        angles=angles.tolist()
    boxes=response_data.boxes
    if isinstance(boxes,np.ndarray):
        boxes=boxes.tolist()
    rec_centers=response_data.rec_centers

    if isinstance(rec_centers,np.ndarray):
        rec_centers=rec_centers.tolist()

    return {
        'angles':angles,
        # 'boxes':boxes,''
        'rec_centers':rec_centers,
        'width_ret':response_data.width_ret,
        'height_ret':response_data.height_ret

    }


def parse_opt():
    """
    Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        --weights (str | list[str], optional): Model path or Triton URL. Defaults to ROOT / 'yolov5s.pt'.
        --source (str, optional): File/dir/URL/glob/screen/0(webcam). Defaults to ROOT / 'data/images'.
        --data (str, optional): Dataset YAML path. Provides dataset configuration information.
        --imgsz (list[int], optional): Inference size (height, width). Defaults to [640].
        --conf-thres (float, optional): Confidence threshold. Defaults to 0.25.
        --iou-thres (float, optional): NMS IoU threshold. Defaults to 0.45.
        --max-det (int, optional): Maximum number of detections per image. Defaults to 1000.
        --device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        --view-img (bool, optional): Flag to display results. Defaults to False.
        --save-txt (bool, optional): Flag to save results to *.txt files. Defaults to False.
        --save-csv (bool, optional): Flag to save results in CSV format. Defaults to False.
        --save-conf (bool, optional): Flag to save confidences in labels saved via --save-txt. Defaults to False.
        --save-crop (bool, optional): Flag to save cropped prediction boxes. Defaults to False.
        --nosave (bool, optional): Flag to prevent saving images/videos. Defaults to False.
        --classes (list[int], optional): List of classes to filter results by, e.g., '--classes 0 2 3'. Defaults to None.
        --agnostic-nms (bool, optional): Flag for class-agnostic NMS. Defaults to False.
        --augment (bool, optional): Flag for augmented inference. Defaults to False.
        --visualize (bool, optional): Flag for visualizing features. Defaults to False.
        --update (bool, optional): Flag to update all models in the model directory. Defaults to False.
        --project (str, optional): Directory to save results. Defaults to ROOT / 'runs/detect'.
        --name (str, optional): Sub-directory name for saving results within --project. Defaults to 'exp'.
        --exist-ok (bool, optional): Flag to allow overwriting if the project/name already exists. Defaults to False.
        --line-thickness (int, optional): Thickness (in pixels) of bounding boxes. Defaults to 3.
        --hide-labels (bool, optional): Flag to hide labels in the output. Defaults to False.
        --hide-conf (bool, optional): Flag to hide confidences in the output. Defaults to False.
        --half (bool, optional): Flag to use FP16 half-precision inference. Defaults to False.
        --dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Defaults to False.
        --vid-stride (int, optional): Video frame-rate stride, determining the number of frames to skip in between
            consecutive frames. Defaults to 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Example:
        ```python
        from ultralytics import YOLOv5
        args = YOLOv5.parse_opt()
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train/exp119/weights/best.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images/多码", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/qr-custom-data.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--enable_clear", type=bool, default=True, help="clear the resolution,1280")
    parser.add_argument("--conf-thres", type=float, default=0.7, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold") # 这个大一点，否则容易过滤挺多
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # parser.add_argument("--device", default="1", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument( "--save-format",type=int,default=0,help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",)
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    # parser.add_argument("--visualize", action="store_true",default='True', help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--tilt", type=bool, default=False, help="qrcode tilt or not")
    parser.add_argument("--use_config", type=bool, default=True, help="whether to use configuration file")
    # parser.add_argument("--qr_anchor_config_path", type=str, default=ROOT/'config/qr_anchor_config.yaml', help="path to the configuration file")
    parser.add_argument("--qr_anchor_config_path", type=str, default=ROOT/'config/qr_anchor_config_transparent_paper.yaml', help="path to the configuration file")
    parser.add_argument("--class_to_use", type=str, choices=["Table", "Marker"], default="Marker",
                        help="Choose which class to use for processing: 'Table' or 'Marker'")
    opt = parser.parse_args()
    if opt.enable_clear==True:
        opt.imgsz=[1280]
    del opt.enable_clear
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":

    for i in range(10):
        start_time=time.time()
        opt = parse_opt()
        main(opt)
        end_time=time.time()
        print('完成时间：',end_time-start_time)
