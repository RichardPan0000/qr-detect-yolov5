import time

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.dataloaders import LoadMemoryImages
from utils.torch_utils import select_device
from utils.plots import Annotator
from models.common import DetectMultiBackend
from utils.plots import Annotator,colors
import platform
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
import numpy as np
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from utils.torch_utils import select_device, smart_inference_mode

def run_inference(images_list,imgsz=(640,640),device='cpu',dnn=False,half=False,augment=False,conf_thres=0.25,iou_thres=0.45,max_det=1000,save_dir='.'):
    '''

    return num_det is a list, which include number detection. number detection contains x1,y1,x2,y2,obj_conf,cls
    '''


    data=ROOT / "data/number-custom-data.yaml"
    # weights=ROOT / "checkpoint/num_det_weights/weights/best.pt"  # model path or triton URL
    weights=ROOT / "checkpoint/num_det_weights2/weights/best.pt"  # model path or triton URL
    # Load model
    device = select_device(device)
    agnostic_nms=False
    visualize = False
    save_crop=False
    classes=None  # filter by class: --class 0, or --class 0 2 3
    hide_conf=False  # hide confidences
    line_thickness=3  # bounding box thickness (pixels)

    t1=time.time()
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print('数字模型加载时间',time.time()-t1)
    bs = 1  # batch_size

    # images_list = []
    # for _ in range(5):
    #     img_tensor = torch.randint(0, 1, size=(1280, 1280, 3), dtype=torch.float32)
    #     img_numpy = img_tensor.numpy()
    #     images_list.append(img_numpy)

    dataset = LoadMemoryImages(images_list, img_size=imgsz, stride=stride, auto=pt)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    number_det=[]
    t_start=time.time()
    for im, index, s in dataloader:  # 只要看出来这个img，是什么格式即可。然后调用，上面就可以不要了


        img0s=[dataset.images[i] for i in index]

        with dt[0]:
            # im = torch.from_numpy(im).to(model.device)
            im = im.to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                # pred = model(im, augment=augment, visualize=visualize)
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # csv_path = save_dir / "predictions.csv"

        # Process predictions
        idx = 0
        for i, det in enumerate(pred):  # per image

            # save every number center coordinate
            number_center_per_image=[]
            im0s=img0s[i]
            seen += 1
            if torch.is_tensor(im0s):
                im0s = im0s.cpu().numpy()
            im0, frame = im0s.copy(), getattr(dataset, "frame", 0)
            p = 's.jpg'
            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            # s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                max_index=torch.argmax(det[:,4])
                det=det[max_index]
                det_np=det.cpu().numpy()
                number_det.append(det_np)
                if det.shape[0] >1 :
                    det=det.unsqueeze(0)
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    label = (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # if save_crop:
                    #     save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
            else:

                # 如果没检测到，也需要放默认值，否则顺序乱

                zeros_array=np.zeros(4,dtype=np.float32)
                number_det.append(zeros_array)


            # Stream results
            im0 = annotator.result()
            view_img=True
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # cv2.imshow(str(p), im0)
                # cv2.waitKey(0)  # 1 millisecond
                cv2.imwrite(f'tmp_pic/{idx}.png', im0)
                idx+=1


    print('推理完成',time.time()-t_start)
    return number_det
