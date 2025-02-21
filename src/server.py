from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
import sys

# 添加项目根目录到系统路径
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 导入你的检测模块
from detect_barcode_bifpn_calling import run

app = FastAPI(
    title="Sigen Microinverter Barcode Detection API",
    description="API for barcode detection",
    version="1.0.0"
)

# 初始化模型（全局变量）
WEIGHTS_PATH = ROOT / "runs/train/exp95/weights/best.pt"
DEVICE = ""  # 自动选择设备


@app.get("/")
async def root():
    return {"message": "Welcome to YOLOv5 Detection API"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # 读取上传的图片
        image_data = await file.read()

        # 保存临时文件
        temp_dir = ROOT / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / file.filename

        with open(temp_path, "wb") as f:
            f.write(image_data)

        # 设置检测参数
        detect_args = {
            "weights": WEIGHTS_PATH,
            "source": str(temp_path),
            "data": ROOT / "data/qr-custom-data.yaml",
            "imgsz": (640, 640),
            "conf_thres": 0.25,
            "iou_thres": 0,
            "max_det": 1000,
            "device": DEVICE,
            "view_img": False,
            "save_txt": False,
            "save_conf": False,
            "save_crop": False,
            "nosave": False,
            "classes": None,
            "agnostic_nms": False,
            "augment": False,
            "visualize": False,
            "update": False,
            "project": ROOT / "runs/detect",
            "name": "exp",
            "exist_ok": False,
            "line_thickness": 3,
            "hide_labels": False,
            "hide_conf": False,
            "half": False,
            "dnn": False,
            "vid_stride": 1
        }

        # 运行检测
        results = run(**detect_args)

        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Detection completed successfully",
                "results": results if results else []
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

'''
测试10.1.49.4 的
'''
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="10.1.49.4", port=5000)
