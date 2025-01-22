# QR Code Batch Detection with YOLOv5

This repository provides a solution for detecting QR codes in multiple images using the YOLOv5 object detection model.

## Features

- **Batch Processing**: Efficiently process multiple images to detect QR codes.
- **High Accuracy**: Utilizes YOLOv5's state-of-the-art object detection capabilities.
- **Customization**: Easily adaptable to different datasets and detection requirements.

## Prerequisites

- **Python**: Ensure Python 3.8 or higher is installed.
- **Dependencies**: Install required packages using `pip install -r requirements.txt`.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone git@github.com:RichardPan0000/qr-detect-yolov5.git
   ```

2. **Install Dependencies**:
  
   ```bash
   cd src
   pip install -r requirements.txt
   ```

## Dataset Preparation

Prepare a dataset with images containing QR codes. Annotate the images in YOLO format, specifying the bounding boxes around the QR codes.



## Results

After running inference, the results will include:

- **Annotated Images**: Images with bounding boxes around detected QR codes.
- **Detection Logs**: Details of detected QR codes, including confidence scores and coordinates.

## Customization

- **Model Architecture**: Modify the YOLOv5 configuration files to change the model size or architecture.
- **Hyperparameters**: Adjust training hyperparameters in the `hyp.yaml` file for better performance.

## References

- [YOLOv5 Official Repository](https://github.com/ultralytics/yolov5)
- [YOLOv5 Documentation](https://docs.ultralytics.com/)
## License

This project is licensed under the AGPL-3.0 License.

---

*Note*: This README provides a general framework. Customize it based on your project's specific details and requirements. 

## Citation

If you use YOLOv5 in your work, please cite it as follows:

```bibtex
@software{yolov5,
  title = {YOLOv5 by Ultralytics},
  author = {Glenn Jocher},
  year = {2020},
  version = {7.0},
  license = {AGPL-3.0},
  url = {https://github.com/ultralytics/yolov5},
  doi = {10.5281/zenodo.3908559},
  orcid = {0000-0001-5950-6979}
}

