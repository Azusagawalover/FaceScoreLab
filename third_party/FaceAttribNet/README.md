# Face Attribute Detection

[![Downloads](https://img.shields.io/github/downloads/yakhyo/face-attribute/total.svg?label=Downloads)](https://github.com/yakhyo/face-attribute/releases)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-green.svg)](LICENSE)

> [!TIP]
> The models and functionality in this repository are **integrated into [UniFace](https://github.com/yakhyo/uniface)** — an all-in-one face analysis toolkit.<br>
> [![PyPI Version](https://img.shields.io/pypi/v/uniface.svg)](https://pypi.org/project/uniface/) [![GitHub Stars](https://img.shields.io/github/stars/yakhyo/uniface)](https://github.com/yakhyo/uniface/stargazers)

PyTorch inference, ONNX export, and ONNX Runtime inference for [FaceAttribNet](https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/models/face_attrib_net) — a lightweight model that predicts five face attributes at once from a 128×128 face crop. Uses [UniFace](https://github.com/yakhyo/uniface) for face detection.

![FaceAttribNet demo](assets/banner.png)

## Models

| Model         | Params | Input   | PyTorch (.pt)                                                                                         | ONNX                                                                                            |
| ------------- | ------ | ------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| FaceAttribNet | 10.84M | 128x128 | [Link](https://github.com/yakhyo/face-attribute/releases/download/weights/detection_only_05302025.pt) | [Link](https://github.com/yakhyo/face-attribute/releases/download/weights/face_attrib_net.onnx) |

The `.pt` is an unmodified mirror of the original checkpoint. The `.onnx` is re-exported from it at opset 17 with a dynamic batch dimension, as a single self-contained file (loadable on `onnxruntime>=1.16`), numerically identical to the original export.

## Attributes

The model outputs five probabilities in `[0, 1]`: `left_eye_open`, `right_eye_open`, `eyeglasses`, `mask`, `sunglasses` — in that order.

> [!IMPORTANT]
> These are **five independent binary heads**, not one 5-way softmax: values do not sum to 1 and several can be high at once. Threshold each attribute separately; never `argmax`.

## Installation

```bash
pip install -r requirements.txt
```

Download the weights from the [release](https://github.com/yakhyo/face-attribute/releases/tag/weights) (the `.onnx` alone is enough for inference):

```bash
mkdir -p weights
curl -L -o weights/face_attrib_net.onnx https://github.com/yakhyo/face-attribute/releases/download/weights/face_attrib_net.onnx
curl -L -o weights/detection_only_05302025.pt https://github.com/yakhyo/face-attribute/releases/download/weights/detection_only_05302025.pt
```

## Demo

Attributes above the detection threshold (0.5), predicted per face:

|                        Sunglasses                        |                                   Tinted sunglasses                                   |                  Face mask                   |                        Eyeglasses                        |                      Eyes closed                       |
| :------------------------------------------------------: | :-----------------------------------------------------------------------------------: | :------------------------------------------: | :------------------------------------------------------: | :----------------------------------------------------: |
|   ![](assets/results/sunglasses.jpg)<br>`sunglasses` 1.000   | ![](assets/results/sunglasses_tinted.jpg)<br>`sunglasses` 0.993, eyes still detected open |   ![](assets/results/mask.jpg)<br>`mask` 1.000   |   ![](assets/results/eyeglasses.jpg)<br>`eyeglasses` 0.999   | ![](assets/results/closed_eyes.jpg)<br>eye openness ≤ 0.16 |
| ![](assets/results/sunglasses_man.jpg)<br>`sunglasses` 0.999 |                                                                                       | ![](assets/results/mask_man.jpg)<br>`mask` 1.000 | ![](assets/results/eyeglasses_man.jpg)<br>`eyeglasses` 0.903 |                                                        |

| ![](assets/results/group.jpg)<br>Multiple faces: only the left person triggers `eyeglasses` (0.893) |
| :---------------------------------------------------------------------------------------------: |

```bash
python onnx_inference.py assets/test_images/*.jpg --save-dir assets/results
```

Test photos are free-license images from [Pexels](https://www.pexels.com).

## Inference

The model takes a 128×128 face crop. Both scripts take full images and handle detection and cropping for you.

### PyTorch

```bash
python main.py assets/test_images/group.jpg
```

### ONNX

```bash
python onnx_inference.py assets/test_images/group.jpg
```

Sources can be image paths or webcam indices, freely mixed. A numeric source opens that camera with a live annotated preview:

```bash
python onnx_inference.py 0                        # default webcam; q/ESC quits
python onnx_inference.py 0 --save-dir snapshots   # press s to save an annotated frame
```

```python
import cv2
from models import FaceAttribute
from uniface.detection import SCRFD

detector = SCRFD()
attrib = FaceAttribute('weights/face_attrib_net.onnx')

image = cv2.imread('image.jpg')
for face in detector.detect(image):
    result = attrib.predict(image, face.bbox)
    print(result.as_dict())            # {'left_eye_open': 0.99, ...}
    print(result.labels(threshold=0.5))  # e.g. ['left_eye_open', 'right_eye_open', 'eyeglasses']
```

### Preprocessing

Aspect-preserving resize to 128×128 (centered letterbox), RGB, values in `[0, 1]`. Mean/std normalization is **baked into the model graph** — do not normalize again.

## ONNX Export

```bash
python onnx_export.py -w weights/detection_only_05302025.pt -o weights/face_attrib_net.onnx
```

The export is verified against the PyTorch reference on the spot. `--dynamic` enables a dynamic batch dimension.

## Reference

- [AI Hub Models — face_attrib_net](https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/models/face_attrib_net) — Original model, architecture, and weights
- [UniFace](https://github.com/yakhyo/uniface) — Face detection

## License

[BSD-3-Clause](./LICENSE)
