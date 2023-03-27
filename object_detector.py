"""
Implement object detector class with helper functions
Load model, preprocess image, infer, post-process results
And returns detected bboxs.
"""
from argparse import ArgumentParser

import torch
from torch import nn
import torchvision

from torchvision import transforms
import cv2
import numpy as np


def showbox(img, boxes):
    """
    Boxes is a dictionnary of (x, y, w, h, angle, score) bounding boxes
    Draw them on the image img
    """

    for box in boxes:
        cos_angle = np.cos(box["angle"])
        sin_angle = np.sin(box["angle"])
        rot = np.array([[cos_angle, sin_angle], [-sin_angle, cos_angle]])

        corners = (
            np.array(
                [
                    [box["w"], box["w"], -box["w"], -box["w"]],
                    [box["h"], -box["h"], -box["h"], box["h"]],
                ]
            )
            / 2
        )

        # Apply angle rotation
        corners = np.dot(rot, corners) + np.array([box["x"], box["y"]]).reshape(2, 1)
        corners = corners.astype(int)

        br = tuple(corners[:, 0])
        tr = tuple(corners[:, 1])
        tl = tuple(corners[:, 2])
        bl = tuple(corners[:, 3])

        # Draw bounding box with a different color for the front edge
        thickness = 2
        cv2.line(img, br, tr, (0, 220, 0), thickness)
        cv2.line(img, br, bl, (220, 220, 0), thickness)
        cv2.line(img, tl, bl, (220, 220, 0), thickness)
        cv2.line(img, tl, tr, (220, 220, 0), thickness)

    return img


def select(hm, threshold):
    """
    Keep only local maxima (kind of NMS).
    We make sure to have no adjacent detection in the heatmap.
    """

    pred = hm > threshold
    pred_centers = np.argwhere(pred)

    for i, ci in enumerate(pred_centers):
        for j in range(i + 1, len(pred_centers)):
            cj = pred_centers[j]
            if np.linalg.norm(ci - cj) <= 2:
                score_i = hm[ci[0], ci[1]]
                score_j = hm[cj[0], cj[1]]
                if score_i > score_j:
                    hm[cj[0], cj[1]] = 0
                else:
                    hm[ci[0], ci[1]] = 0

    return hm


class ObjectDetector:
    """
    Initialize ObjectDetector by loading model
    Define confidence threshold
    Then preprocess, infer, post-process
    """

    INPUT_WIDTH = 1280
    INPUT_HEIGHT = 720
    MODEL_SCALE = 16

    def __init__(self, model_pth, conf_threhsold):
        self.conf = conf_threhsold
        assert torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device}")

        # Define and load model
        self.model = centernet()
        self.model.load_state_dict(torch.load(model_pth))
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def detect(self, frame):
        """
        Detect objects on frame and return bounding boxes.
        First resize image to expected shape.
        Pre-process for model input
        Extract heatmaps, and compute predicted bounding boxes
        """

        img = cv2.resize(frame, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
        input_tensor = self.preprocess(img)

        # Inference
        hm, offset, wh, cos_sin_hm = self.model(
            input_tensor.to(self.device).float().unsqueeze(0)
        )

        hm = torch.sigmoid(hm)

        hm = hm.cpu().detach().numpy().squeeze(0).squeeze(0)
        offset = offset.cpu().detach().numpy().squeeze(0)
        wh = wh.cpu().detach().numpy().squeeze(0)
        cos_sin_hm = cos_sin_hm.cpu().detach().numpy().squeeze(0)

        hm = select(hm, self.conf)

        boxes = self.pred2box(hm, offset, wh, cos_sin_hm)

        return boxes

    def pred2box(self, hm, offset, regr, cos_sin_hm):
        """
        Predict bounding boxes as (X, Y, W, H, angle, score) dictionnary
        """

        # get center
        pred = hm > self.conf
        pred_center = np.where(hm > self.conf)

        # get regressions
        pred_r = regr[:, pred].T
        pred_angles = cos_sin_hm[:, pred].T

        boxes = []
        scores = hm[pred]

        pred_center = np.asarray(pred_center).T
        for (center, wh, pred_angle, score) in zip(
            pred_center, pred_r, pred_angles, scores
        ):
            # print(b)
            offset_xy = offset[:, center[0], center[1]]
            angle = np.arctan2(pred_angle[1], pred_angle[0])
            bbox = {
                "x": (center[1] + offset_xy[0]) * self.MODEL_SCALE,
                "y": (center[0] + offset_xy[1]) * self.MODEL_SCALE,
                "w": wh[0] * self.MODEL_SCALE,
                "h": wh[1] * self.MODEL_SCALE,
                "angle": angle,
                "score": score,
                "sin_cos_norm": pred_angle[0] ** 2 + pred_angle[1] ** 2,
            }
            boxes.append(bbox)
        return boxes


class centernet(nn.Module):
    """
    Centernet simplified version
    Input = 1280x720 RGB image
    Output = 4 heatmaps
    * Main = [1, 45, 80]
    * Offset = [2, 45, 80]
    * Width/Height = [2, 45, 80]
    * Cos/sin angle = [2, 45, 80]
    """

    def __init__(self):
        super().__init__()

        # Resnet-18 as backbone.
        basemodel = torchvision.models.resnet18(weights=None)

        # Select only first layers up when you reach 160x90 dimensions with 256 channels
        self.base_model = nn.Sequential(*list(basemodel.children())[:-3])

        num_ch = 256
        head_conv = 64
        self.outc = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1, stride=1),
        )

        self.outo = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, stride=1),
        )

        self.outr = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, stride=1),
        )

        self.outa = nn.Sequential(
            nn.Conv2d(num_ch, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # [b, 3, 720, 1280]

        x = self.base_model(x)
        # [b, 128, 45, 80]

        assert not torch.isnan(x).any()

        outc = self.outc(x)
        # [b, 1, 45, 80]
        assert not torch.isnan(outc).any()

        outo = self.outo(x)
        # [b, 2, 45, 80]
        assert not torch.isnan(outo).any()

        outr = self.outr(x)
        outa = self.outa(x)

        return outc, outo, outr, outa


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-object detection")
    parser.add_argument("video", type=str, help="Video")
    parser.add_argument(
        "model", type=str, help="Pytorch model for oriented cars bbox detection"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Threshold to keep an object"
    )
    args = parser.parse_args()

    object_detector = ObjectDetector(args.model, args.conf)

    cap = cv2.VideoCapture(args.video)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        bboxs = object_detector.detect(frame)
        frame = showbox(frame, bboxs)

        cv2.imshow("Detection", frame)
        k = cv2.waitKey(1)

        if k == 27:
            break
