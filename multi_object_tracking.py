"""
Perform multi object tracking on a mp4 video with a pytorch neural network model (.pth)
Instead of only performing detections, we try to determine objects position over time.
"""

from argparse import ArgumentParser
from time import time
import numpy as np
import cv2
from tqdm import tqdm

from utils import draw_fps, draw_frame_id, COLORS
from object_detector import ObjectDetector
from kalman_filter import KalmanFilter


class Object:
    """
    Object class with its current state (position, yaw, width/height),
    age since creation, trajectory, Kalman Filter.
    We also have an unassociated counter to be robust to misdetections
    """

    def __init__(self, id, det):
        self.id = id
        self.traj = [[int(det["x"]), int(det["y"])]]
        self.last_det = det
        self.unassociated_counter = 0
        self.kf = KalmanFilter(
            nx=8,
            nz=5,
            first_measurement=np.asarray(
                [det["x"], det["y"], det["angle"], det["w"], det["h"]]
            ).reshape(-1),
        )

    def dist(self, det):
        """Compute distance between object center and detection"""
        return np.linalg.norm(np.array([self.x() - det["x"], self.y() - det["y"]]))

    def update(self, det):
        """Update object after a match with a detection"""

        meas = np.asarray([det["x"], det["y"], det["angle"], det["w"], det["h"]])

        # Make sure we had a consistent sine and cosine values from heatmaps
        # Normally => cos(yaw)**2 + sin(yaw)**2 = 1
        if det["sin_cos_norm"] < 0.8:
            print(f"Ignore angle for {self.id}")
            meas[2] = self.last_det["angle"]
            det["angle"] = self.last_det["angle"]

        self.kf.update(meas)
        self.last_det = det
        self.unassociated_counter = 0

    def get_age(self):
        """Return number of frames since creation"""
        return len(self.traj)

    def x(self):
        return self.kf.estimate[0, 0]

    def y(self):
        return self.kf.estimate[1, 0]

    def yaw(self):
        return self.kf.estimate[2, 0]

    def w(self):
        return self.kf.estimate[6, 0]

    def h(self):
        return self.kf.estimate[7, 0]

    def get_direction(self):
        vx = self.kf.estimate[3, 0]
        vy = self.kf.estimate[4, 0]
        # print(f"Object {self.id} : vx = {vx:.2f} vy={vy:.2f}")
        norm = np.linalg.norm(self.kf.estimate[3:5, 0])
        if norm < 10:
            return 0, 0
        vx /= norm
        vy /= norm
        return vx, vy


class Tracking:
    """
    Tracking class: responsible to create/kill objects,
    and to match current detections with previous Tracking state
    """

    def __init__(self):
        self.objects = []
        self.last_id = -1

    def add(self, det):
        obj = Object(self.last_id + 1, det)
        self.objects.append(obj)
        self.last_id += 1

    def kill(self, id):
        """Kill object with id"""
        next_objects = []
        for object in self.objects:
            if object.id != id:
                next_objects.append(object)

        self.objects = next_objects

    def print_match(self, detections):
        """Debug only"""
        print("Match :")
        matches = np.zeros((len(self.objects), len(detections)))
        for i, object in enumerate(self.objects):
            for j, detection in enumerate(detections):
                matches[i, j] = object.dist(detection)

        print(matches)

    def hungarian(self, detections):
        """Compute distances between detection bboxes and tracking objects"""
        distances = np.zeros((len(self.objects), len(detections)))
        for i, object in enumerate(self.objects):
            for j, detection in enumerate(detections):
                distances[i, j] = int(object.dist(detection))

        return distances

    def update(self, detections):
        """
        Update tracking with latest detections (from Pytorch model)

        """

        # First predict objects state with KF
        for object in self.objects:
            object.kf.predict()

        # Compute distances between detections and tracking objects
        IMPOSSIBLE = 1000
        distances = self.hungarian(detections)
        if len(distances) == 0:
            # No objects yet
            for detection in detections:
                print(f"Create new object with det {detection}")
                self.add(detection)
            return

        # Match objects and detections if their respective distance is < 30px
        match_objects = [False] * len(self.objects)
        match_detections = [False] * len(detections)

        if distances.shape[1] > 0:
            while True:
                assert distances.shape[0] > 0
                assert distances.shape[1] > 0
                match_idx = np.asarray(
                    np.unravel_index(np.argmin(distances), distances.shape)
                )
                i = match_idx[0]
                j = match_idx[1]
                if distances[i, j] > 30:
                    break

                self.objects[i].update(detections[j])
                distances[i, :] = IMPOSSIBLE
                distances[:, j] = IMPOSSIBLE
                match_objects[i] = True
                match_detections[j] = True

        # Create objects if detection didnt match any previous object
        for j, detection in enumerate(detections):
            if not match_detections[j]:
                print(f"Create new object with det {detection}")
                self.add(detection)

        for object in self.objects:
            object.traj.append([int(object.x()), int(object.y())])

        # Delete objects that have not matched with any detections for 20 scans
        objects = self.objects.copy()
        for i, match_object in enumerate(match_objects):
            if not match_object:
                # Kill object
                pos_x = objects[i].traj[-1][0]
                pos_y = objects[i].traj[-1][1]
                objects[i].unassociated_counter += 1
                # Delete objects that go out of bounds or that are very young
                if (
                    objects[i].get_age() < 3
                    or pos_x < 0
                    or pos_x >= 1280
                    or pos_y < 0
                    or pos_y >= 720
                ):
                    print(
                        f"Kill object {objects[i].id} with age {objects[i].get_age()} at position "
                        f"{objects[i].traj[-1]} after {objects[i].unassociated_counter} unassociated frames"
                    )
                    self.kill(objects[i].id)

                elif objects[i].unassociated_counter >= 20:
                    print(
                        f"!!!!!! Kill object {objects[i].id} with age {objects[i].get_age()} at position "
                        f"{objects[i].traj[-1]} after {objects[i].unassociated_counter} unassociated frames"
                    )
                    # assert False
                    self.kill(objects[i].id)

    def __str__(self) -> str:
        if len(self.objects) == 0:
            return "Tracking is empty"

        str = "Tracking contains:\n"
        str += "------------------------------------------------------\n"
        str += "   Id   |   X   |   Y   |   W   |   H   |   Angle   \n"
        str += "------------------------------------------------------\n"
        for object in self.objects:
            str += f"{object.id:5}   | {int(object.x()):4}  | {int(object.y()):4}  |"
            str += f" {int(object.w()):4}  | {int(object.h()):4}  |  {int(object.yaw()/np.pi*180):4}  \n"
        return str

    def display(self, frame):
        """
        Draw each objects with oriented bounding boxes and trajectory
        """

        for object in self.objects:
            # Select color
            color = COLORS[object.id % len(COLORS)]

            # Uncomment to draw object ID numbers
            # cv2.putText(
            #    frame,
            #    f"{object.id}",
            #    (int(object.x()), int(object.y())),
            #    cv2.FONT_HERSHEY_SIMPLEX,
            #    fontScale=0.8,
            #    color=color,
            #    thickness=2,
            # )

            # Draw trajectory
            traj = np.array(object.traj[-args.traj :]).reshape(-1, 1, 2)
            frame = cv2.polylines(
                frame, [traj], isClosed=False, color=color, thickness=1
            )

            cos_angle = np.cos(object.yaw())
            sin_angle = np.sin(object.yaw())
            rot = np.array([[cos_angle, sin_angle], [-sin_angle, cos_angle]])

            box_w, box_h = object.w(), object.h()
            corners = (
                np.array(
                    [
                        [box_w, box_w, -box_w, -box_w],
                        [box_h, -box_h, -box_h, box_h],
                    ]
                )
                / 2
            )
            x, y = object.x(), object.y()
            # Apply angle rotation
            corners = np.dot(rot, corners) + np.array([x, y]).reshape(2, 1)
            corners = corners.astype(int)

            br = tuple(corners[:, 0])
            tr = tuple(corners[:, 1])
            tl = tuple(corners[:, 2])
            bl = tuple(corners[:, 3])

            # Draw bounding box with a different color for the front edge
            thickness = 2
            cv2.line(frame, br, tr, (0, 220, 255), thickness)
            cv2.line(frame, br, bl, color=color, thickness=thickness)
            cv2.line(frame, tl, bl, color=color, thickness=thickness)
            cv2.line(frame, tl, tr, color=color, thickness=thickness)

            # Uncomment to draw speed direction
            # vx, vy = object.get_direction()
            # cv2.line(
            #    frame,
            #    (int(x), int(y)),
            #    (int(x + vx * 30), int(y + vy * 30)),
            #    (0, 220, 255),
            #    thickness,
            # )


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-object tracking")
    parser.add_argument("video", type=str, help="Video")
    parser.add_argument("model", type=str, help="Pytorch model for bbox cars detection")
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Threshold to keep an object"
    )
    parser.add_argument(
        "-f", type=int, default=np.inf, help="Pause viz at specified frame"
    )
    parser.add_argument(
        "--traj", type=int, default=15, help="Number of points to draw for trajectory"
    )
    args = parser.parse_args()

    object_detector = ObjectDetector(args.model, args.conf)

    cap = cv2.VideoCapture(args.video)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error opening video stream or file")

    prev_time = time()

    tracking = Tracking()
    PLAY = False

    for idx in tqdm(range(n_frames)):
        if not cap.isOpened():
            break

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        print(f"\n#################### Frame {idx} #####################")

        bboxs = object_detector.detect(frame)

        tracking.update(bboxs)
        # print(tracking)

        # frame = showbox(frame, bboxs)
        tracking.display(frame)

        fps = 1 / (time() - prev_time)
        prev_time = time()
        draw_fps(frame, fps)
        draw_frame_id(frame, idx)

        cv2.imshow("Detection", frame)
        k = cv2.waitKey(10 if ((idx < args.f < np.inf) or PLAY) else 0)

        if k == 27:
            break

        if k == 32:  # SPACE
            PLAY = not PLAY

        # cv2.imwrite(f"video/image_{idx:04d}.png", frame)
