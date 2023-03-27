import cv2


def draw_fps(frame, fps):
    """Draw fps to demonstrate performance"""
    cv2.putText(
        frame,
        f"{int(fps)} fps",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )


def draw_frame_id(frame, frame_id):
    """Draw fps to demonstrate performance"""
    cv2.putText(
        frame,
        f"Frame {frame_id}",
        (frame.shape[1] - 150, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )


COLORS = [
    (0, 0, 128),
    (0, 0, 255),
    (0, 128, 0),
    (0, 128, 128),
    (0, 128, 255),
    (0, 255, 0),
    (0, 255, 128),
    (0, 255, 255),
    (128, 0, 0),
    (128, 0, 128),
    (128, 0, 255),
    (128, 128, 0),
    (128, 128, 128),
    (128, 128, 255),
    (128, 255, 0),
    (128, 255, 128),
    (128, 255, 255),
    (255, 0, 0),
    (255, 0, 128),
    (255, 0, 255),
    (255, 128, 0),
    (255, 128, 128),
    (255, 128, 255),
    (255, 255, 0),
    (255, 255, 128),
    (255, 255, 255),
]
