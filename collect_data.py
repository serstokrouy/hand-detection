"""
Collect hand gesture images for dataset creation.
"""

import logging
import os
import urllib.request
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class DataCollector:

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    def __init__(self):
        self.detector = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.auto_save = False

        self._setup_detector()

    def _setup_detector(self):

        try:

            if not os.path.exists(config.HAND_MODEL_PATH):

                logger.info("Downloading MediaPipe model...")

                urllib.request.urlretrieve(
                    config.HAND_MODEL_URL,
                    config.HAND_MODEL_PATH
                )

                logger.info("Download complete")

            base_options = python.BaseOptions(
                model_asset_path=config.HAND_MODEL_PATH
            )

            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=config.HAND_NUMBER,
                min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
            )

            self.detector = vision.HandLandmarker.create_from_options(
                options
            )

            logger.info("Hand detector initialized")

        except Exception as e:

            logger.error(
                f"Detector setup failed: {e}"
            )

            raise

    def _setup_camera(self):

        self.cap = cv2.VideoCapture(
            config.CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            config.FRAME_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            config.FRAME_HEIGHT
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {config.CAMERA_INDEX}"
            )

        logger.info(
            "Camera opened successfully"
        )

    @classmethod
    def draw_landmarks(
        cls,
        frame,
        landmarks
    ):

        h, w = frame.shape[:2]

        for start, end in cls.HAND_CONNECTIONS:

            x1 = int(landmarks[start].x * w)
            y1 = int(landmarks[start].y * h)

            x2 = int(landmarks[end].x * w)
            y2 = int(landmarks[end].y * h)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        for lm in landmarks:

            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )

    @staticmethod
    def get_hand_bbox(
        frame,
        landmarks
    ) -> Tuple[int, int, int, int]:

        h, w = frame.shape[:2]

        coords = np.array([
            [
                int(lm.x * w),
                int(lm.y * h)
            ]
            for lm in landmarks
        ])

        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        padding = max(
            int(max(
                x_max - x_min,
                y_max - y_min
            ) * 0.10),
            10
        )

        x1 = max(x_min - padding, 0)
        y1 = max(y_min - padding, 0)

        x2 = min(x_max + padding, w)
        y2 = min(y_max + padding, h)

        return x1, y1, x2, y2

    @staticmethod
    def crop_hand_region(
        frame,
        bbox
    ):

        x1, y1, x2, y2 = bbox

        return frame[
            y1:y2,
            x1:x2
        ]

    @staticmethod
    def ensure_save_dir(label):

        path = os.path.join(
            config.DATASET_PATH,
            label
        )

        os.makedirs(
            path,
            exist_ok=True
        )

        return path

    def save_image(
        self,
        roi,
        save_dir,
        count
    ):

        roi = cv2.resize(
            roi,
            config.SAVE_SIZE
        )

        roi = cv2.GaussianBlur(
            roi,
            (3, 3),
            0
        )

        filename = os.path.join(
            save_dir,
            f"{count}.jpg"
        )

        cv2.imwrite(
            filename,
            roi
        )

        logger.info(
            f"Saved {filename}"
        )

        return count + 1

    def run(self, label):

        if (
            not label.isdigit()
            or int(label) not in range(
                config.NUM_CLASSES
            )
        ):

            logger.error(
                f"Label must be 0-{config.NUM_CLASSES - 1}"
            )

            return

        self._setup_camera()

        save_dir = self.ensure_save_dir(
            label
        )

        count = len([
            f for f in os.listdir(
                save_dir
            )
            if f.endswith(".jpg")
        ])

        target = config.TARGET_IMAGES

        logger.info("S = Save")
        logger.info("A = Auto Save ON/OFF")
        logger.info("Q = Quit")

        try:

            while True:

                ret, frame = self.cap.read()

                if not ret:
                    break

                frame = cv2.flip(
                    frame,
                    1
                )

                original = frame.copy()

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                results = self.detector.detect(
                    mp_image
                )

                hand_found = False
                bbox = None

                if results.hand_landmarks:

                    all_landmarks = []

                    for landmarks in results.hand_landmarks:

                        self.draw_landmarks(
                            frame,
                            landmarks
                        )

                        all_landmarks.extend(
                            landmarks
                        )

                    bbox = self.get_hand_bbox(
                        frame,
                        all_landmarks
                    )

                    x1, y1, x2, y2 = bbox

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )

                    hand_found = True

                cv2.putText(
                    frame,
                    f"Label: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Saved: {count}/{target}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Auto Save: {'ON' if self.auto_save else 'OFF'}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.imshow(
                    "Collect Data",
                    frame
                )

                if (
                    self.auto_save
                    and hand_found
                    and bbox is not None
                ):

                    roi = self.crop_hand_region(
                        original,
                        bbox
                    )

                    if roi.size > 0:

                        count = self.save_image(
                            roi,
                            save_dir,
                            count
                        )

                        cv2.waitKey(200)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):

                    if hand_found and bbox:

                        roi = self.crop_hand_region(
                            original,
                            bbox
                        )

                        if roi.size > 0:

                            count = self.save_image(
                                roi,
                                save_dir,
                                count
                            )

                elif key == ord('a'):

                    self.auto_save = (
                        not self.auto_save
                    )

                elif key == ord('q'):

                    break

                if count >= target:

                    logger.info(
                        "Target reached!"
                    )

                    break

        finally:

            self.cap.release()
            cv2.destroyAllWindows()


def main():

    label = input(
        f"Enter label (0-{config.NUM_CLASSES-1}): "
    ).strip()

    collector = DataCollector()

    collector.run(label)


if __name__ == "__main__":
    main()