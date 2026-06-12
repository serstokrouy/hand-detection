

import logging
import os
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class HandNumberDetector:

    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]

    def __init__(self):

        self.detector = None
        self.model = None
        self.cap = None

        self.predictions = deque(
            maxlen=config.SMOOTHING_WINDOW
        )

        self._load_model()
        self._setup_detector()

    def _load_model(self):

        if not os.path.exists(
            config.MODEL_PATH
        ):

            raise FileNotFoundError(
                f"Model not found: "
                f"{config.MODEL_PATH}"
            )

        self.model = tf.keras.models.load_model(
            config.MODEL_PATH
        )

        logger.info(
            "CNN model loaded"
        )

    def _setup_detector(self):

        if not os.path.exists(
            config.HAND_MODEL_PATH
        ):

            logger.info(
                "Downloading MediaPipe model..."
            )

            urllib.request.urlretrieve(
                config.HAND_MODEL_URL,
                config.HAND_MODEL_PATH
            )

        base_options = python.BaseOptions(
            model_asset_path=
            config.HAND_MODEL_PATH
        )

        options = vision.HandLandmarkerOptions(

            base_options=base_options,

            num_hands=
            config.HAND_NUMBER,

            min_hand_detection_confidence=
            config.MIN_HAND_DETECTION_CONFIDENCE,

            min_hand_presence_confidence=
            config.MIN_HAND_PRESENCE_CONFIDENCE,

            min_tracking_confidence=
            config.MIN_TRACKING_CONFIDENCE
        )

        self.detector = (
            vision.HandLandmarker
            .create_from_options(
                options
            )
        )

        logger.info(
            "Hand detector initialized"
        )

    @classmethod
    def draw_landmarks(
        cls,
        frame,
        landmarks
    ):

        h, w = frame.shape[:2]

        for start, end in cls.HAND_CONNECTIONS:

            x1 = int(
                landmarks[start].x * w
            )

            y1 = int(
                landmarks[start].y * h
            )

            x2 = int(
                landmarks[end].x * w
            )

            y2 = int(
                landmarks[end].y * h
            )

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
    def get_bbox(
        frame,
        landmarks
    ):

        h, w = frame.shape[:2]

        coords = np.array([

            [
                int(lm.x * w),
                int(lm.y * h)
            ]

            for lm in landmarks

        ])

        x_min, y_min = (
            coords.min(axis=0)
        )

        x_max, y_max = (
            coords.max(axis=0)
        )

        padding = max(

            int(
                max(
                    x_max - x_min,
                    y_max - y_min
                ) * 0.10
            ),

            10
        )

        x1 = max(
            x_min - padding,
            0
        )

        y1 = max(
            y_min - padding,
            0
        )

        x2 = min(
            x_max + padding,
            w
        )

        y2 = min(
            y_max + padding,
            h
        )

        return (
            x1,
            y1,
            x2,
            y2
        )

    def predict(
        self,
        roi
    ):

        roi = cv2.resize(

            roi,

            (
                config.INPUT_SIZE,
                config.INPUT_SIZE
            )
        )

        roi = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        )

        roi = preprocess_input(
            roi.astype(np.float32)
        )

        roi = np.expand_dims(
            roi,
            axis=0
        )

        prediction = (
            self.model.predict(
                roi,
                verbose=0
            )[0]
        )

        class_id = int(
            np.argmax(
                prediction
            )
        )

        confidence = float(
            np.max(
                prediction
            )
        )

        return (
            class_id,
            confidence
        )

    def run(self):

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
                "Cannot open camera"
            )

        while True:

            ret, frame = (
                self.cap.read()
            )

            if not ret:
                break

            frame = cv2.flip(
                frame,
                1
            )

            original = (
                frame.copy()
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(

                image_format=
                mp.ImageFormat.SRGB,

                data=rgb
            )

            results = (
                self.detector.detect(
                    mp_image
                )
            )

            if results.hand_landmarks:

                all_landmarks = []

                for landmarks in (
                    results.hand_landmarks
                ):

                    self.draw_landmarks(
                        frame,
                        landmarks
                    )

                    all_landmarks.extend(
                        landmarks
                    )

                bbox = self.get_bbox(
                    frame,
                    all_landmarks
                )

                x1, y1, x2, y2 = (
                    bbox
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                roi = original[
                    y1:y2,
                    x1:x2
                ]

                if roi.size > 0:

                    digit, conf = (
                        self.predict(
                            roi
                        )
                    )

                    self.predictions.append(
                        digit
                    )

                    smooth_digit = max(

                        set(
                            self.predictions
                        ),

                        key=
                        self.predictions.count
                    )

                    if (
                        conf >=
                        config.MIN_CONFIDENCE
                    ):

                        text = (
                            f"Number: "
                            f"{config.CLASS_NAMES[smooth_digit]}"
                            f" ({conf:.2f})"
                        )

                        color = (
                            0,
                            255,
                            0
                        )

                    else:

                        text = (
                            f"Low Confidence "
                            f"({conf:.2f})"
                        )

                        color = (
                            0,
                            0,
                            255
                        )

                    cv2.putText(

                        frame,

                        text,

                        (20, 50),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        1,

                        color,

                        2
                    )

            cv2.imshow(
                "Hand Number Detection",
                frame
            )

            if (
                cv2.waitKey(1)
                & 0xFF
                == ord('q')
            ):

                break

        self.cap.release()

        cv2.destroyAllWindows()


def main():

    detector = (
        HandNumberDetector()
    )

    detector.run()


if __name__ == "__main__":

    main()