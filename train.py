import os
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from tensorflow.keras.utils import (
    to_categorical
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class HandNumberTrainer:

    def __init__(self):
        self.X = []
        self.y = []

    def load_dataset(self):

        logger.info("Loading dataset...")

        for label in range(config.NUM_CLASSES):

            folder = os.path.join(
                config.DATASET_PATH,
                str(label)
            )

            if not os.path.exists(folder):
                logger.warning(f"Folder not found: {folder}")
                continue

            files = [
                f for f in os.listdir(folder)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            logger.info(
                f"Class {label}: {len(files)} images"
            )

            for file in files:

                path = os.path.join(folder, file)

                try:

                    img = cv2.imread(path)

                    if img is None:
                        continue

                    img = cv2.cvtColor(
                        img,
                        cv2.COLOR_BGR2RGB
                    )

                    img = cv2.resize(
                        img,
                        (
                            config.INPUT_SIZE,
                            config.INPUT_SIZE
                        )
                    )

                    img = preprocess_input(img)

                    self.X.append(img)
                    self.y.append(label)

                except Exception as e:

                    logger.warning(
                        f"Skipping {path}: {e}"
                    )

        self.X = np.array(
            self.X,
            dtype=np.float32
        )

        self.y = np.array(
            self.y,
            dtype=np.int32
        )

        logger.info(
            f"Total samples: {len(self.X)}"
        )

        logger.info(
            f"Unique Labels: {np.unique(self.y)}"
        )

        if len(self.X) == 0:
            raise ValueError(
                "Dataset is empty!"
            )

    def prepare_data(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )

        print("\nTrain Distribution")

        unique, counts = np.unique(
            y_train,
            return_counts=True
        )

        print(
            dict(zip(unique, counts))
        )

        y_train = to_categorical(
            y_train,
            config.NUM_CLASSES
        )

        y_val = to_categorical(
            y_val,
            config.NUM_CLASSES
        )

        y_test = to_categorical(
            y_test,
            config.NUM_CLASSES
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )

    def build_model(self):

        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(
                config.INPUT_SIZE,
                config.INPUT_SIZE,
                3
            )
        )

        base_model.trainable = True

        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model = Sequential([

            Input(shape=(
                config.INPUT_SIZE,
                config.INPUT_SIZE,
                3
            )),

            base_model,

            GlobalAveragePooling2D(),

            Dense(
                128,
                activation='relu'
            ),

            Dropout(0.5),

            Dense(
                config.NUM_CLASSES,
                activation='softmax'
            )
        ])

        model.compile(

            optimizer=Adam(
                learning_rate=1e-5
            ),

            loss='categorical_crossentropy',

            metrics=['accuracy']
        )

        model.summary()

        return model

    def train(self):

        self.load_dataset()

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        ) = self.prepare_data()

        model = self.build_model()

        os.makedirs(
            "models",
            exist_ok=True
        )

        callbacks = [

            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True
            ),

            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),

            ModelCheckpoint(
                filepath=config.MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        datagen = ImageDataGenerator(

            rotation_range=15,

            zoom_range=0.1,

            width_shift_range=0.1,

            height_shift_range=0.1,

            brightness_range=[0.9, 1.1],

            horizontal_flip=False,

            fill_mode='nearest'
        )

        history = model.fit(

            datagen.flow(
                X_train,
                y_train,
                batch_size=config.BATCH_SIZE
            ),

            validation_data=(
                X_val,
                y_val
            ),

            epochs=config.EPOCHS,

            callbacks=callbacks,

            verbose=1
        )

        loss, accuracy = model.evaluate(
            X_test,
            y_test,
            verbose=1
        )

        logger.info(
            f"Test Accuracy: {accuracy:.4f}"
        )

        self.plot(history)

    def plot(self, history):

        plt.figure(figsize=(10, 5))

        plt.plot(
            history.history['accuracy'],
            label='Train Accuracy'
        )

        plt.plot(
            history.history['val_accuracy'],
            label='Validation Accuracy'
        )

        plt.title(
            "Model Accuracy"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))

        plt.plot(
            history.history['loss'],
            label='Train Loss'
        )

        plt.plot(
            history.history['val_loss'],
            label='Validation Loss'
        )

        plt.title(
            "Model Loss"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


def main():

    trainer = HandNumberTrainer()

    trainer.train()


if __name__ == "__main__":
    main()