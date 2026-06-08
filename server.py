from fastapi import FastAPI
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

app = FastAPI(title="MNIST Cloud API")

MODEL_PATH = "mnist_model.h5"


@app.get("/")
def home():
    return {
        "message": "MNIST Cloud API Running"
    }


@app.get("/train")
def train_model():

    (images, labels), _ = mnist.load_data()

    images = images.astype("float32") / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42
    )

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=5,
        verbose=1
    )

    model.save(MODEL_PATH)

    predictions = np.argmax(
        model.predict(X_test),
        axis=1
    )

    acc = accuracy_score(
        y_test,
        predictions
    )

    return {
        "status": "Training Completed",
        "accuracy": float(acc)
    }


@app.get("/evaluate")
def evaluate_model():

    if not os.path.exists(MODEL_PATH):
        return {
            "error": "Train model first"
        }

    (images, labels), _ = mnist.load_data()

    images = images.astype("float32") / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42
    )

    model = load_model(MODEL_PATH)

    predictions = np.argmax(
        model.predict(X_test),
        axis=1
    )

    acc = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    return {
        "accuracy": float(acc),
        "report": report
    }


@app.get("/predict/{index}")
def predict_digit(index: int):

    if not os.path.exists(MODEL_PATH):
        return {
            "error": "Train model first"
        }

    (images, labels), _ = mnist.load_data()

    images = images.astype("float32") / 255.0

    model = load_model(MODEL_PATH)

    img = images[index]

    prediction = np.argmax(
        model.predict(
            np.expand_dims(img, axis=0)
        ),
        axis=1
    )[0]

    return {
        "sample_index": index,
        "actual_label": int(labels[index]),
        "predicted_label": int(prediction)
    }
