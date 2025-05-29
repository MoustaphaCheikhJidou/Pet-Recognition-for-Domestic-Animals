
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Charger le modèle
model = load_model("cnn_animals_model.h5")

# Classes
classes = ["dog", "cat", "rabbit"]

# Fonction de prédiction
def predict(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0  # Normalisation
    image = image.reshape(1, 128, 128, 3)
    predictions = model.predict(image)[0]
    results = {classes[i]: float(predictions[i]) for i in range(len(classes))}
    return results

# Interface Gradio
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Animal Classifier",
    description="Upload an image of a dog, cat or rabbit and let the model classify it.",
)

interface.launch()
