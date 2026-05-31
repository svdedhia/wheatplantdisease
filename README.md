# Wheat Disease Classification Model

A MobileNetV2-based CNN model for classifying wheat plant diseases, served through a Flask API with a browser-based frontend.

## Description

This is the final project for CS 250: Foundation of AI and Data Science at Eastern University. The model accepts an uploaded wheat leaf image and predicts which disease (or healthy state) it belongs to.

## Disease Classes

| Class | Conditions |
|---|---|
| Blight Spot | Leaf Blight, Septoria, Tan Spot |
| Fusarium Blast | Fusarium Head Blight, Blast |
| Healthy | — |
| Mildew | — |
| Pests | Aphid, Mite, Stem Fly |
| Rust | — |
| Smut / Rot | Common Root Rot, Smut |

## Tech Stack

- **Model:** TensorFlow / Keras — MobileNetV2 (transfer learning), input size 160×160
- **Backend:** Flask + Flask-CORS
- **Frontend:** HTML, CSS, vanilla JavaScript (`index.html`, `style.css`)
- **Image processing:** Pillow, NumPy

## Requirements

```
Python 3.9.6
tensorflow 2.20.0
Flask 3.1.2
flask-cors 6.0.1
NumPy 2.0.2
Pillow 11.3.0
```

Install dependencies:

```bash
pip install -r requirement.txt
```

## Running the App

```bash
python app.py
```

The API starts on `http://localhost:5000`. Open `index.html` in a browser to use the frontend.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/WheatDisease` | Health check — returns model load status |
| POST | `/predict` | Upload an image (`multipart/form-data`, field: `image`) and receive a prediction |

#### Example `/predict` response

```json
{
  "class_index": 2,
  "label": "Healthy",
  "confidence": 0.9732,
  "all_predictions": {
    "Blight_Spot (Leaf Blight, Septoria, Tan Spot)": 0.005,
    "Fusarium_Blast (Fusarium Head Blight, Blast)": 0.003,
    "Healthy": 0.9732,
    "Mildew": 0.007,
    "Pests (Aphid, Mite, Stem Fly)": 0.004,
    "Rust": 0.006,
    "Smut_Rot (Common Root Rot, Smut)": 0.002
  }
}
```

## Model

The trained model is stored as `wheat_model_v2.keras`. Images are preprocessed using MobileNetV2's built-in normalization (scaled to −1 to 1) before inference. Logits are converted to probabilities via softmax.

## Presentation

A project presentation PDF is included: `CS250 - WheatDiseaseClassificationModel.pdf`.
