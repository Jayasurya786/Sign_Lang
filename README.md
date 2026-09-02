# Sign Language Recognition

A Flask + MediaPipe + BiLSTM sign-language recognition app for the American Sign Language alphabet. The app uses the online ASL alphabet dataset from Hugging Face and recognizes letters from live webcam input.

## Features

- Online dataset support for ASL A-Z
- MediaPipe hand landmark extraction
- Wrist-centered and scale-normalized landmark features
- BiLSTM sequence training for classification
- Real-time webcam prediction
- Majority vote for stable letter recognition
- Model quality evaluation and dataset health reporting

## Project structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── scripts/
│   └── download_hf_asl.py
├── src/
│   ├── data/
│   ├── models/
│   ├── processing/
│   └── utils/
├── templates/
├── tests/
├── data/
│   ├── online_asl/
│   └── processed/
└── .gitignore
```

## Tech stack

- Python 3.11+
- Flask
- MediaPipe
- TensorFlow / Keras
- Pandas / NumPy
- OpenCV

## Setup

1. Create a virtual environment

```bash
python -m venv .venv
```

2. Activate it

Windows:

```bash
.\.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Download the online ASL dataset

The project is configured to use the online A-Z dataset.

```bash
python scripts/download_hf_asl.py
```

This downloads images into:

```text
data/online_asl/
```

## Run the app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Train the model

The app can train the model from the online dataset automatically from the UI. If you want to train from the terminal, run the Flask app and click the Train model button.

## Model details

The model uses:

- MediaPipe hand landmarks
- Hand-relative coordinate normalization
- Sequence generation for BiLSTM input
- A softmax output layer for the 26 ASL letters

## Testing

```bash
pytest -q
```

## Notes

- Large generated files such as datasets and trained Keras models are ignored in git.
- Avoid committing raw downloaded datasets unless you want to store them in Git LFS or a separate storage solution.
- The project is designed for local development and experimentation.

## Git push commands

Initialize and push to GitHub:

```bash
git init
git add .
git commit -m "Initial commit: ASL recognition app"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

If the repo already exists:

```bash
git add .
git commit -m "Update ASL recognition app"
git push origin main
```

## Recommended next improvements

- Add a stronger CNN/Transformer image model for better accuracy
- Add confidence thresholding for unknown gestures
- Add dataset balancing and augmentation
- Export the model to TensorFlow Lite for mobile deployment
# Sign_Lang
