# 🖐️ Real-Time American Sign Language (ASL) Recognition System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg?logo=flask&logoColor=white)](https://palletsprojects.com/p/flask/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00C4B4.svg)](https://developers.google.com/mediapipe)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Accuracy](https://img.shields.io/badge/Holdout%20Accuracy-95.87%25-brightgreen.svg)]()
[![Validation Accuracy](https://img.shields.io/badge/Validation%20Accuracy-95.54%25-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-27%2F27%20Passing-success.svg)]()
[![TFLite](https://img.shields.io/badge/TFLite%20Model-1.28%20MB-blueviolet.svg)]()

A complete, high-accuracy, production-ready American Sign Language (ASL) alphabet and gesture recognition platform. Built with **MediaPipe Hands**, enriched with **100+ geometric biomechanical features**, classified by a **Deep Bidirectional LSTM (BiLSTM)** neural network, and delivered through an interactive **Flask web interface**, a **Dockerized container**, and a lightweight **TensorFlow Lite edge runtime**.

---

## 📑 Table of Contents

- [Key Highlights & Features](#-key-highlights--features)
- [Architecture & Processing Pipeline](#-architecture--processing-pipeline)
- [Biomechanical Feature Engineering](#-biomechanical-feature-engineering)
- [Data Augmentation Engine](#-data-augmentation-engine)
- [⚡ Dataset Generation: Complete Creation & Synthesis Pipeline](#-dataset-generation-complete-creation--synthesis-pipeline)
  - [Stage 1: Raw Image Generation (Webcam Burst Capture & In-Browser Studio)](#stage-1-raw-image-generation-webcam-burst-capture--in-browser-studio)
  - [Stage 2: Landmark Matrix Generation (MediaPipe 21 Hand Joints)](#stage-2-landmark-matrix-generation-mediapipe-21-hand-joints)
  - [Stage 3: High-Dimensional Feature Generation (100+ Spatial Descriptors)](#stage-3-high-dimensional-feature-generation-100-spatial-descriptors)
  - [Stage 4: Synthetic Dataset Generation via Augmentation (2x Scaling)](#stage-4-synthetic-dataset-generation-via-augmentation-2x-scaling)
  - [Stage 5: BiLSTM Temporal Sequence Tensor Generation](#stage-5-bilstm-temporal-sequence-tensor-generation)
- [Deep Learning Model Architecture](#-deep-learning-model-architecture)
- [Model Performance & Evaluation Benchmarks](#-model-performance--evaluation-benchmarks)
- [Supported ASL Alphabet Signs](#-supported-asl-alphabet-signs)
- [Quickstart Guide for New Users](#-quickstart-guide-for-new-users)
  - [Method 1: Docker Container (Recommended)](#method-1-docker-container-recommended)
  - [Method 2: Local Python Environment](#method-2-local-python-environment)
- [In-Depth Application User Guide](#-in-depth-application-user-guide)
  - [Real-Time Sign Recognition](#1-real-time-sign-recognition)
  - [Live Skeletal Visualizer](#2-live-skeletal-visualizer)
  - [Text-to-Speech (TTS) Voice Engine](#3-text-to-speech-tts-voice-engine)
  - [In-Air Gesture Controls & Shortcuts](#4-in-air-gesture-controls--shortcuts)
  - [In-Browser Custom Dataset Studio](#5-in-browser-custom-dataset-studio-dataset)
  - [Background Model Retraining](#6-background-model-retraining-from-ui)
- [TensorFlow Lite (TFLite) Edge & Mobile Deployment](#-tensorflow-lite-tflite-edge--mobile-deployment)
- [Cloud Deployment Guide (Render, Railway, GCP, AWS)](#-cloud-deployment-guide)
- [🗂️ Complete Dataset & Script Pipeline Guide](#-complete-dataset--script-pipeline-guide)
  - [1. Dataset Architecture & File Hierarchy](#1-dataset-architecture--file-hierarchy)
  - [2. Dataset Schema & Metadata Caching](#2-dataset-schema--metadata-caching)
  - [3. Script-by-Script Reference & CLI Arguments](#3-script-by-script-reference--cli-arguments)
    - [download_hf_asl.py](#script-1-download_hf_aslpy--hugging-face-dataset-downloader)
    - [check_dataset_balance.py](#script-2-check_dataset_balancepy--class-balance-auditor)
    - [build_combined_dataset.py](#script-3-build_combined_datasetpy--landmark-extractor--merger)
    - [train_model.py](#script-4-train_modelpy--bilstm-model-training-pipeline)
    - [evaluate_all_classes.py](#script-5-evaluate_all_classespy--holdout-image-evaluator)
    - [export_tflite.py](#script-6-export_tflitepy--tflite-exporter--parity-validator)
  - [4. End-to-End Walkthrough: Retraining on Custom Data](#4-end-to-end-walkthrough-retraining-on-custom-data)
- [Comprehensive REST API Specification](#-comprehensive-rest-api-specification)
- [Automated Test Suite](#-automated-test-suite)
- [Configuration & Environment Variables](#-configuration--environment-variables)
- [Project Directory Structure](#-project-directory-structure)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [License](#-license)

---

## 🌟 Key Highlights & Features

1. **High-Accuracy Classification (95.87% on Test Images)**:
   - Accurately classifies all 26 ASL alphabet signs ($A$–$Z$).
   - Trained on an expanded multi-source dataset of **8,749 real landmark samples** and **17,498 augmented sequences**.
2. **🦴 Real-Time Live Skeletal Visualizer**:
   - Transparent HTML5 canvas overlay directly synchronized with the client webcam.
   - Renders 21 anatomical joints and 21 interconnecting skeletal bones with real-time feedback.
3. **🔊 Web Speech API Text-to-Speech (TTS)**:
   - Voice synthesis automatically pronounces recognized words and sentences aloud.
   - Includes automatic word-boundary speech and a manual `🔊 Speak` trigger.
4. **🎮 In-Air Control Gestures & Interactive Buttons**:
   - In-air gestures: `SPACE` (flat open palm), `BACKSPACE` (fist retract), `CLEAR` (reset buffer).
   - Physical UI buttons for manual control.
5. **📸 In-Browser Dataset Studio (`/dataset`)**:
   - Capture new training samples directly through your web browser without touching files.
   - Sequential 20-frame burst recording with automatic class folder management.
6. **📱 Ultra-Lightweight TFLite Export (`1.28 MB`)**:
   - Exported with TensorFlow concrete function wrapping and custom ops.
   - Verified numerical parity ($< 10^{-7}$ maximum difference compared to Keras model).
7. **👐 Two-Handed & Multi-Hand Tracking**:
   - Landmark extraction engine upgraded with `max_num_hands=2` to support future multi-hand words and gestures.
8. **🐳 1-Command Docker & Docker Compose Support**:
   - Fully containerized Debian environment with OpenCV, MediaPipe, and TensorFlow pre-installed.
9. **🌐 Client-Side Streaming Viable Anywhere**:
   - Video frames are captured in the client browser and sent over HTTP/HTTPS, enabling full functionality on cloud hosting (Render, Railway, Cloud Run).

---

## 🏗️ Architecture & Processing Pipeline

```mermaid
flowchart TD
    A["Client Webcam Feed<br/>(getUserMedia / HTML5 Video)"] --> B["Client Canvas Snapshot<br/>(JPEG Data URL / Base64)"]
    B -->|"HTTP POST /api/predict"| C["Flask Server (app.py)"]
    C --> D["MediaPipe Hands Engine<br/>(21 3D Spatial Landmarks)"]
    D --> E["Feature Engineering (src/processing/features.py)<br/>(Wrist norm, 5 Extension Ratios, 10 Inter-tip Distances, 4 Thumb-knuckle Distances, 3D Palm Normal)"]
    E --> F["Temporal Sequence Buffer<br/>(Shape: 1 x 30 x 85)"]
    F --> G["Deep BiLSTM Model<br/>(sign_bilstm.keras / .tflite)"]
    G --> H["Softmax Probability Vector (26 Classes)"]
    H --> I["Debouncing & Majority Voting Filter"]
    I --> J["JSON Response Payload<br/>(letter, confidence, 21 landmarks, word, sentence)"]
    J --> K["Client Browser DOM Update"]
    K --> L["Canvas Overlay (Skeleton Joints & Bones)"]
    K --> M["Web Speech API (TTS Voice Synthesis)"]
```

---

## 🔬 Biomechanical Feature Engineering

Unlike naive computer vision models that pass raw pixel grids into large CNNs, this system extracts **invariant anatomical hand measurements** unaffected by background clutter, lighting variations, or hand size:

```text
Hand Landmark Skeleton (MediaPipe 21 Joints):
              [4] TIP
             /
         [3] IP
         /
     [2] MCP
     /
 [1] CMC             [8] TIP  [12] TIP  [16] TIP  [20] TIP
   \                   |        |         |         |
    \                [7] PIP  [11] PIP  [15] PIP  [19] PIP
     \                 |        |         |         |
      \              [6] DIP  [10] DIP  [14] DIP  [18] DIP
       \               |        |         |         |
        \            [5] MCP  [9] MCP   [13] MCP  [17] MCP
         \             \        \         /        /
          \-------------\--------\-------/--------/
                               [0] WRIST
```

### 1. Translation & Scale Invariance
- **Translation Normalization**: The wrist joint $[0]$ is centered at the coordinate origin:
  $$\mathbf{P}'_i = \mathbf{P}_i - \mathbf{P}_0 \quad \text{for } i = 0, \dots, 20$$
- **Scale Normalization**: Coordinates are normalized by the maximum distance across MCP knuckles, making hand size invariant whether the user is close to or far from the camera.

### 2. Five Finger Extension Ratios
Distinguishes curled vs. extended fingers:
$$R_{\text{finger}} = \frac{\|\mathbf{P}_{\text{TIP}} - \mathbf{P}_{\text{WRIST}}\|_2}{\|\mathbf{P}_{\text{MCP}} - \mathbf{P}_{\text{WRIST}}\|_2}$$
- $R \ge 1.8$: Fully extended finger (e.g., $B$, $D$, $V$, $W$).
- $R \le 1.1$: Curled finger curled into a fist (e.g., $A$, $E$, $S$).

### 3. Thumb-to-Knuckle Spatial Metrics
Disambiguates difficult "fist" signs that confuse standard vision models ($A, E, S, T, M, N$):
- $d(\text{Thumb\_Tip}, \text{Index\_MCP})$
- $d(\text{Thumb\_Tip}, \text{Middle\_MCP})$
- $d(\text{Thumb\_Tip}, \text{Ring\_MCP})$
- $d(\text{Thumb\_Tip}, \text{Pinky\_MCP})$

### 4. Inter-Fingertip Distances (10 Pairwise Combinations)
Measures the distance between all pairs of fingertips $\binom{5}{2} = 10$:
- Distinguishes **$U$** (index and middle fingers held tightly together) from **$V$** (index and middle fingers spread apart).
- Distinguishes **$W$** (three spread fingers) from **$B$** (four closed fingers).

### 5. 3D Palm Normal Orientation Vector
Calculated from the cross-product of hand plane vectors:
$$\mathbf{v}_1 = \mathbf{P}_{\text{Index\_MCP}} - \mathbf{P}_{\text{Wrist}}, \quad \mathbf{v}_2 = \mathbf{P}_{\text{Pinky\_MCP}} - \mathbf{P}_{\text{Wrist}}$$
$$\mathbf{n} = \frac{\mathbf{v}_1 \times \mathbf{v}_2}{\|\mathbf{v}_1 \times \mathbf{v}_2\|_2} = (n_x, n_y, n_z)$$
Identifies whether the palm faces directly forward, sideways (like $G$ and $H$), or downwards (like $P$ and $Q$).

---

## 🧬 Data Augmentation Engine

To prevent overfitting and ensure real-world robustness, [`src/processing/augmentation.py`](file:///d:/Sign%20Lang/src/processing/augmentation.py) applies vectorized transformations:

1. **Horizontal Flip / Mirroring (Left- & Right-Hand Signer Parity)**:
   $$x' = -x$$
   Doubles the dataset and enables equal accuracy for left-handed signers.
2. **2D In-Plane Rotation Jitter**:
   $$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} \quad \text{where } \theta \sim \mathcal{U}(-12^\circ, +12^\circ)$$
3. **Scale Perturbation**:
   $$\mathbf{P}' = s \cdot \mathbf{P} \quad \text{where } s \sim \mathcal{U}(0.92, 1.08)$$
4. **Gaussian Sensor Noise**:
   $$\mathbf{P}' = \mathbf{P} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, 0.005^2)$$

---

## ⚡ Dataset Generation: Complete Creation & Synthesis Pipeline

This project employs a multi-tiered **Dataset Generation Architecture** that spans physical image capture, landmark coordinate extraction, invariant feature transformations, synthetic dataset expansion, and temporal sequence tensor generation.

```mermaid
flowchart TD
    subgraph "Tier 1: Physical Data Ingestion & Collection"
        A1["Hugging Face Datasets<br/>(Marxulia v02 & v03)"] -->|download_hf_asl.py| B1["Raw Image Folders<br/>(data/online_asl/A..Z)"]
        A2["Webcam Video Stream<br/>(HTML5 / getUserMedia)"] -->|/dataset Studio Burst Capture| B1
    end

    subgraph "Tier 2: Landmark Extraction"
        B1 -->|build_combined_dataset.py<br/>MediaPipe Hands (static_image_mode=True)| C1["63 Landmark Coordinates<br/>(f0 .. f62)"]
        C2["Historical Archive CSV<br/>(archive_image_landmarks.csv)"] --> C1
        C1 -->|Deduplication & Cache Tagging| D1["online_asl_landmarks.csv<br/>(8,749 Samples)"]
    end

    subgraph "Tier 3: Feature Engineering & Transformation"
        D1 -->|src/processing/features.py| E1["85 Discriminative Features<br/>(Ratios, Distances, Palm Normal)"]
    end

    subgraph "Tier 4: Synthetic Data Generation (2x Scaling)"
        E1 -->|src/processing/augmentation.py<br/>Mirroring, Rotation, Scale, Jitter| F1["17,498 Augmented Sequences"]
    end

    subgraph "Tier 5: Temporal Sequence Tensor Generation"
        F1 -->|src/processing/sequences.py<br/>30-Timestep Windowing| G1["Tensor Tensors<br/>X: (17498, 30, 85), y: (17498, 26)"]
    end
```

### Stage 1: Raw Image Generation (Webcam Burst Capture & In-Browser Studio)
- **Client-Side Generation Mechanism**:
  - In `templates/dataset.html`, the user selects an ASL target letter ($A$ through $Z$) and clicks **Start Recording**.
  - A high-frequency JavaScript timer captures a continuous burst of **20 frames** from the webcam at 150ms intervals.
  - Each frame is drawn to an off-screen HTML5 `<canvas>`, rendered as a JPEG data URL (`data:image/jpeg;base64,...`), and posted to `/api/collect`.
- **Server-Side Storage Engine (`/api/collect` in `app.py`)**:
  - Validates base64 data URL formatting and decodes the image using OpenCV (`cv2.imdecode`).
  - Scans `data/online_asl/<label>/`, determines the current maximum sequence number $N$, and saves the frame as `data/online_asl/<label>/f'{N+1:05d}.jpg'` at 95% JPEG quality.
  - Dynamically returns updated sample counts so the web UI updates its progress counter in real time.

### Stage 2: Landmark Matrix Generation (MediaPipe 21 Hand Joints)
- **Execution Script**: `python scripts/build_combined_dataset.py --input data/online_asl --max-per-class 350`
- **Extraction Mechanics**:
  - Uses `LandmarkExtractor(static_image_mode=True, min_detection_confidence=0.30)`.
  - For each image, MediaPipe scans for hand presence and outputs 21 normalized 3D keypoints:
    $$\mathbf{P}_i = (x_i, y_i, z_i) \quad \text{for } i \in [0, 20]$$
  - Failed detections (blurry images or hands outside the camera field) are automatically discarded.
  - The 21 joints are serialized into a 63-dimensional coordinate array:
    $$\mathbf{f} = [x_0, y_0, z_0, x_1, y_1, z_1, \dots, x_{20}, y_{20}, z_{20}]$$
- **Archive Ingestion & Synchronization**:
  - Normalizes historical archive sign labels via `normalize_archive_label_name()` to resolve legacy label mismatches.
  - Merges extracted landmarks with `archive_image_landmarks.csv` to produce `online_asl_landmarks.csv` (**8,749 validated rows**).
  - Emits the cache tag `v3-augmented-scale` in `online_asl_landmarks.csv.meta.json`.

### Stage 3: High-Dimensional Feature Generation (100+ Spatial Descriptors)
- **Execution Module**: `src/processing/features.py` (`build_feature_dataset`)
- Transforms raw $(x, y, z)$ coordinates into rotation-, scale-, and position-invariant geometric metrics:
  1. **Wrist Normalization**: Translates coordinate origin to the wrist joint $\mathbf{P}_0 = (0, 0, 0)$.
  2. **Bounding Box Scaling**: Divides coordinates by the maximum span of the palm knuckles.
  3. **Finger Extension / Curl Ratios (5 Features)**: Calculates $\frac{\|\mathbf{P}_{\text{TIP}} - \mathbf{P}_{\text{WRIST}}\|}{\|\mathbf{P}_{\text{MCP}} - \mathbf{P}_{\text{WRIST}}\|}$ for thumb, index, middle, ring, and pinky.
  4. **Thumb-to-Knuckle Distances (4 Features)**: Measures Euclidean distance from thumb tip to all 4 finger MCP joints, allowing the model to distinguish compact fist signs ($A, E, S, T, M, N$).
  5. **Pairwise Inter-Fingertip Distances (10 Features)**: Evaluates spread vs. touching fingers for all 10 fingertip pairs.
  6. **3D Palm Normal Vector (3 Features)**: Computes cross-product $\mathbf{n} = \frac{\mathbf{v}_1 \times \mathbf{v}_2}{\|\mathbf{v}_1 \times \mathbf{v}_2\|}$, encoding hand orientation in 3D space.
- The output is an enriched tabular matrix with **85 numeric feature columns per sample**.

### Stage 4: Synthetic Dataset Generation via Augmentation (2x Scaling: 8,749 $\to$ 17,498)
- **Execution Module**: `src/processing/augmentation.py` (`augment_landmarks`)
- To provide robust left- and right-hand support and eliminate sensor sensitivity, new synthetic samples are generated on the fly:
  1. **Horizontal Mirroring Generator**: Inverts $X$-coordinates ($x' = -x$), effectively generating a synthetic left-handed sample for every right-handed sample.
  2. **2D Rotation Generator**: Rotates coordinates in the image plane by a random angle $\theta \in [-12^\circ, +12^\circ]$:
     $$x' = x\cos\theta - y\sin\theta, \quad y' = x\sin\theta + y\cos\theta$$
  3. **Scale Jitter Generator**: Multiplies coordinates by a random scaling factor $s \sim \mathcal{U}(0.92, 1.08)$.
  4. **Coordinate Noise Generator**: Adds Gaussian sensor jitter $\epsilon \sim \mathcal{N}(0, 0.005^2)$.
- Combined with the original samples, this generates a massive, balanced training dataset of **17,498 sequences**.

### Stage 5: BiLSTM Temporal Sequence Tensor Generation
- **Execution Module**: `src/processing/sequences.py` (`create_fixed_length_sequences`) & `app.py` (`build_live_sequence`)
- **Training Tensor Formulation**:
  - The Bidirectional LSTM expects 3D sequence tensors of shape `(Batch, Sequence_Length=30, Features=85)`.
  - Static sign samples are projected across a 30-timestep temporal window, simulating a continuous static hold:
    $$\mathbf{S} = \operatorname{repeat}(\mathbf{f}_{\text{sample}}, \text{repeats}=30, \text{axis}=0) \in \mathbb{R}^{30 \times 85}$$
  - Dynamic gestures ($J, Z$) slide a 30-frame temporal window across consecutive landmark extractions.
  - Final generated training tensors:
    - $\mathbf{X}_{\text{train}} \in \mathbb{R}^{17498 \times 30 \times 85}$
    - $\mathbf{y}_{\text{train}} \in \mathbb{R}^{17498 \times 26}$ (one-hot or label-encoded)

---

## 🧠 Deep Learning Model Architecture

The core classifier in [`src/models/bilstm.py`](file:///d:/Sign%20Lang/src/models/bilstm.py) is a **Deep Bidirectional LSTM**:

```text
Input Tensor: (Batch, Timesteps=30, Features=85)
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Bidirectional LSTM (96 units, return_sequences=True)   │  --> 192 features
│ LayerNormalization()                                   │
│ Dropout(0.25)                                          │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Bidirectional LSTM (64 units, return_sequences=False)  │  --> 128 features
│ LayerNormalization()                                   │
│ Dropout(0.25)                                          │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Dense(128, activation='relu')                          │
│ BatchNormalization()                                   │
│ Dropout(0.25)                                          │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Dense(64, activation='relu')                           │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Dense(26, activation='softmax')                        │  --> P(Class A..Z)
└────────────────────────────────────────────────────────┘
```

### Hyperparameters:
- **Sequence Window**: 30 timesteps (dynamic context sliding window).
- **Optimizer**: Adam with initial learning rate $\eta = 0.001$.
- **Learning Rate Schedule**: `ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5)`.
- **Early Stopping**: `EarlyStopping(patience=8, restore_best_weights=True)`.
- **Batch Size**: 64.

---

## 📊 Model Performance & Evaluation Benchmarks

### 1. Validation Split Metrics
Documented in [`src/models/model_quality.json`](file:///d:/Sign%20Lang/src/models/model_quality.json):

| Metric | Previous Baseline | Enhanced BiLSTM Model | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 84.96% | **95.54%** | **+10.58%** |
| **Precision** | 86.79% | **95.60%** | **+8.81%** |
| **Recall** | 84.96% | **95.54%** | **+10.58%** |
| **F1 Score** | 84.48% | **95.53%** | **+11.05%** |

### 2. Real Holdout Test Image Benchmark Across All 26 Classes
Evaluated with [`scripts/evaluate_all_classes.py`](file:///d:/Sign%20Lang/scripts/evaluate_all_classes.py) across 339 real images:

```text
======================================================================
  ASL ALPHABET RECOGNITION BENCHMARK REPORT (ALL 26 CLASSES)
======================================================================
[EXCELLENT] Class A: 14/14 (100.0%) | Avg Confidence:  96.7%
[EXCELLENT] Class B: 14/14 (100.0%) | Avg Confidence:  98.4%
[EXCELLENT] Class C: 12/13 ( 92.3%) | Avg Confidence:  90.0%
[EXCELLENT] Class D: 13/13 (100.0%) | Avg Confidence:  95.5%
[EXCELLENT] Class E: 13/14 ( 92.9%) | Avg Confidence:  93.6%
[EXCELLENT] Class F: 13/13 (100.0%) | Avg Confidence:  99.9%
[EXCELLENT] Class G: 12/13 ( 92.3%) | Avg Confidence:  96.9%
[EXCELLENT] Class H: 13/13 (100.0%) | Avg Confidence:  99.6%
[EXCELLENT] Class I: 12/12 (100.0%) | Avg Confidence:  98.5%
[EXCELLENT] Class J: 14/14 (100.0%) | Avg Confidence:  99.7%
[EXCELLENT] Class K: 13/13 (100.0%) | Avg Confidence:  98.8%
[EXCELLENT] Class L: 14/14 (100.0%) | Avg Confidence:  99.7%
[EXCELLENT] Class M: 14/14 (100.0%) | Avg Confidence:  97.8%
[EXCELLENT] Class N: 13/13 (100.0%) | Avg Confidence:  99.8%
[EXCELLENT] Class O: 12/12 (100.0%) | Avg Confidence:  92.9%
[GOOD     ] Class P: 11/13 ( 84.6%) | Avg Confidence:  95.7%
[EXCELLENT] Class Q: 10/10 (100.0%) | Avg Confidence:  99.5%
[EXCELLENT] Class R: 12/13 ( 92.3%) | Avg Confidence:  88.9%
[EXCELLENT] Class S: 11/12 ( 91.7%) | Avg Confidence:  94.9%
[EXCELLENT] Class T: 12/13 ( 92.3%) | Avg Confidence:  93.9%
[EXCELLENT] Class U: 12/13 ( 92.3%) | Avg Confidence:  85.5%
[EXCELLENT] Class V: 13/13 (100.0%) | Avg Confidence:  93.5%
[EXCELLENT] Class W: 12/13 ( 92.3%) | Avg Confidence:  99.7%
[EXCELLENT] Class X: 13/13 (100.0%) | Avg Confidence:  96.0%
[EXCELLENT] Class Y: 13/14 ( 92.9%) | Avg Confidence:  93.7%
[GOOD     ] Class Z: 10/13 ( 76.9%) | Avg Confidence:  89.3%
----------------------------------------------------------------------
TOTAL ACCURACY: 325 / 339 correct (95.87%)
======================================================================
```

---

## 🔤 Supported ASL Alphabet Signs

The model supports all 26 standard American Sign Language alphabet gestures:

| Class | Hand Shape Description | Category |
| :---: | :--- | :---: |
| **A** | Compact fist, thumb resting upright along the side of index finger | Static |
| **B** | Flat palm upright, four fingers held together, thumb folded across palm | Static |
| **C** | Fingers curved into a semi-circle resembling the letter "C" | Static |
| **D** | Index finger pointing straight up, remaining fingers curled touching thumb | Static |
| **E** | All fingertips curled tightly resting on top of the folded thumb | Static |
| **F** | Index finger and thumb tips touching (circle), other 3 fingers upright | Static |
| **G** | Index finger pointing horizontally forward, thumb parallel | Static |
| **H** | Index and middle fingers pointing horizontally together | Static |
| **I** | Pinky finger pointing straight up, all other fingers closed into a fist | Static |
| **J** | Pinky finger extended, tracing a "J" curve in the air | Dynamic |
| **K** | Index finger up, middle finger angled forward, thumb resting between them | Static |
| **L** | Thumb and index finger extended at $90^\circ$ forming an "L" shape | Static |
| **M** | Fist with thumb tucked beneath index, middle, and ring fingers | Static |
| **N** | Fist with thumb tucked beneath index and middle fingers | Static |
| **O** | All fingertips curved to touch the thumb tip, forming an "O" | Static |
| **P** | Like "K" pointed downward with palm facing down | Static |
| **Q** | Like "G" pointed downward with palm facing down | Static |
| **R** | Index and middle fingers extended and crossed over each other | Static |
| **S** | Tight fist with thumb wrapped across the front of the curled fingers | Static |
| **T** | Fist with thumb tucked under the index finger | Static |
| **U** | Index and middle fingers extended straight up, held close together | Static |
| **V** | Index and middle fingers extended straight up, spread apart in a "V" | Static |
| **W** | Index, middle, and ring fingers extended upright and spread apart | Static |
| **X** | Fist with index finger crooked like a hook | Static |
| **Y** | Thumb and pinky fingers extended outward, middle 3 fingers curled | Static |
| **Z** | Index finger tracing a "Z" pattern in the air | Dynamic |

---

## 🚀 Quickstart Guide for New Users

### Method 1: Docker Container (Recommended)

Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.

#### One-Command Start with Docker Compose:
```bash
docker compose up --build
```
Navigate to **`http://localhost:5000`** in your browser and click "Allow" on camera access.

#### Using Docker CLI directly:
```bash
# 1. Build the Docker image
docker build -t sign-language-app .

# 2. Run the container mapping port 5000
docker run -d -p 5000:5000 --name sign_lang_app sign-language-app

# 3. Check logs
docker logs -f sign_lang_app
```

To stop:
```bash
docker compose down
# or: docker stop sign_lang_app
```

---

### Method 2: Local Python Environment

#### 1. System Requirements
- Python 3.11 or 3.12
- Git
- A functioning webcam

#### 2. Clone the Repository
```bash
git clone https://github.com/Jayasurya786/Sign_Lang.git
cd Sign_Lang
```

#### 3. Set Up Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 4. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Dataset Setup (Bundled vs. Downloading Raw Images)
- **Preprocessed Landmarks (Ready-to-Use)**: The full master landmark training dataset (`data/processed/online_asl_landmarks.csv`, 8,749 samples across all 26 classes) is **bundled directly in this repository**. You can train models, run benchmarks, or launch the app immediately!
- **Download Raw Training Images (Optional)**: If you want to download all ~11,000 raw camera images into `data/online_asl/`, run the root-level download utility:
  ```bash
  # Download full raw image dataset:
  python download.py

  # Or fast download (60 images per letter class for quick testing):
  python download.py --quick
  ```

#### 6. Launch the Application
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 📖 In-Depth Application User Guide

### 1. Real-Time Sign Recognition
- Hold your hand roughly $0.5 - 1.5$ meters from the camera in a reasonably lit room.
- As you form an ASL letter, the model classifies the pose in real time.
- The **Confidence Bar** displays the model's certainty.
- Holding the sign steady for ~0.5 seconds locks the letter into the **Current Word**.

### 2. Live Skeletal Visualizer
- The live video element has a transparent HTML5 canvas overlay (`#skeleton-canvas`).
- MediaPipe 21 hand landmarks are mapped directly to your screen:
  - Green dots mark anatomical joints.
  - White lines draw the interconnecting bones.
- Works smoothly at 30+ FPS directly in the browser.

### 3. Text-to-Speech (TTS) Voice Engine
- **Auto-Speak**: Toggle the **Audio Voice (TTS)** checkbox. Whenever a word boundary is detected (space gesture or button), the completed word is automatically spoken aloud.
- **Manual Speak**: Click the **🔊 Speak** button at any time to hear the full sentence read aloud.

### 4. In-Air Gesture Controls & Shortcuts
In addition to the physical screen buttons, you can control the text editor using hands-free in-air gestures:
- **`SPACE`**: Show a flat open palm facing the camera, or click the `␣ Space` button.
- **`BACKSPACE`**: Retract your hand quickly into a fist or leftward swipe, or click the `⌫ Backspace` button.
- **`CLEAR`**: Click the `🗑️ Clear` button to wipe all accumulated words and sentences.

### 5. In-Browser Custom Dataset Studio (`/dataset`)
Want to improve accuracy for your own hand or add custom gestures?
1. Navigate to **`http://localhost:5000/dataset`**.
2. Select the target letter from the dropdown ($A$–$Z$).
3. Click **Start Recording**.
4. The system automatically captures a sequential 20-frame burst directly from your webcam into `data/online_asl/<letter>/`.
5. The sample counter updates dynamically.

### 6. Background Model Retraining from UI
- Click the **Train Model** button on the home dashboard.
- The Flask backend asynchronously executes the full feature extraction, augmentation, and BiLSTM training pipeline.
- Live progress is reported through the status indicator without freezing the web interface.

---

## 📱 TensorFlow Lite (TFLite) Edge & Mobile Deployment

The production model is exported to TensorFlow Lite format: [`src/models/sign_bilstm.tflite`](file:///d:/Sign%20Lang/src/models/sign_bilstm.tflite).

- **Size Comparison**:
  - Original Keras model: `3.89 MB`
  - Exported TFLite model: `1.28 MB` (reduced by **66.1%**)
- **Target Platforms**: Raspberry Pi, Android, iOS, microcontrollers, edge TPUs.

### Running Inference with TFLite in Python:
```python
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="src/models/sign_bilstm.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare sample sequence: shape (1, 30, num_features)
sample_input = np.random.randn(1, 30, input_details[0]['shape'][2]).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], sample_input)
interpreter.invoke()

# Retrieve class probabilities
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class_index = np.argmax(output_data[0])
print(f"Predicted class index: {predicted_class_index}")
```

To re-export or verify numerical parity:
```bash
python scripts/export_tflite.py
```

---

## ☁️ Cloud Deployment Guide

Because the application captures webcam video in the client browser (via HTML5 canvas and JavaScript) and transmits frames over HTTP POST, **it can be hosted on any cloud provider without requiring a physical camera on the server**.

### Option 1: Render / Railway (Free & Simple)
1. Fork or push this repository to your GitHub account.
2. Sign in to [Render](https://render.com) or [Railway](https://railway.app).
3. Create a **New Web Service** and connect the repository.
4. Set the Build Command:
   ```bash
   pip install -r requirements.txt
   ```
5. Set the Start Command:
   ```bash
   python app.py
   ```
   *(Or using Gunicorn: `gunicorn app:app --bind 0.0.0.0:$PORT`)*

### Option 2: Docker on AWS EC2 / DigitalOcean / GCP Cloud Run
```bash
# Build and run container with restart policy
docker build -t sign-lang-prod .
docker run -d -p 80:5000 --restart always --name sign_app sign-lang-prod
```

---

## ️ Complete Dataset & Script Pipeline Guide

The system includes a 1-step root download utility (`download.py`) and an end-to-end data engineering, validation, landmark extraction, and model training pipeline located in the `scripts/` directory.

| Script | Command | Primary Function |
| :--- | :--- | :--- |
| **`download.py`** | `python download.py` | **1-Step Root Downloader**: Downloads raw ASL images, verifies bundled landmarks, supports `--quick` and `--extract`. |
| **`download_hf_asl.py`** | `python scripts/download_hf_asl.py` | Ingests multi-source public datasets from Hugging Face Hub with in-memory caching and optional `--build-landmarks`. |
| **`check_dataset_balance.py`** | `python scripts/check_dataset_balance.py` | Audits class distribution, identifies missing/underfilled classes, initializes folders. |
| **`build_combined_dataset.py`** | `python scripts/build_combined_dataset.py` | Extracts MediaPipe hand landmarks from raw images and writes processed CSVs with cache tags. |
| **`train_model.py`** | `python scripts/train_model.py` | Computes 100+ geometric features, applies 4-way augmentation, and trains the BiLSTM model. |
| **`evaluate_all_classes.py`** | `python scripts/evaluate_all_classes.py` | Evaluates holdout test images across all 26 classes, printing per-sign accuracy and confidence. |
| **`export_tflite.py`** | `python scripts/export_tflite.py` | Converts Keras model to ultra-lightweight `.tflite` format and validates numerical parity. |

---

### 1. Dataset Architecture & File Hierarchy

The dataset pipeline separates raw image files from processed landmark coordinates:

```text
data/
├── online_asl/                           # Raw image dataset (organized by letter)
│   ├── A/                                # Sign class subfolder
│   │   ├── 00001.jpg                     # Sequential RGB JPEG images (quality 95)
│   │   ├── 00002.jpg
│   │   └── ... (~340 images)
│   ├── B/
│   │   └── ...
│   └── ... Z/
└── processed/                            # Extracted landmark matrices & caches
    ├── online_asl_landmarks.csv          # Master training dataset (8,749 rows)
    ├── online_asl_landmarks.csv.meta.json# Cache metadata file (tag: v3-augmented-scale)
    └── archive_image_landmarks.csv       # Supplementary historical archive landmarks
```

---

### 2. Dataset Schema & Metadata Caching

#### Processed Landmark CSV Format (`online_asl_landmarks.csv`):
Each row represents a single static hand pose:
- **`label`**: Target ASL letter (`A` through `Z`).
- **`f0` through `f62`**: 63 raw normalized landmark coordinate values corresponding to the 21 MediaPipe hand joints:
  $$\mathbf{f} = [x_0, y_0, z_0, x_1, y_1, z_1, \dots, x_{20}, y_{20}, z_{20}]$$
  - Joint index $0$: Wrist
  - Joint indices $1 - 4$: Thumb (CMC, MCP, IP, Tip)
  - Joint indices $5 - 8$: Index Finger (MCP, PIP, DIP, Tip)
  - Joint indices $9 - 12$: Middle Finger (MCP, PIP, DIP, Tip)
  - Joint indices $13 - 16$: Ring Finger (MCP, PIP, DIP, Tip)
  - Joint indices $17 - 20$: Pinky Finger (MCP, PIP, DIP, Tip)

#### Cache Metadata System (`.meta.json`):
To prevent slow re-extractions of thousands of images upon server restarts, the system writes a metadata sidecar file:
```json
{
  "version": "v3-augmented-scale",
  "num_samples": 8749
}
```
If `app.py` detects a valid cached CSV matching the current `CACHE_VERSION`, it loads the pre-computed landmarks instantly (sub-second startup).

---

### 3. Script-by-Script Reference & CLI Arguments

#### Script 1A: `download.py` — 1-Step Root Dataset Downloader & Setup
The fastest and most convenient way to set up the raw image dataset on your machine. Located directly in the root directory.

```bash
python download.py [OPTIONS]
```

**Supported Arguments:**
- `--quick`: Quick test mode; downloads a fast subset of at most 60 images per letter class.
- `--extract`: Automatically runs MediaPipe landmark extraction on newly downloaded images immediately after download.
- `--max-per-class <int>`: Custom limit on images downloaded per class (`0` for all available).
- `--output <dir>`: Target folder for images (Default: `data/online_asl`).

**Usage Examples:**
```bash
# Standard download (full ~11,000 images):
python download.py

# Fast download for quick testing (60 images per letter):
python download.py --quick

# Download raw images AND automatically extract landmarks in 1 step:
python download.py --extract
```

---

#### Script 1B: `scripts/download_hf_asl.py` — Hugging Face Dataset Downloader
Direct CLI script for downloading and formatting ASL datasets from the Hugging Face Hub. Features in-memory sample indexing (100x faster than disk scanning) and live progress rates.

```bash
python scripts/download_hf_asl.py [OPTIONS]
```

**Supported Arguments:**
- `--output <dir>`: Destination folder for downloaded images (Default: `data/online_asl`).
- `--datasets <id1> [id2 ...]`: Hugging Face dataset IDs to download (Default: `Marxulia/asl_sign_languages_alphabets_v03` and `Marxulia/asl_sign_languages_alphabets_v02`).
- `--max-per-class <int>`: Cap the number of downloaded images per letter (`0` for unlimited).
- `--build-landmarks`: Automatically trigger landmark extraction after image downloads complete.

**Usage Examples:**
```bash
# Ingest full multi-source dataset into data/online_asl:
python scripts/download_hf_asl.py

# Ingest with a maximum cap of 250 images per class and build landmarks:
python scripts/download_hf_asl.py --max-per-class 250 --build-landmarks

# Download from a specific custom Hugging Face repository:
python scripts/download_hf_asl.py --datasets Marxulia/asl_sign_languages_alphabets_v03 --output data/custom_asl
```

---

#### Script 2: `check_dataset_balance.py` — Class Balance Auditor
Audits your image directories to ensure every class from `A` to `Z` has sufficient samples for balanced neural network training.

```bash
python scripts/check_dataset_balance.py [OPTIONS]
```

**Supported Arguments:**
- `--root <dir>`: Path to the dataset root directory (Default: `data/online_asl`).
- `--min-per-class <int>`: Minimum recommended sample count per class (Default: `300`).
- `--init`: Automatically creates any missing `A` through `Z` subdirectories.

**Return Codes (CI/CD Ready):**
- Returns `0` if all 26 classes meet or exceed the `--min-per-class` threshold.
- Returns `1` if any class is missing or underfilled (ideal as a pre-commit or training pipeline gate).

**Usage Examples:**
```bash
# Initialize missing A-Z class directories:
python scripts/check_dataset_balance.py --root data/online_asl --init

# Audit dataset balance with a 300-sample threshold:
python scripts/check_dataset_balance.py --root data/online_asl --min-per-class 300
```

---

#### Script 3: `build_combined_dataset.py` — Landmark Extractor & Merger
Iterates through image directories, executes MediaPipe Hand Landmark extraction, merges new extractions with existing archive data, removes corrupted records, and updates the cache metadata.

```bash
python scripts/build_combined_dataset.py [OPTIONS]
```

**Supported Arguments:**
- `--input <dir>`: Root image folder containing letter subdirectories (Default: `data/online_asl`).
- `--archive-csv <path>`: Path to optional historical archive landmark CSV (Default: `data/processed/archive_image_landmarks.csv`).
- `--output-csv <path>`: Destination CSV for merged landmarks (Default: `data/processed/online_asl_landmarks.csv`).
- `--max-per-class <int>`: Maximum images to extract per class to ensure balance (`0` for all, Default: `320`).

**How It Works:**
1. Initializes `LandmarkExtractor(static_image_mode=True, min_detection_confidence=0.30)`.
2. Reads images via OpenCV, filters unreadable or un-detectable hands.
3. Packages 63 landmark coordinates into normalized row records.
4. Normalizes historical archive labels using `normalize_archive_label_name()`.
5. Merges, validates, writes CSV, and generates `.meta.json` with cache version `v3-augmented-scale`.

**Usage Examples:**
```bash
# Build unified dataset with up to 350 samples per class:
python scripts/build_combined_dataset.py --input data/online_asl --max-per-class 350

# Extract from custom directory without merging historical archive:
python scripts/build_combined_dataset.py --input data/custom_asl --archive-csv data/none.csv --output-csv data/processed/custom_landmarks.csv
```

---

#### Script 4: `train_model.py` — BiLSTM Model Training Pipeline
The complete, self-contained neural network training script.

```bash
python scripts/train_model.py
```

**Pipeline Execution Flow:**
1. **Data Ingestion**: Loads `data/processed/online_asl_landmarks.csv` (8,749 samples).
2. **Feature Engineering**: Calls [`src/processing/features.py`](file:///d:/Sign%20Lang/src/processing/features.py) to compute 100+ discriminative spatial metrics (5 finger extension ratios, 4 thumb-to-knuckle distances, 10 inter-fingertip distances, and 3D palm normal orientation vectors).
3. **Data Augmentation**: Calls [`src/processing/augmentation.py`](file:///d:/Sign%20Lang/src/processing/augmentation.py) to apply horizontal coordinate flipping, random rotation ($\pm 12^\circ$), scale scaling ($0.92 - 1.08$), and Gaussian noise, expanding the dataset to **17,498 training sequences**.
4. **Sequence Generation**: Converts spatial landmark rows into 30-timestep temporal sequence tensors `(Batch, 30, Features)`.
5. **Stratified Split**: Splits dataset 80% for training and 20% for validation.
6. **Model Compilation & Training**: Compiles the 2-layer BiLSTM model with `LayerNormalization` and `Dropout(0.25)`. Trains using `Adam(lr=0.001)` with `ReduceLROnPlateau(factor=0.5, patience=4)` and `EarlyStopping(patience=8)`.
7. **Artifact Export**:
   - Saves production model to [`src/models/sign_bilstm.keras`](file:///d:/Sign%20Lang/src/models/sign_bilstm.keras).
   - Generates full evaluation metrics and confusion matrix in [`src/models/model_quality.json`](file:///d:/Sign%20Lang/src/models/model_quality.json).
   - Exports label mapping in [`src/models/model_classes.json`](file:///d:/Sign%20Lang/src/models/model_classes.json).

---

#### Script 5: `evaluate_all_classes.py` — Holdout Image Evaluator
Runs an exhaustive classification benchmark across real holdout images from every letter class, evaluating camera-readiness in real-world conditions.

```bash
python scripts/evaluate_all_classes.py [OPTIONS]
```

**Supported Arguments:**
- `--root <dir>`: Root directory of holdout images (Default: `data/online_asl`).
- `--samples-per-class <int>`: Number of random unseen images to evaluate per letter (Default: `15`).

**Outputs:**
- Per-class accuracy percentage and average model confidence.
- Confusion diagnostics for any misclassified signs.
- Total benchmark accuracy summary across all 26 classes (achieving **95.87%**).

**Usage Example:**
```bash
python scripts/evaluate_all_classes.py --samples-per-class 20
```

---

#### Script 6: `export_tflite.py` — TFLite Exporter & Parity Validator
Exports the trained Keras BiLSTM model to an edge-optimized TensorFlow Lite (`.tflite`) file and verifies mathematical parity.

```bash
python scripts/export_tflite.py
```

**Key Capabilities:**
- **Dynamic Control Flow Wrapping**: Bidirectional LSTMs use dynamic control flow that breaks naive `from_keras_model` conversions. `export_tflite.py` wraps the model in a concrete `tf.function` signature `(1, 30, num_features)` to ensure a reliable conversion graph.
- **Select TF Ops**: Enables `tf.lite.OpsSet.SELECT_TF_OPS` for full RNN cell operator support.
- **Automated Parity Verification**: Generates random test tensors, executes both the Keras model and the TFLite interpreter, and confirms that maximum absolute difference is $< 10^{-7}$.
- Outputs: [`src/models/sign_bilstm.tflite`](file:///d:/Sign%20Lang/src/models/sign_bilstm.tflite) (1.28 MB).

---

### 4. End-to-End Walkthrough: Retraining on Custom Data

Follow this complete step-by-step tutorial to ingest new sign data, extract landmarks, and produce an updated model:

#### Step 1: Collect Custom Images
Use the in-browser studio at `http://localhost:5000/dataset` to record bursts of 20 webcam frames for your target classes, or manually place JPEG images into:
```text
data/online_asl/<CLASS_NAME>/
```

#### Step 2: Audit Dataset Balance
```bash
python scripts/check_dataset_balance.py --root data/online_asl --min-per-class 300
```
Fix any classes flagged as missing or underfilled.

#### Step 3: Extract Landmarks & Rebuild Cache
```bash
python scripts/build_combined_dataset.py --input data/online_asl --max-per-class 350
```
This extracts MediaPipe hand landmarks from your images and writes `data/processed/online_asl_landmarks.csv` with updated metadata.

#### Step 4: Retrain the Neural Network
```bash
python scripts/train_model.py
```
Trains the BiLSTM classifier, reports validation accuracy, and saves `src/models/sign_bilstm.keras`.

#### Step 5: Validate on Holdout Test Images
```bash
python scripts/evaluate_all_classes.py
```
Verifies that classification accuracy across all 26 classes remains $> 95\%$.

#### Step 6: Export Lightweight Edge Model
```bash
python scripts/export_tflite.py
```
Exports `src/models/sign_bilstm.tflite` for edge deployment.

---

## 📡 Comprehensive REST API Specification

### 1. `GET /`
Returns the main live recognition single-page application.

### 2. `GET /dataset`
Returns the in-browser custom dataset recording studio.

### 3. `POST /api/predict`
Accepts a base64 encoded video frame and returns the recognized sign, confidence, and skeletal coordinates.

**Request Payload:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response Payload (Success):**
```json
{
  "status": "success",
  "letter": "B",
  "confidence": 0.984,
  "word": "HELL",
  "sentence": "HELLO WORLD",
  "landmarks": [
    [0.512, 0.784],
    [0.485, 0.723],
    [0.461, 0.654],
    "..."
  ]
}
```

### 4. `POST /api/collect`
Saves a frame captured in the `/dataset` studio to the target class folder.

**Request Payload:**
```json
{
  "label": "A",
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response Payload:**
```json
{
  "status": "success",
  "message": "Saved sample 341 for class A.",
  "label": "A",
  "sample_index": 341,
  "total_class_samples": 341
}
```

### 5. `POST /api/clear-buffer`
Clears the temporal feature window and accumulated words/sentences.

**Response:**
```json
{
  "status": "success",
  "message": "Prediction buffer cleared."
}
```

### 6. `GET /api/classes`
Returns the list of all 26 supported ASL alphabet sign labels.

**Response:**
```json
{
  "classes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
  "total": 26
}
```

### 7. `POST /api/retrain`
Starts background model retraining on the current dataset.

---

## 🧪 Automated Test Suite

The project includes an automated test suite covering feature computation, data augmentation, model architectures, TFLite parity, and Flask endpoints:

```bash
python -m pytest -v
```

### Test Coverage Breakdown (27 Tests):
- [`tests/test_enhanced_features.py`](file:///d:/Sign%20Lang/tests/test_enhanced_features.py):
  - `test_compute_sample_features_has_all_discriminative_metrics`: Verifies finger extension ratios, thumb-to-knuckle metrics, and 3D palm normal vectors.
  - `test_build_feature_dataset_enriches_dataframe`: Validates feature dataframe dimensions.
  - `test_augment_landmarks_adds_flipped_and_jittered_samples`: Tests horizontal mirroring and jitter augmentation.
  - `test_horizontal_flip_inverts_x_coordinate`: Verifies left-hand / right-hand coordinate flipping.
  - `test_bilstm_classifier_builds_with_enriched_dimensions`: Ensures BiLSTM layers compile properly.
  - `test_tflite_model_executes_successfully`: Tests TFLite execution and output tensor shape.
  - `test_landmark_extractor_supports_multi_hand_signature`: Tests multi-hand tracking API (`max_num_hands=2`).
- [`tests/test_bilstm_training.py`](file:///d:/Sign%20Lang/tests/test_bilstm_training.py): Sequence generation and label encoding.
- [`tests/test_dataset_config.py`](file:///d:/Sign%20Lang/tests/test_dataset_config.py): ASL 26-class validation and string normalization.
- [`tests/test_prediction_validation.py`](file:///d:/Sign%20Lang/tests/test_prediction_validation.py): Majority voting, confidence thresholding, live prediction debouncing, and API error handling.
- [`tests/test_preprocessing.py`](file:///d:/Sign%20Lang/tests/test_preprocessing.py): Outlier removal, IQR filtering, coordinate normalization, and confusion matrix reporting.

---

## ⚙️ Configuration & Environment Variables

| Variable | Default Value | Description |
| :--- | :---: | :--- |
| `PORT` | `5000` | Port for the web server. |
| `SIGN_LANG_DATASET_PATH` | *(empty)* | Optional path to an external offline dataset directory. |
| `FLASK_APP` | `app.py` | Flask application entrypoint. |
| `PYTHONUNBUFFERED` | `1` | Forces unbuffered stdout/stderr for real-time Docker logging. |

---

## 📁 Project Directory Structure

```text
Sign_Lang/
├── Dockerfile                  # Production container definition (Debian + OpenCV/MediaPipe)
├── docker-compose.yml          # Docker compose file for 1-command startup
├── .dockerignore               # Optimized build context exclusions
├── app.py                      # Core Flask web application & API routing
├── requirements.txt            # Python dependencies
├── README.md                   # Complete system documentation
├── DEPLOYMENT.md               # Cloud hosting guide (Render, Railway, Hugging Face)
├── scripts/
│   ├── build_combined_dataset.py   # Multi-source dataset merger & landmark extractor
│   ├── check_dataset_balance.py    # Class distribution diagnostic utility
│   ├── download_hf_asl.py          # Hugging Face dataset downloader
│   ├── evaluate_all_classes.py     # Comprehensive holdout test set benchmark
│   ├── export_tflite.py            # TensorFlow Lite exporter with parity check
│   └── train_model.py              # BiLSTM training script with callbacks
├── src/
│   ├── data/
│   │   ├── dataset_config.py       # ASL 26-class definitions & validation
│   │   └── landmarks.py            # MediaPipe hand landmark extraction (multi-hand)
│   ├── models/
│   │   ├── bilstm.py               # BiLSTM neural network definition & training
│   │   ├── evaluation.py           # Precision, recall, F1, confusion matrix reporting
│   │   ├── model_classes.json      # Class label index mappings
│   │   ├── model_quality.json      # Model performance statistics
│   │   ├── realtime.py             # Prediction smoothing & temporal sliding window
│   │   ├── sign_bilstm.keras       # Trained Keras production model (3.89 MB)
│   │   └── sign_bilstm.tflite      # Exported TFLite edge model (1.28 MB)
│   └── processing/
│       ├── augmentation.py         # Mirroring, rotation, scale, and noise augmentation
│       ├── cleaning.py             # Outlier rejection & coordinate normalization
│       ├── features.py             # 100+ geometric feature engineering functions
│       └── sequences.py            # Temporal sequence window generator
├── templates/
│   ├── home.html                   # Main interface (Skeletal visualizer, TTS, gestures)
│   └── dataset.html                # In-browser custom dataset recording studio
└── tests/
    ├── test_bilstm_training.py     # BiLSTM sequence & tensor tests
    ├── test_dataset_config.py      # ASL alphabet configuration tests
    ├── test_enhanced_features.py   # Features, TFLite, multi-hand, flip tests
    ├── test_prediction_validation.py # Prediction smoothing & API route tests
    └── test_preprocessing.py       # Data cleaning & metrics tests
```

---

## ❓ Troubleshooting & FAQs

### 1. Camera screen is black or permissions are denied
- **Browser Permission**: Make sure camera access is allowed. In Chrome, click the camera icon on the address bar or visit `chrome://settings/content/camera`.
- **Hardware Conflict**: Ensure no other application (Zoom, Teams, OBS, Skype) is currently using the webcam.

### 2. Audio Text-to-Speech (TTS) is not playing
- Modern web browsers block audio autoplay until the user interacts with the page.
- Click anywhere on the webpage or press the **🔊 Speak** button once to initialize the browser's audio synthesis context.

### 3. Docker fails on Windows (`cannot find file specified`)
- Ensure **Docker Desktop** is running in your taskbar before executing `docker compose up`.
- If using WSL 2, make sure the WSL 2 integration is enabled in Docker Desktop settings.

### 4. How can I get the highest possible recognition accuracy?
- **Lighting**: Front lighting onto your hands works significantly better than strong backlighting.
- **Hand Visibility**: Ensure your entire hand (from wrist to fingertips) is in the camera view.
- **Steady Poses**: Hold the sign steady for half a second for optimal temporal window smoothing.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute for personal, educational, and commercial purposes.
