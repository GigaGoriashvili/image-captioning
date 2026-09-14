# 🖼️ Neural Image Caption Generator (CNN-LSTM)

[![Open In Colab (Training)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GigaGoriashvili/image-captioning/blob/main/data_and_training.ipynb)
[![Open In Colab (Inference)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GigaGoriashvili/image-captioning/blob/main/inference.ipynb)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras)
![Dataset](https://img.shields.io/badge/Dataset-Flickr8k-green)
![License](https://img.shields.io/badge/License-MIT-purple)

An end-to-end Deep Learning pipeline for automated **Image Captioning** that fuses Computer Vision and Natural Language Processing. The project implements a **Merge-Model CNN-LSTM Encoder-Decoder architecture** trained on the Flickr8k dataset, featuring pre-trained visual feature extraction, tokenized caption generation, a memory-optimized data streaming generator, and an interactive standalone inference engine.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Model Architecture](#-model-architecture)
- [Repository Structure](#-repository-structure)
- [Security & Credential Management](#-security--credential-management)
- [Dataset Information](#-dataset-information)
- [Installation & Setup](#-installation--setup)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
  - [1. Data Preparation & Training](#1-data-preparation--training)
  - [2. Standalone Inference & Testing](#2-standalone-inference--testing)
- [Key Features & Design Choices](#-key-features--design-choices)
- [Error Analysis & Known Limitations](#-error-analysis--known-limitations)
- [Future Enhancements](#-future-enhancements)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🌟 Project Overview

Image captioning is the task of generating a human-readable natural language description of visual content in an image. This system combines:
1. **Visual Encoder**: A pre-trained **VGG16** Convolutional Neural Network used as a feature extractor to convert raw pixels into rich 4,096-dimensional visual latent representations.
2. **Language Decoder**: A sequence processor employing word embeddings and a **Long Short-Term Memory (LSTM)** network to model linguistic context.
3. **Multimodal Fusion (Merge Architecture)**: Merges visual representations with partial caption sequences to predict the next word token through categorical cross-entropy optimization.

---

## 🧠 Model Architecture

The model follows the **Merge Architecture** (Tanti et al.), keeping visual representations separate from the recurrent language model before fusing them in a shared feed-forward multimodal space.

```
                  +-------------------------------+
                  |      Raw Image (224x224)      |
                  +---------------+---------------+
                                  |
                                  v
                  +-------------------------------+
                  |      VGG16 (Pretrained)       |
                  |  Extract FC2 Layer: (4096,)   |
                  +---------------+---------------+
                                  |
                                  v
                  +-------------------------------+
                  |          Dropout(0.5)         |
                  |        Dense(256, ReLU)       |
                  +---------------+---------------+
                                  |
                                  |  [Visual Vector (256,)]
                                  v
      +------------------------->(+) <-------------------------+
      |                           |                            |
      |             +-------------+-------------+              |
      |             |     Dense(256, ReLU)      |              |
      |             +-------------+-------------+              |
      |                           |                            |
      |                           v                            |
      |             +---------------------------+              |
      |             | Dense(vocab_size, Softmax)|              |
      |             +-------------+-------------+              |
      |                           |                            |
      |                           v                            |
      |                   [Next Word Token]                    |
      |                                                        |
      | [Text Vector (256,)]                                   |
+-----+-------------------------+                              |
|          LSTM(256)            |                              |
+-----+-------------------------+                              |
      |                                                        |
+-----+-------------------------+                              |
|        Dropout(0.5)           |                              |
+-----+-------------------------+                              |
|   Embedding(8353, 256)        |                              |
+-----+-------------------------+                              |
|   Input Sequence (max_len=35) |                              |
+-----+-------------------------+                              |
      |                                                        |
      +--------------------------------------------------------+
```

### Layer Specification

| Layer | Input Shape | Output Shape | Parameters / Activation | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Image Input (`inputs1`)** | `(4096,)` | `(4096,)` | — | VGG16 penultimate layer features |
| **Image Dropout (`fe1`)** | `(4096,)` | `(4096,)` | Rate: `0.5` | Regularization to prevent overfitting |
| **Image Projection (`fe2`)** | `(4096,)` | `(256,)` | Dense, `ReLU` | Linear projection of visual features |
| **Text Input (`inputs2`)** | `(35,)` | `(35,)` | — | Tokenized word indices with padding |
| **Text Embedding (`se1`)** | `(35,)` | `(35, 256)` | `mask_zero=True` | Word embedding representation |
| **Text Dropout (`se2`)** | `(35, 256)` | `(35, 256)` | Rate: `0.5` | Regularization on word representations |
| **LSTM (`se3`)** | `(35, 256)` | `(256,)` | 256 units | Recurrent sequence encoding |
| **Fusion Layer (`decoder1`)** | `[(256,), (256,)]`| `(256,)` | Element-wise `add` | Combines image and sequence modalities |
| **Dense Classifier (`decoder2`)**| `(256,)` | `(256,)` | Dense, `ReLU` | Multimodal representation learning |
| **Softmax Output (`outputs`)** | `(256,)` | `(8353,)` | Softmax | Probability distribution over vocabulary |

---

## 📁 Repository Structure

```plaintext
image-captioning/
├── .env.example              # Configuration template for paths & credentials
├── .gitignore                # Safeguards secrets, credentials, data & weights
├── README.md                 # Project documentation and architectural guide
├── requirements.txt          # Python runtime dependencies
├── data_and_training.ipynb   # Notebook 1: Preprocessing, generator, model training
└── inference.ipynb           # Notebook 2: Standalone inference, test runner & evaluation
```

---

## 🔐 Security & Credential Management

This project strictly adheres to secure coding standards and secret-zero principles.

### 1. Zero Hardcoded Credentials
- No API keys, passwords, authentication tokens, personal email addresses, or cloud secrets are stored in any source code, notebooks, or git history.
- Verification completed: Repository history and notebook metadata have been audited for credential leaks.

### 2. Safeguarded via `.gitignore`
A comprehensive [`.gitignore`](file:///.gitignore) file is configured to prevent accidental commits of:
- **Environment variables and secret files**: `.env`, `.env.*`, `kaggle.json`, `credentials.json`, `client_secret*.json`, `*.pem`, `*.key`.
- **Large binary weights and cached tensors**: `*.h5`, `*.pkl`, `features.pkl`, `tokenizer.pkl`, `final_caption_model.h5`.
- **Dataset folders**: `images/`, `captions.txt`, `data/`, `Flickr8k/`.
- **Operating system and IDE temporary files**: `.ipynb_checkpoints/`, `__pycache__/`, `.vscode/`, `.DS_Store`.

### 3. Recommended Environment Configuration
Copy the provided template and adapt your local or cloud paths without committing sensitive values:
```bash
cp .env.example .env
```

If you download datasets via Kaggle's API or use private object storage, read credentials from environment variables using `python-dotenv`:

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env if present

BASE_DIR = os.getenv("BASE_DIR", "./data")
WORKING_DIR = os.getenv("WORKING_DIR", "./working")
```

### 4. Best Practices for Google Colab
When working inside Google Colab:
- **Avoid hardcoding API keys in notebook cells.**
- Store private tokens (e.g., Kaggle, Hugging Face, or Google Cloud) using **Colab Secrets** (the 🔑 icon in the sidebar) and access them securely:
  ```python
  from google.colab import userdata
  kaggle_key = userdata.get('KAGGLE_KEY')
  ```

---

## 📊 Dataset Information

The model is trained on the **Flickr8k** benchmark dataset:
- **Total Images**: 8,091 real-world photographs depicting everyday activities, outdoor sports, animals, and people.
- **Captions**: 5 independent, human-annotated reference captions per image (totaling ~40,455 captions).
- **Format**:
  - `images/`: Directory containing JPEG images named by ID (e.g. `1000268201_693b08cb0e.jpg`).
  - `captions.txt`: CSV file structured as `image,caption`.

### Preprocessing Protocol
1. **Normalization**: Lowercase conversion and regex removal of special characters and digits.
2. **Sentence Boundary Markers**: Every caption is wrapped in delimiting tokens:
   $$\text{caption} = \text{"startseq "} + \text{tokens} + \text{" endseq"}$$
3. **Vocabulary Curation**: Words with length $> 1$ and alphabetic characters are indexed.
   - **Vocabulary Size ($V$)**: 8,353 unique tokens (including index 0 reserved for padding).
   - **Maximum Sequence Length ($T$)**: 35 tokens.

---

## 🚀 Installation & Setup

### Option 1: Running in Google Colab (Recommended)
You can train and run inference directly with free GPU acceleration using the Colab badges at the top of this document or by navigating to:
1. Training: [`data_and_training.ipynb`](file:///c:/Users/AdminR/Documents/image-captioning/data_and_training.ipynb)
2. Inference: [`inference.ipynb`](file:///c:/Users/AdminR/Documents/image-captioning/inference.ipynb)

### Option 2: Running Locally / Self-Hosted GPU Server

#### 1. Clone the Repository
```bash
git clone https://github.com/GigaGoriashvili/image-captioning.git
cd image-captioning
```

#### 2. Create and Activate a Virtual Environment
```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Prepare the Data & Working Directories
Create a `data/` directory and organize your files:
```plaintext
data/
├── captions.txt
└── images/
    ├── 1000268201_693b08cb0e.jpg
    └── ...
```

---

## 🔄 Pipeline Walkthrough

### 1. Data Preparation & Training

All preprocessing, generator creation, and model training routines reside in [`data_and_training.ipynb`](file:///c:/Users/AdminR/Documents/image-captioning/data_and_training.ipynb):

1. **Feature Extraction**:
   - VGG16 is instantiated with `include_top=True`, extracting the output of layer `-2` (dense layer `fc2`, 4,096 dimensions).
   - Extracted features are serialized to `features.pkl` to bypass re-computation in subsequent runs.
2. **Tokenization**:
   - `Tokenizer` is fitted across all 40,000+ captions and serialized to `tokenizer.pkl`.
3. **Data Generator**:
   - Generates streaming mini-batches $(X_1, X_2) \to y$ on the fly to prevent RAM saturation:
     - $X_1$: Image feature vector of shape `(4096,)`.
     - $X_2$: Left-aligned prefix sequence padded to length 35.
     - $y$: One-hot ground truth next token vector of length 8,353.
4. **Model Training**:
   - Train/Test split: 90% training (7,281 images) / 10% test (810 images).
   - Optimizer: `Adam(learning_rate=0.001)`.
   - Loss Function: `categorical_crossentropy`.
   - Epochs: 20 | Batch Size: 32.
   - Saves final weights to `final_caption_model.h5`.

### 2. Standalone Inference & Testing

The inference engine in [`inference.ipynb`](file:///c:/Users/AdminR/Documents/image-captioning/inference.ipynb) does not require raw training data:

1. **Loads Artifacts**:
   - Restores `tokenizer.pkl` and reconstructs the neural architecture with trained weights from `final_caption_model.h5`.
2. **Encodes Target Image**:
   - Downloads/loads any image, resizes to $(224 \times 224 \times 3)$, applies VGG16 preprocessing, and extracts the 4,096-dim visual feature vector.
3. **Autoregressive Greedy Decoding**:
   - Begins sequence with `"startseq"`.
   - At step $t$, feeds the current token sequence and image feature into the model.
   - Computes $\hat{y} = \operatorname{argmax}(\text{model.predict}([X_{\text{img}}, X_{\text{seq}}]))$.
   - Appends predicted word until `"endseq"` is generated or `max_length` (35) is reached.
4. **Visualization**:
   - Plots the image alongside its generated caption using Matplotlib.

#### Example Python Snippet for Programmatic Inference:
```python
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_caption(model, tokenizer, image_features, max_length=35):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        
        yhat = model.predict([image_features, sequence], verbose=0)
        word_idx = np.argmax(yhat)
        
        # Reverse map index to word
        word = tokenizer.index_word.get(word_idx, None)
        if word is None:
            break
            
        in_text += ' ' + word
        if word == 'endseq':
            break
            
    return in_text.replace('startseq', '').replace('endseq', '').strip()
```

---

## 💡 Key Features & Design Choices

- **Memory Efficiency**: The custom Python generator prevents Out-Of-Memory (OOM) crashes by creating word-level training pairs on-demand rather than pre-allocating the entire combinatorial matrix in RAM.
- **Merge-Model Formulation**: Keeping image features out of recurrent steps decreases parameter count and speeds up training convergence compared to injecting visual features at every LSTM recurrent step.
- **Zero Masking Handling**: The embedding layer utilizes `mask_zero=True` during training to ensure padded tokens do not bias the recurrent cell hidden state.
- **Standalone Artifact Decoupling**: Inference requires only the exported `final_caption_model.h5` and `tokenizer.pkl`, permitting deployment into lightweight microservices or APIs.

---

## 🔬 Error Analysis & Known Limitations

An honest assessment of potential failure cases and empirical constraints:

1. **Dataset Bias**:
   - Flickr8k is heavily skewed toward specific outdoor scenarios (e.g., dogs playing on grass, beach activities, people running).
   - **Failure Mode**: The model frequently generalizes poorly on indoor scenes, complex technical diagrams, documents, or exotic food images, often defaulting to generic descriptions like `"a dog is running through the grass"` or `"a person in a red shirt is standing"`.
2. **Greedy Search Pitfalls**:
   - Current decoding selects the single highest-probability token at each step ($\operatorname{argmax}$).
   - **Failure Mode**: Greedy search can fall into repetitive token loops or make early sub-optimal choices from which it cannot recover.
3. **Grammar & Alignment Artifacts**:
   - Occasionally, low-frequency tokens or rare spatial compositions result in mismatched prepositions or unnatural phrases.

---

## 🔮 Future Enhancements

- [ ] **Beam Search Decoding**: Implement $k$-beam search to evaluate multiple candidate hypothesis trajectories simultaneously.
- [ ] **Modern Visual Backbones**: Upgrade the image encoder from VGG16 to **ResNet50**, **EfficientNet**, or a **Vision Transformer (ViT)**.
- [ ] **Visual Attention Mechanisms**: Integrate Bahdanau or Luong spatial cross-attention (*Show, Attend and Tell*) to enable visual grounding of words.
- [ ] **Quantitative Evaluation Benchmarks**: Automated computation of standard NLP translation metrics: **BLEU-1 to BLEU-4**, **METEOR**, **ROUGE-L**, and **CIDEr**.
- [ ] **Web Deployment**: Package the inference engine into a responsive **FastAPI** backend with a Streamlit or React web interface.

---

## 📜 License & Acknowledgments

- **License**: This project is distributed under the [MIT License](https://opensource.org/licenses/MIT).
- **Dataset**: Flickr8k dataset created by Hodosh et al.
- **Pre-trained Weights**: ImageNet weights provided via `tensorflow.keras.applications`.
- **Author**: Giga Goriashvili ([GitHub](https://github.com/GigaGoriashvili))