# Arabic-NLP-Project

This project focuses on Arabic Natural Language Processing (NLP), specifically **Text Classification** and **Text Summarization**, using both **Traditional** and **Modern** approaches.

---

## 📌 Project Overview

| Phase     | Approach       | Task             | Methods / Models Used                      |
|-----------|----------------|------------------|---------------------------------------------|
| Phase 1   | Traditional     | Classification   | TF-IDF + Naive Bayes / SVM                  |
|           |                | Summarization    | Frequency-based Extractive Summarization    |
| Phase 2   | Modern (DL & Transformers) | Classification   | LSTM, AraBERT (Transformers)                |
|           |                | Summarization    | LSTM + Attention, mT5 (XLSum) Transformer   |

---

## 🔹 Phase 1 – Traditional NLP Methods

**Notebook:** `Nlp_phase1 (3).ipynb`

### 🧪 Text Classification:
- Arabic text cleaning, normalization, stemming using `ISRIStemmer`
- TF-IDF vectorization with `TfidfVectorizer`
- Classifiers:
  - Multinomial Naive Bayes
  - Linear SVM (`LinearSVC`)
- Evaluation using accuracy, confusion matrix, classification report

### 📝 Text Summarization:
- Extractive method based on word frequency scoring
- Sentence selection using vector-based ranking

---

## 🔸 Phase 2 – Modern Methods

### 🟩 Text Classification:

#### 1. `LSTM Classification.ipynb`
- Tokenization + Padding using Keras
- Embedding Layer + LSTM + Dense Layers
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

#### 2. `AraBert Classification.ipynb`
- Preprocessing with `ArabertPreprocessor`
- Tokenization using `AutoTokenizer` from `aubmindlab/bert-base-arabertv2`
- Fine-tuning `AutoModelForSequenceClassification` using Hugging Face `Trainer`

---

### 🟨 Text Summarization:

#### 1. `LSTM+Attntion.ipynb`
- Encoder-Decoder architecture using LSTM
- Custom attention mechanism
- Vocabulary built manually with tokenization via `spaCy`
- `<sos>`, `<eos>` tokens used for seq2seq generation

#### 2. `Transformer.ipynb`
- Model: `mT5_multilingual_XLSum`
- Summarization pipeline using Hugging Face Transformers
- Tokenization and generation with `T5Tokenizer`
- Evaluation through pipeline results

---

## 🛠️ Libraries & Tools

- **NLP & ML:** `nltk`, `sklearn`, `spaCy`, `arabert`, `transformers`, `datasets`
- **Deep Learning:** `TensorFlow`, `Keras`, `PyTorch`
- **Other:** `pandas`, `numpy`, `matplotlib`, `tqdm`, `sentencepiece`, `evaluate`

---




