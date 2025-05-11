# Multi-Kernel CNN + BiLSTM + Attention Model for Sequence Classification

##  Abstract
This model combines multi-scale convolutional neural networks (CNN), bidirectional LSTMs (BiLSTM), and an attention mechanism to efficiently capture both local n-gram features and long-range sequential dependencies in text or sequence-based inputs. It is designed to perform well in real-world cybersecurity applications such as phishing URL detection, intrusion detection, and protocol anomaly classification.

---

## Architecture Overview

**Input → Embedding → Multi-Kernel CNN → BiLSTM → Attention → Softmax Output**

### Diagram:

```
Input (Sequence)
     ↓
Embedding Layer
     ↓
Multi-Kernel CNN (kernels = 3, 5, 7)
     ↓
Concatenation + MaxPool
     ↓
BiLSTM Layer (bidirectional)
     ↓
Attention Layer (feature-level weighting)
     ↓
Fully Connected + Softmax (Multi-class)
```

---



## Why Use Them for URL Classification?

| Kernel Size | Captures Pattern Type | Example             |
|-------------|------------------------|---------------------|
| 2           | Character Bigrams      | `ht`, `tp`, `ww`    |
| 3           | Trigrams               | `www`, `log`, `com` |
| 4           | Sub-word tokens        | `http`, `.com`, `mail` |

These n-grams help the model detect:

- **Benign indicators** (e.g., `about`, `index`)
- **Phishing cues** (e.g., `login`, `update`, `verify`)
- **Malicious hints** (e.g., `exec`, `admin`, `.xyz`)

---
                   ┌────────────────────────────┐
                   │       Input Sequence       │
                   │   (Tokenized & Embedded)   │
                   └────────────┬───────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Multiple CNN Kernels  │   ← Kernel sizes (e.g., 3, 5, 7)
                    │ (Conv1D + ReLU + Pool) │
                    └───────────┬────────────┘
                                │
                      ┌────────▼─────────┐
                      │   Concatenation  │
                      └────────┬─────────┘
                               │
                        ┌─────▼─────┐
                        │  BiLSTM   │   ← Captures sequence context
                        └─────┬─────┘
                              │
                        ┌────▼─────┐
                        │ Attention│   ← Learns weighted features
                        └────┬─────┘
                             │
                       ┌────▼────┐
                       │ Softmax │   ← Multi-class classification
                       └─────────┘



![**Multi Kernel**](<Pasted image (2).png>)

![**Layers flow**](<Pasted image.png>)

##  Internal Architecture Details

| Component              | Description |
|------------------------|-------------|
| **Embedding Layer**    | Transforms input tokens into dense vectors. |
| **Multi-Kernel CNN**   | Multiple Conv1D layers with different kernel sizes (e.g., 3, 5, 7) to extract n-gram features. |
| **BiLSTM**             | Captures forward and backward sequential dependencies of token-level features. |
| **Attention Layer**    | Learns importance weights over each time step from BiLSTM outputs. |
| **Classifier**         | Fully connected layer followed by Softmax for multi-class output. |

---


## Benefits

- **Captures multiple granularities** of text features
- Improves **generalization** and **robustness**
- Reduces overfitting compared to single kernel-size models
- Inspired by the successful **TextCNN** architecture (Yoon Kim, 2014)

---

##  How It Works in Your Model

```python
self.convs = nn.ModuleList([
    nn.Conv1d(embed_dim, 128, kernel_size=k, padding=k//2) for k in [2, 3, 4]
])
```

- Each `Conv1d` operates in parallel over the embedding output.
- Outputs are concatenated:
```python
x = torch.cat([F.relu(conv(x)) for conv in self.convs], dim=1)
```

---

##  Result

Enables the model to:
- Detect **local patterns** and **global context**
- Improve accuracy across **all URL categories**, especially with attention + BiLSTM





## Use Cases

- Phishing URL classification (e.g., PHIUSIIL dataset)
- HTTP/HTTPS sequence anomaly detection (e.g., CSIC-2010, CIC datasets)
- Android or Windows malware command pattern classification
- Protocol payload classification in NIDS (Network Intrusion Detection Systems)
- SMS/email text intent classification

---

##  Features Learned Implicitly

- Local n-gram patterns (via multi-kernel CNN)
- Global token sequence semantics (via BiLSTM)
- Contextual importance (via attention mechanism)
- Robustness to input length variation
- Sequential anomaly cues across domains (URL, payload, commands)

---

##  Why It Is Different

| Feature                     | Traditional CNN | Transformer | This Model |
|-----------------------------|-----------------|-------------|-------------|
| Multi-kernel feature capture| ❌              | ❌          | ✅          |
| Sequential memory           | ❌              | ✅ (self-attn) | ✅ (BiLSTM) |
| Local context sensitivity   | ✅              | ❌          | ✅          |
| Lightweight for edge        | ✅              | ❌          | ✅          |
| Attention-based weighting   | ❌              | ✅          | ✅          |

This model provides a **hybrid lightweight alternative** to transformer-based architectures while preserving sequential learning and attention, making it ideal for edge devices or time-constrained cybersecurity inference.

---

##  Pros and Cons

###  Pros
- Captures multi-scale patterns using CNN filters
- Learns long-term dependencies via BiLSTM
- Attention improves interpretability
- Works well on short/medium-length sequences
- Suitable for real-time and edge deployment

### Cons
- Not as parallelizable as Transformers
- Slightly heavier than pure CNN models
- Needs careful tuning of kernel sizes and hidden units
- May require padding/truncation for variable-length inputs

---

## Suggested Hyperparameters

| Parameter        | Recommended Value |
|------------------|-------------------|
| Embedding dim    | 100 or 300         |
| CNN filters      | 64 or 128          |
| Kernel sizes     | [3, 5, 7]          |
| BiLSTM hidden dim| 128 or 256         |
| Dropout          | 0.3 to 0.5         |
| Optimizer        | Adam or AdamW      |

---

## Suggested Datasets

###  PHIUSIIL Phishing URL Dataset
- UCI Link: https://data.mendeley.com/datasets/vfszbj9b36/1
          : https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset?utm_source=chatgpt.com
- Target: Multi-class (Phishing, Malware, Defacement, Benign)


 
---

## Deployment Tip

This model can be converted using TorchScript or ONNX for deployment on edge devices (e.g., NVIDIA Jetson, Raspberry Pi with Coral TPU, or mobile inference).

---

## Citation Suggestion

If you adapt or extend this model, consider citing the following for reference inspiration:
