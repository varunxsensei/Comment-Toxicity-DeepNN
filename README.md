
# 🧠 Toxic Comment Classification with LSTM & Gradio

This is a personal deep learning project built to classify toxic comments using a custom NLP pipeline. The model is trained on the **Jigsaw Toxic Comment Classification** dataset using TensorFlow and an LSTM-based architecture, and deployed with a Gradio interface for interactive testing.

---

## 📌 Project Highlights

- 🔄 **Custom NLP Pipeline** using `TextVectorization` for tokenization and sequence preparation.
- 📊 Multi-label classification across 6 categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.
- 🧠 **Bidirectional LSTM** for capturing contextual word dependencies in both directions.
- 🧱 Layered dense blocks for deep feature extraction.
- 🎛️ Gradio-based interface for real-time comment scoring and experimentation.

---

## 🧰 Tech Stack

- Python 3.9+
- TensorFlow 2.x
- Pandas, NumPy, Matplotlib
- Gradio
- Jigsaw Dataset (Kaggle)

---

## 🏗️ Model Architecture

```text
Embedding Layer (200k vocab size)
↓
Bidirectional LSTM (32 units, tanh)
↓
Dense(128, relu)
↓
Dense(256, relu)
↓
Dense(128, relu)
↓
Dense(6, sigmoid)
```

---

## 📈 Performance

Evaluated on held-out test set using:
- Precision
- Recall
- Categorical Accuracy

Custom evaluation loop used for better metric tracking across batches.

---

## 🧪 Example Usage

```python
score_comment("You are a piece of garbage and deserve nothing!")
# Output:
# toxic: True
# severe_toxic: False
# obscene: True
# threat: False
# insult: True
# identity_hate: False
```

---

## 🖥️ Gradio Interface

Once model is trained and saved (`toxicity.h5`), launch:

```python
interface.launch(share=True)
```

> Enter a comment and get toxicity labels in real-time.

---

## 📌 To-Do / Future Plans

- Improve model with attention mechanisms or Transformer-based embeddings.
- Add class-specific thresholds.
- Host permanently with Hugging Face Spaces.
- Train on larger datasets for generalization.

---

## 🙌 Acknowledgements

- [Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
- TensorFlow/Keras for deep learning components
- Gradio for easy model deployment

---

## 📬 Connect with Me

If you’re working in NLP or AI, let’s connect on [LinkedIn]([https://www.linkedin.com/in/varunxsensei](https://www.linkedin.com/in/varun-saxena-5678b5340/)) and collaborate!
