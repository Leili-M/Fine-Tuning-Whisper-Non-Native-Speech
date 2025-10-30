## Fine-Tuning Whisper for Non-Native and Child Speech Recognition
### Overview

This project focuses on fine-tuning OpenAI’s Whisper model to improve speech recognition performance on English audio from non-native speakers and children, using the Speechocean762
 dataset.
While Whisper is robust on general English speech, it underperforms on accented, non-fluent, or child-like voices due to pronunciation variability and prosodic differences.
The main objective was to adapt a lightweight version of Whisper (whisper-tiny.en) to this domain through fine-tuning, evaluate it both quantitatively (via WER) and qualitatively (via transcript analysis), and investigate how the model behaves across fluency levels.

---
### Project Objectives

Adapt Whisper to handle non-native and child speech more effectively.

Build a reproducible fine-tuning pipeline with clear data, metric, and training stages.

Measure WER improvement and interpret the relationship between loss and WER.

Analyze qualitative changes in the model’s transcription behavior after fine-tuning.

---

### Why Whisper?

Pretrained foundation: Whisper is trained on ~680,000 hours of multilingual, multitask data.

Lightweight and efficient: The tiny.en variant has only ~39M parameters, making it ideal for limited hardware and fast iteration.

Strong English representation: The .en model focuses on English-only data, providing a good baseline for adaptation.

---
### Dataset: Speechocean762

Domain: English speech from non-native speakers (adults + children).

Characteristics: High phonetic and prosodic diversity; balanced gender, age, and fluency levels.

---

### Methodology
 #### Data Preparation

Converted raw audio signals into log-Mel spectrograms using Whisper’s feature extractor.

Cleaned and normalized transcripts (lowercase, punctuation removed).

Tokenized text to numerical label IDs for supervised learning.

This ensured data compatibility with Whisper’s encoder–decoder architecture.
####  Model Setup

A pretrained Whisper Tiny (English) checkpoint was loaded, and certain parameters were modified for fine-tuning stability:

Disabled forced decoder tokens to allow natural generation (forced_decoder_ids = None).

Turned off caching (use_cache = False) due to gradient checkpointing conflicts.

Suppressed irrelevant tokens to reduce decoding overhead.

These adjustments maintained training efficiency and prevented redundant outputs during decoding.

#### Training Configuration

Fine-tuning was performed with Hugging Face Transformers using the Seq2SeqTrainer.

Key parameters:

Learning rate: 1e-5

Batch size: 8 per device

Warmup steps: 500

Max steps: 600

Loss function: Cross-Entropy

Evaluation metric: WER (Word Error Rate)

Why cross-entropy instead of WER?
Because WER is a discrete metric, it cannot provide gradients for optimization.
Cross-entropy, being continuous and differentiable, allows stable weight updates through backpropagation — leading to reliable convergence.
#### Training Process

The model was trained for 600 steps, with both training and validation loss monitored periodically.

WER was computed at each evaluation step to measure recognition accuracy.

TensorBoard was used for real-time visualization of the training curves.

The training completed smoothly with no sign of instability or overfitting.

---

### Results
Quantitative Summary
| Metric                | Baseline (`whisper-tiny.en`) | Fine-tuned Model         |
| --------------------- | ---------------------------- | ------------------------ |
| **Training Steps**    | –                            | **600**                  |
| **Validation Loss**   | 1.66 → **0.49**              |  Stable convergence     |
| **WER (%)**           | 65.4 → **21.7**              | 67% relative reduction |
| **Training Duration** | –                            | ~1.5 hours on T4 GPU     |


Interpretation:

The model rapidly reduced loss during the first 150 steps and gradually converged afterward.

The lowest WER (~21.3%) was achieved around step 600.

Training and validation losses decreased in parallel, suggesting no overfitting and good generalization.

----

### Qualitative Analysis

The improvement is most apparent in handling incomplete or mispronounced phrases.
Below are representative examples from the test set:
| Type                     | Reference                            | Baseline Prediction | Fine-tuned Prediction                |
| ------------------------ | ------------------------------------ | ------------------- | ------------------------------------ |
| Normal sentence          | he likes the famous city sydney      | my like             | he likes the famous city sydney      |
| Non-native pronunciation | we eat less meat                     | we less meat        | we eat less meat                     |
| Longer utterance         | i will bring the yellow bag tomorrow | i bring the bag     | i will bring the yellow bag tomorrow |

Observations:

Fine-tuning improved sentence completeness and grammatical accuracy.

The model handled child-like articulation better, reducing omission and insertion errors.

The predictions became semantically closer to human references.

---

### Fluency-Based Evaluation

WER was also analyzed relative to speaker fluency scores.

#### Findings:

Fine-tuning led to a substantial WER reduction for low-fluency utterances — where the baseline made the most mistakes.

Even high-fluency speakers saw moderate improvements (a few percentage points).

The model learned to better generalize across varying articulation rates and accent intensities.

---
### Discussion
| Aspect             | Insight                                                                       |
| ------------------ | ----------------------------------------------------------------------------- |
| **Loss Function**  | Cross-entropy ensures smooth optimization; WER is non-differentiable.         |
| **Model Behavior** | WER and loss decreased consistently, reflecting effective learning.           |
| **Generalization** | Validation performance matched training trends — no overfitting.              |
| **Error Sources**  | Residual errors stem from homophones, background noise, or very short clips.  |
| **Efficiency**     | Whisper-Tiny adapted well using only 600 steps and minimal compute resources. |

---

### Conclusion

Fine-tuning Whisper on the Speechocean762 dataset led to a remarkable improvement in ASR accuracy, especially for non-native and child speech.
Even with only 600 training steps, the lightweight model achieved a relative 67% reduction in word error rate, proving that small-scale fine-tuning can yield large domain-specific gains.

#### This experiment demonstrates:
The potential of small ASR models for targeted domain adaptation.
The correlation between loss reduction and improved transcription accuracy.
The effectiveness of Whisper’s pretrained architecture in capturing new speech patterns with limited data.
