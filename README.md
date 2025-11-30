# Mercor AI Text Detection

![Badge](https://img.shields.io/badge/Python-3.9-blue)
![Badge](https://img.shields.io/badge/Task-Binary_Classification-orange)
![Badge](https://img.shields.io/badge/Model-DeBERTa_V3-green)

This repository contains my solution for the [Mercor AI Text Detection Competition](https://kaggle.com/competitions/mercor-ai-detection). The goal is to detect whether a writing sample is authentic human writing or generated/assisted by AI ("cheating").

## Project Overview

With the rise of LLMs, distinguishing between human and machine-generated text is a critical challenge. This project implements a deep learning approach to classify text samples.

* **Input:** `topic` (string), `answer` (text sample).
* **Output:** Probability of `is_cheating` (0 = Human, 1 = AI/Cheating).
* **Evaluation Metric:** ROC-AUC.

## Methodology / Approach

My solution utilizes transfer learning with a Transformer-based architecture. Specifically, I fine-tuned the **DeBERTa V3 (Base)** model, which is known for its superior performance on natural language understanding tasks compared to standard BERT or RoBERTa.

### Key Implementation Details:

1.  **Input Formatting:**
    To provide context to the model, I concatenated the `topic` and the `answer` using a separator token. This allows the model to understand the relationship between the prompt and the response.
    * Format: `topic [SEP] answer`

2.  **Tokenization:**
    Used `DebertaV2Tokenizer` with a maximum sequence length of **512 tokens**. Padding and truncation were applied to handle varying text lengths.

3.  **Model Architecture:**
    * **Base:** `microsoft/deberta-v3-base`
    * **Head:** A linear classification layer on top of the pooled output (`[CLS]` token embedding).
    * **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy) is used for optimization.

## Training Configuration

The model was trained using the following hyperparameters. A very low learning rate was selected to carefully fine-tune the pre-trained weights without catastrophic forgetting.

| Parameter | Value |
| :--- | :--- |
| **Base Model** | `microsoft/deberta-v3-base` |
| **Batch Size** | 8 |
| **Learning Rate** | 5e-7 |
| **Epochs** | 50 |
| **Max Length** | 512 |
| **Optimizer** | AdamW |
| **Device** | GPU (CUDA) |

## Installation

To reproduce the environment, install the required dependencies:

```bash
pip install torch transformers sentencepiece pandas numpy matplotlib tqdm
````

*Note: `sentencepiece` is required for DeBERTa V2/V3 tokenizers.*

## Project Structure

  * `train.csv` / `test.csv`: Data files (Input).
  * `notebook.ipynb` (or `main.py`): Contains data loading, model definition, training loop, and inference.
  * `deberta_v3_fulltrain/`: Directory where artifacts are saved.
      * `final_model.pth`: The saved model weights.
      * `loss_plot.png`: Visualization of the training loss.

## Usage

1.  **Prepare Data:** Ensure `train.csv` and `test.csv` are present in the root directory.
2.  **Train:** Run the script to start fine-tuning. The script will:
      * Load and tokenize data.
      * Train the model for 50 epochs.
      * Save the training loss plot to the output directory.
      * Save the final model weights to `deberta_v3_fulltrain/final_model.pth`.
3.  **Inference:** The script automatically generates predictions on `test.csv` after training is complete.

## Results

  * **Training Loss:** Can be monitored via the generated plot `deberta_v3_fulltrain/loss_plot.png`.
  * **Submission:** The model generates probability scores suitable for ROC-AUC evaluation.
