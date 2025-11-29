# Mercor AI Text Detection 🕵️‍♂️🤖

![Badge](https://img.shields.io/badge/Python-3.9-blue)
![Badge](https://img.shields.io/badge/Task-Binary_Classification-orange)

This repository contains my solution for the [Mercor AI Text Detection Competition](https://kaggle.com/competitions/mercor-ai-detection). The goal is to detect whether a writing sample is authentic human writing or generated/assisted by AI ("cheating").

## 📌 Project Overview
With the rise of LLMs, distinguishing between human and machine-generated text is a critical challenge. This project implements a deep learning approach to classify text samples.

* **Input:** `topic` (string), `answer` (text sample).
* **Output:** Probability of `is_cheating` (0 = Human, 1 = AI/Cheating).
* **Evaluation Metric:** ROC-AUC.

## 🧠 Methodology / Approach
