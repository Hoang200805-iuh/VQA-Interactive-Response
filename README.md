# Improved Interactivity and Automated Response for Visual Question Answering
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](./VQA_Research_Paper.pdf)[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-BLIP%20%2B%20T5-green)](https://huggingface.co/Salesforce/blip-vqa-base)

[cite_start]This repository contains the implementation of the research paper **"Improved interactivity and automated response for visual question answering"**[cite: 296]  [![Link Paper](https://img.shields.io/badge/Paper-PDF-red)](./VQA_Research_Paper.pdf)

The project proposes a two-stage VQA framework:
1.  [cite_start]**Question Generation:** Using **T5TP3** to automatically generate diverse questions from image captions[cite: 316].
2.  [cite_start]**Visual Question Answering:** Fine-tuning the **BLIP** model with **Prompt Engineering** to generate descriptive, context-aware answers[cite: 316, 504].

## 🏗️ System Architecture

![Architecture](assets/architecture.png.jpg)
*Figure 1: Overview of the proposed model architecture[cite: 424].*

The system processes data through the following pipeline:
1.  **Input:** Images and Captions from the Flickr8k dataset.
2.  **T5TP3 Model:** Generates question-answer pairs from captions.
3.  **BLIP Fine-tuning:** The model is trained using the generated pairs with a **Frozen Vision Encoder** to save resources.
4.  **Inference:** Uses a specific prompt template to guide the model: *"You are a descriptive VQA assistant. Question: {q}"*.

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone [https://github.com/Hoang200805-iuh/VQA-Interactive-Response.git](https://github.com/your-username/VQA-Interactive-Response)
cd VQA-Interactive-Response.
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Data Preparation
```bash
# This project utilizes the Flickr8k dataset.
# Place them in the data/ folder or update the paths in src/dataset.py to point to your local image directory.
```

## Usage / How to Run
+ Step 1: Automatic Question Generation
Use the T5TP3 model to generate diverse question-answer pairs from the Flickr8k captions. This augments the dataset for better training.
```bash
python src/question_gen.py
```

+ Step 2: Fine-tune BLIP Model
Train the model using the generated dataset. We utilize Gradient Accumulation and FP16 mixed precision to optimize performance on limited GPU memory.
```bash
python train.py --epochs 10 --batch_size 16
```

+ Step 3: Evaluation
Evaluate the fine-tuned model using BLEU and ROUGE-L metrics on the test set.
```bash
python main.py --image_path "assets/demo_image.jpg" --question "What are the people doing?"
```

# 📊 Results & Performance

### Our approach outperforms baseline models like BLIP-2 and InstructBLIP in generating descriptive, context-aware answers.


| Method | BLEU-1 | BLEU-4 | ROUGE-L |
| :--- | :---: | :---: | :---: |
| BLIP-2 | 0.22 | 0.14 | 0.42 |
| InstructBLIP | 0.08 | 0.04 | 0.15 |
| **Ours (Fine-tuned)** | **0.32** | **0.19** | **0.52** |.