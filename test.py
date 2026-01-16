import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import json
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
import nltk
import os
from tqdm import tqdm
# Description
# Download NLTK data
nltk.download("punkt")

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dữ liệu
with open("MyData/Flickr8k_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

_, eval_data = train_test_split(data, test_size=0.3, random_state=42)

# Load mô hình và processor
model = BlipForQuestionAnswering.from_pretrained("blip-vqa-descriptive/final-model").to(device)
processor = BlipProcessor.from_pretrained("blip-vqa-descriptive/processor")

# Hàm tính BLEU
def compute_bleu_scores(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    return {
        "BLEU-1": sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothie),
        "BLEU-2": sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
        "BLEU-3": sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie),
        "BLEU-4": sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie),
    }

# Khởi tạo biến tổng
metrics_total = {
    "BLEU-1": 0, "BLEU-2": 0, "BLEU-3": 0, "BLEU-4": 0,
    "METEOR": 0, "ROUGE-L": 0
}
num_samples = 100  # Số mẫu đánh giá

# Bộ tính ROUGE
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Đánh giá
for sample in tqdm(eval_data[:num_samples]):
    image_path = os.path.join("Flickr8k/Images", sample["image"])
    image = Image.open(image_path).convert("RGB")
    question = sample["question"]
    ground_truth = sample["answer"]

    # Inference
    inputs = processor(image, question, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    pred = processor.decode(out[0], skip_special_tokens=True)

    # Token hóa
    reference = word_tokenize(ground_truth.lower())
    hypothesis = word_tokenize(pred.lower())

    # BLEU
    bleu_scores = compute_bleu_scores(reference, hypothesis)
    for k in bleu_scores:
        metrics_total[k] += bleu_scores[k]

    # METEOR
    gt_tokens = word_tokenize(ground_truth.lower())
    pred_tokens = word_tokenize(pred.lower())
    metrics_total["METEOR"] += single_meteor_score(gt_tokens, pred_tokens)

    # ROUGE-L
    rouge_score = rouge.score(ground_truth.lower(), pred.lower())
    metrics_total["ROUGE-L"] += rouge_score["rougeL"].fmeasure

# Trung bình
print("\nEvaluation Results on Eval Set:")
for metric, total in metrics_total.items():
    print(f"  {metric}: {total / num_samples:.4f}")
