import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import json
from sklearn.model_selection import train_test_split
import os
# Description
# Thiết lập thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dữ liệu và chia tập
with open("MyData/Flickr8k_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

_, eval_data = train_test_split(data, test_size=0.3, random_state=42)

# Load mô hình đã fine-tune
processor = BlipProcessor.from_pretrained("blip-vqa-descriptive/processor")
model = BlipForQuestionAnswering.from_pretrained("blip-vqa-descriptive/final-model")

# ----- Hàm tiện ích test một mẫu ----- #
def test_sample(sample, image_root="Flickr8k/Images"):
    image_path = os.path.join(image_root, sample["image"])
    image = Image.open(image_path).convert("RGB")
    question = sample["question"]
    ground_truth = sample["answer"]

    # Trả lời không prompt và có prompt
    inputs = processor(image, question, return_tensors="pt").to(model.device)
    out = model.generate(**inputs)
    answer_no_prompt = processor.decode(out[0], skip_special_tokens=True)
    # In kết quả
    print("=" * 60)
    print(f" Image: {sample['image']}")
    print(f" Question       : {question}")
    print(f" Ground truth   : {ground_truth}")
    print(f" No Prompt Ans  : {answer_no_prompt}")


# ----- Test ngẫu nhiên vài mẫu ----- #
import random
samples_to_test = random.sample(eval_data, 5)

for sample in samples_to_test:
    test_sample(sample)
