import pandas as pd
import json
import random
from transformers import pipeline

# Đường dẫn dữ liệu
caption_file = "Flickr8k/captions.txt"
image_folder = "Flickr8k/Images/"
output_json = "Flickr8k.json"

# Load caption từ file CSV (có header: image,caption)
df = pd.read_csv(caption_file)

# Loại bỏ caption trùng nhau (nếu có)
df = df.drop_duplicates()

# Shuffle và lấy ngẫu nhiên 200 dòng
#df_sample = df.sample(n=200, random_state=42)
df_sample = df.copy()

# Khởi tạo mô hình sinh câu hỏi (lightweight)
qg = pipeline("text2text-generation", model="valhalla/t5-small-qg-prepend")

# Tạo dữ liệu dạng VQA
data = []
for idx, row in df_sample.iterrows():
    image_path = f"{image_folder}{row['image']}"
    caption = row['caption'].strip()
    prompt = f"generate question: {caption}"

    try:
        question = qg(prompt, max_length=64)[0]["generated_text"]
    except Exception:
        question = "What is happening in the image?"

    data.append({
        "image": image_path,
        "question": question,
        "answer": caption[0].upper() + caption[1:]
    })

# Ghi ra file JSON
with open(output_json, "w") as f:
    json.dump(data, f, indent=2)

print(f"Đã tạo {len(data)} mẫu dữ liệu VQA tại: {output_json}")

#Fix images

with open("Flickr8k.json", "r") as f:
    data = json.load(f)

for item in data:
    item["image"] = item["image"].split("/")[-1]  # chỉ lấy tên file ảnh

with open("Flickr8k_data.json", "w") as f:
    json.dump(data, f, indent=2)

print("Đã sửa đường dẫn ảnh, lưu vào Flickr8k_data.json")
