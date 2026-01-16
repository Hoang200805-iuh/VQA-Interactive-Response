from PIL import Image
import os
import torch
# Core module for VQA processing
class VQADescriptiveDataset(torch.utils.data.Dataset):
    def __init__(self, data, processor, root_dir):
        self.data = data
        self.processor = processor
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = os.path.join(self.root_dir, sample["image"])
        image = Image.open(image_path).convert("RGB")

        question = sample["question"]
        answer = sample["answer"]

        # Encode image + question
        inputs = self.processor(image, question, return_tensors="pt", padding="max_length", truncation=True)

        # Encode answer + apply padding/truncation
        labels = self.processor.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64  # hoặc dài hơn nếu cần
        ).input_ids

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }
