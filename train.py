import os
import json
import gc
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from dataset_pytorch import VQADescriptiveDataset 
# Description
def main():
    # Giải phóng GPU
    gc.collect()
    torch.cuda.empty_cache()

    # Load processor + model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    processor.save_pretrained("./blip-vqa-descriptive/processor")  # Lưu processor

    # Load dữ liệu JSON
    with open("MyData/Flickr8k_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Chia tập train/val
    train_data, eval_data = train_test_split(data, test_size=0.3, random_state=42)

    # Tạo dataset
    root_dir = "Flickr8k/Images"
    train_dataset = VQADescriptiveDataset(data=train_data, processor=processor, root_dir=root_dir)
    eval_dataset = VQADescriptiveDataset(data=eval_data, processor=processor, root_dir=root_dir)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./blip-vqa-descriptive/final-model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,  # tăng lên để mô hình học tốt hơn
        learning_rate=5e-5,
        save_strategy="steps",
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        fp16=False,
        dataloader_num_workers=0,  # Tránh bug trên Windows, thử đặt 0 nếu 1 gây lỗi
        report_to="none"
    )

    # Collator
    data_collator = DataCollatorWithPadding(tokenizer=processor.tokenizer, return_tensors="pt")

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        data_collator=data_collator
    )

    # Train
    trainer.train()
    trainer.save_model("./blip-vqa-descriptive/final-model")
    processor.save_pretrained("./blip-vqa-descriptive/processor")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
