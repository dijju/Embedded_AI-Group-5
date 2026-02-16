import torch
import os
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import (
    Compose, Normalize, RandomResizedCrop, RandomHorizontalFlip, 
    RandomRotation, ColorJitter, ToTensor, Resize, CenterCrop
)

# 1. SETUP & CONFIGURATION
# -----------------------
# Swinv2 is excellent for plants due to "shifted windows" seeing local lesions better
MODEL_CHECKPOINT = "microsoft/swinv2-tiny-patch4-window8-256" 
BATCH_SIZE = 32
EPOCHS = 5 # Start small, 11k images takes time
LEARNING_RATE = 5e-5 # Low LR is critical for fine-tuning to not break pre-trained weights

# 2. LOAD DATASET
# -----------------------
# Assumes you have a folder named 'data' with subfolders for each disease
print("Loading dataset...")
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
dataset = load_dataset("imagefolder", data_dir=data_dir)

dataset = dataset["train"].train_test_split(test_size=0.2) 

# Extract labels
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# 3. PREPROCESSING & AUGMENTATION
# -----------------------
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

# Normalize using the model's specific mean/std
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

# Augmentation: Critical for plant disease to handle lighting/orientation changes
_train_transforms = Compose([
    RandomResizedCrop(image_processor.size["height"]),
    RandomHorizontalFlip(),
    RandomRotation(degrees=15), # Rotates leaves slightly
    ColorJitter(brightness=0.2, contrast=0.2), # Simulates different sun conditions
    ToTensor(),
    normalize,
])

_val_transforms = Compose([
    Resize(image_processor.size["height"]),
    CenterCrop(image_processor.size["height"]),
    ToTensor(),
    normalize,
])

def train_transforms(examples):
    examples["pixel_values"] = [_train_transforms(image.convert("RGB")) for image in examples["image"]]
    del examples["image"]
    return examples

def val_transforms(examples):
    examples["pixel_values"] = [_val_transforms(image.convert("RGB")) for image in examples["image"]]
    del examples["image"]
    return examples

# Apply transforms
train_ds = dataset["train"].with_transform(train_transforms)
val_ds = dataset["test"].with_transform(val_transforms)
# 4. MODEL SETUP
# -----------------------
model = AutoModelForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True # Necessary when changing number of classes (e.g. 1000 -> 10)
)

# 5. TRAINING ARGUMENTS
# -----------------------
args = TrainingArguments(
    f"tomato-disease-swin-finetuned",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4, # Simulates larger batch size if you have low VRAM
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 6. METRICS
# -----------------------
from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(labels, predictions))

# 7. TRAIN
# -----------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=image_processor,
    compute_metrics=compute_metrics,
    data_collator=None, # Default collator works for most vision tasks
)

print("Starting training...")
trainer.train()

# 8. SAVE FINAL MODEL
trainer.save_model("./final_tomato_model")
print("Model saved to ./final_tomato_model")

# 9. EXPORT TO ONNX
# -----------------------
print("Exporting model to ONNX format...")
model.eval()
dummy_input = torch.randn(1, 3, image_processor.size["height"], image_processor.size["height"])
onnx_path = "./final_tomato_model/model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['pixel_values'],
    output_names=['logits'],
    dynamic_axes={
        'pixel_values': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }
)
print(f"ONNX model saved to {onnx_path}")