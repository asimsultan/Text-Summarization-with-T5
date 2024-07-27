import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_scheduler
from datasets import load_dataset
from utils import get_device, tokenize_data, create_data_loader

# Parameters
model_name = 't5-small'
max_input_length = 512
max_output_length = 150
batch_size = 8
epochs = 3
learning_rate = 5e-5

# Load Dataset
dataset = load_dataset('cnn_dailymail', '3.0.0')
train_dataset = dataset['train']
validation_dataset = dataset['validation']

# Tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Tokenize Data
train_dataset = train_dataset.map(lambda x: tokenize_data(x, tokenizer, max_input_length, max_output_length), batched=True)
validation_dataset = validation_dataset.map(lambda x: tokenize_data(x, tokenizer, max_input_length, max_output_length), batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# DataLoader
train_loader = create_data_loader(train_dataset, batch_size, RandomSampler)
validation_loader = create_data_loader(validation_dataset, batch_size, SequentialSampler)

# Model
device = get_device()
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = len(train_loader) * epochs
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Training Function
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0

    for batch in data_loader:
        b_input_ids = batch['input_ids'].squeeze(1).to(device)
        b_attention_mask = batch['attention_mask'].squeeze(1).to(device)
        b_labels = batch['labels'].squeeze(1).to(device)

        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_attention_mask,
            labels=b_labels
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

# Training Loop
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Train Loss: {train_loss}')

# Save Model
model_dir = './models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
