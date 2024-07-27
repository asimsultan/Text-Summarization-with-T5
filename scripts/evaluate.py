import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from utils import get_device, tokenize_data, create_data_loader
from datasets import load_metric

# Parameters
model_dir = './models'
max_input_length = 512
max_output_length = 150
batch_size = 8

# Load Model and Tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = T5Tokenizer.from_pretrained(model_dir)

# Device
device = get_device()
model.to(device)

# Load Dataset
dataset = load_dataset('cnn_dailymail', '3.0.0')
validation_dataset = dataset['validation']

# Tokenize Data
validation_dataset = validation_dataset.map(lambda x: tokenize_data(x, tokenizer, max_input_length, max_output_length), batched=True)
validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# DataLoader
validation_loader = create_data_loader(validation_dataset, batch_size, SequentialSampler)

# Evaluation Function
def evaluate(model, data_loader, device, tokenizer):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in data_loader:
            b_input_ids = batch['input_ids'].squeeze(1).to(device)
            b_attention_mask = batch['attention_mask'].squeeze(1).to(device)
            b_labels = batch['labels'].squeeze(1).to(device)

            outputs = model.generate(input_ids=b_input_ids, attention_mask=b_attention_mask, max_length=max_output_length)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(b_labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

    metric = load_metric('rouge')
    results = metric.compute(predictions=predictions, references=references, use_stemmer=True)
    return results

# Evaluate
results = evaluate(model, validation_loader, device, tokenizer)
print(results)

# Save the predictions and references
output_dir = './outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'predictions.txt'), 'w') as f:
    for pred in predictions:
        f.write(pred + '\n')

with open(os.path.join(output_dir, 'references.txt'), 'w') as f:
    for ref in references:
        f.write(ref + '\n')
