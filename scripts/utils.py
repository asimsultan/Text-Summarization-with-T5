import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_data(examples, tokenizer, max_input_length, max_output_length):
    inputs = tokenizer(
        examples['article'], max_length=max_input_length, truncation=True, padding='max_length', return_tensors='pt'
    )
    targets = tokenizer(
        examples['highlights'], max_length=max_output_length, truncation=True, padding='max_length', return_tensors='pt'
    )

    inputs['labels'] = targets['input_ids']
    return inputs


def create_data_loader(dataset, batch_size, sampler):
    data_sampler = sampler(dataset)
    data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size)
    return data_loader
