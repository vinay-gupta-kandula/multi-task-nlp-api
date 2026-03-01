import json
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        with open(filepath, 'r') as f:
            # keep only a small subset of examples to avoid OOM in constrained environments
            self.data = json.load(f)[:100]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

class NERDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        with open(filepath, 'r') as f:
            self.data = json.load(f)[:500]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['tokens'],
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = []
        word_ids = encoding.word_ids()
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            else:
                try:
                    labels.append(item['tags'][word_idx])
                except IndexError:
                    labels.append(-100)
                
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

class QADataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=384):
        with open(filepath, 'r') as f:
            self.data = json.load(f)[:500]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answers = item['answers']

        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            padding='max_length',
            truncation="only_second",
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        offset_mapping = encoding.pop('offset_mapping').squeeze(0).tolist()
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        start_positions = 0
        end_positions = 0

        if len(answers['answer_start']) > 0:
            start_char = answers['answer_start'][0]
            try:
                end_char = start_char + len(answers['text'][0])
            except IndexError:
                end_char = start_char

            sequence_ids = encoding.sequence_ids(0)
            
            # Find start and end token indices
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if context_end >= context_start:
                if offset_mapping[context_start][0] <= start_char and offset_mapping[context_end][1] >= end_char:
                    idx = context_start
                    while idx <= context_end and offset_mapping[idx][0] <= start_char:
                        idx += 1
                    start_positions = idx - 1

                    idx = context_end
                    while idx >= context_start and offset_mapping[idx][1] >= end_char:
                        idx -= 1
                    end_positions = idx + 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': torch.tensor(start_positions, dtype=torch.long),
            'end_positions': torch.tensor(end_positions, dtype=torch.long)
        }
