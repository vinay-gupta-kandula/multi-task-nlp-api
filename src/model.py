import torch
from torch import nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_sentiment=2, num_ner=9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment)
        self.ner_head = nn.Linear(hidden_size, num_ner)
        self.qa_head = nn.Linear(hidden_size, 2)
        
    def forward(self, input_ids, attention_mask, task="sentiment"):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if task == "sentiment":
            pooled = outputs.last_hidden_state[:, 0, :] # CLS token output
            return self.sentiment_head(pooled)
            
        elif task == "ner":
            return self.ner_head(outputs.last_hidden_state)
            
        elif task == "qa":
            logits = self.qa_head(outputs.last_hidden_state)
            start_logits, end_logits = logits.split(1, dim=-1)
            return start_logits.squeeze(-1), end_logits.squeeze(-1)
