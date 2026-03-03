import torch
from torch import nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_sentiment=2, num_ner=9):
        super().__init__()
        # Shared transformer encoder used as the backbone for all three tasks
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Task-specific linear layers (heads)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment)
        self.ner_head = nn.Linear(hidden_size, num_ner)
        self.qa_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        """
        Performs a unified forward pass that returns logits for all tasks simultaneously.
        This structure is critical for exporting the full multi-task capability into a 
        single ONNX graph for high-performance production inference.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Sentiment Analysis: Extracts the [CLS] token (index 0) for sequence classification
        pooled = sequence_output[:, 0]
        sentiment_logits = self.sentiment_head(pooled)

        # Named Entity Recognition: Token-level classification for the entire sequence
        ner_logits = self.ner_head(sequence_output)

        # Question Answering: Predicts the start and end positions within the sequence
        qa_logits = self.qa_head(sequence_output)
        start_logits = qa_logits[..., 0]
        end_logits = qa_logits[..., 1]

        return sentiment_logits, ner_logits, start_logits, end_logits