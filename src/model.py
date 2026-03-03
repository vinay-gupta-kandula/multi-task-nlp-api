import torch
from torch import nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_sentiment=2, num_ner=9):
        """
        Initializes a Multi-Task model with a shared Transformer backbone.
        
        Args:
            model_name (str): The name of the pre-trained transformer model.
            num_sentiment (int): Number of classes for sentiment analysis (default 2).
            num_ner (int): Number of labels for NER (standard CoNLL-2003 is 9).
        """
        super().__init__()
        # Shared transformer encoder used as the backbone for all three tasks
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Task-specific linear layers (heads)
        # Sentiment: Binary classification (positive/negative)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment)
        
        # NER: Token-level classification (label per token)
        self.ner_head = nn.Linear(hidden_size, num_ner)
        
        # QA: Predicting start and end span indices (2 outputs)
        self.qa_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        """
        Performs a unified forward pass returning logits for all tasks.
        This structure is required for exporting to a single ONNX graph.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 1. Sentiment Analysis: Uses the [CLS] token (index 0) for sequence classification
        pooled_output = sequence_output[:, 0]
        sentiment_logits = self.sentiment_head(pooled_output)

        # 2. Named Entity Recognition: Provides a logit per token in the sequence
        ner_logits = self.ner_head(sequence_output)

        # 3. Question Answering: Outputs start and end logits for span extraction
        qa_logits = self.qa_head(sequence_output)
        start_logits = qa_logits[..., 0]
        end_logits = qa_logits[..., 1]

        return sentiment_logits, ner_logits, start_logits, end_logits