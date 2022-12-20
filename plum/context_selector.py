import string

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize

stopwords = list(nltk.corpus.stopwords.words("english"))
punctuations = string.punctuation


class ContextSelector(nn.Module):
    def __init__(
            self,
            selector_model_path,
            selector_mode="whole-word",
            device="cuda"
    ):
        super(ContextSelector, self).__init__()
        self.device = device
        self.selector_model_path = selector_model_path
        self.selector_mode = selector_mode

        self.selector_model = AutoModel.from_pretrained(self.selector_model_path).to(self.device)
        self.selector_tokenizer = AutoTokenizer.from_pretrained(self.selector_model_path, add_prefix_space=True)
        self.c = nn.Parameter(torch.rand(self.selector_model.config.vocab_size)).to(self.device)
        self.W = nn.Linear(self.selector_model.config.hidden_size, 1).to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def filter_text(self, passage):
        passage = passage.translate(str.maketrans("", "", punctuations))
        passage = word_tokenize(passage)
        passage = [word for word in passage if word not in stopwords]
        return " ".join(passage)

    def forward_whole_word_selection(self, passages, preprocess=False, threshold=0.5):
        if preprocess:
            passages = list(map(self.filter_text, passages))

        embedding_matrix = self.selector_model.get_input_embeddings()._parameters['weight'].to(self.device)
        context_keywords = list()
        loss_cs = list()
        for passage in passages:
            token_ids_map = {
                word: self.selector_tokenizer.encode(
                    word,
                    add_special_tokens=False
                ) for word in passage.split()
            }
            input_ids = list(token_ids_map.values())
            input_ids = np.asarray([item for sublist in input_ids for item in sublist])

            word_embedding_matrix = embedding_matrix[input_ids, :].to(self.device)
            embedding_dim = word_embedding_matrix.shape[1]
            embedding_dim_root = embedding_dim ** 0.5
            self_attention_matrix = (
                    torch.matmul(word_embedding_matrix, word_embedding_matrix.T) / embedding_dim_root
            )
            self_attention_matrix = F.softmax(self_attention_matrix, dim=0)
            embedding_attention_matrix = torch.matmul(
                self_attention_matrix, word_embedding_matrix
            ).to(self.device)

            q = self.sigmoid(self.W(embedding_attention_matrix).flatten())
            c_subset = self.c[torch.LongTensor(input_ids)]
            input_ids_selection = (c_subset >= threshold).tolist()

            selected_input_ids = input_ids[input_ids_selection]
            current_context_keywords = list()
            for selected_input_id in selected_input_ids:
                for token in token_ids_map:
                    if selected_input_id in token_ids_map[token] and token not in current_context_keywords:
                        current_context_keywords.append(token)
                        break
                # current_context_keywords.append(self.selector_tokenizer.decode(selected_input_id))
            # current_context_keywords = list(map(str.lower, current_context_keywords))
            context_keywords.append(current_context_keywords)
            loss = F.kl_div(q.log(), c_subset)
            loss_cs.append(loss)
        loss_cs = torch.stack(loss_cs).mean()
        return loss_cs, context_keywords

    def forward_token_selection(self, passages, preprocess=False, threshold=0.5):
        if preprocess:
            passages = list(map(self.filter_text, passages))

        embedding_matrix = self.selector_model.get_input_embeddings()._parameters['weight'].to(self.device)
        context_keywords = list()
        loss_cs = list()
        for passage in passages:
            input_ids = self.selector_tokenizer.encode(passage, add_special_tokens=False)
            tokens = [self.selector_tokenizer.decode(w) for w in input_ids]

            word_embedding_matrix = embedding_matrix[input_ids, :].to(self.device)
            embedding_dim = word_embedding_matrix.shape[1]
            embedding_dim_root = embedding_dim ** 0.5
            self_attention_matrix = (
                    torch.matmul(word_embedding_matrix, word_embedding_matrix.T) / embedding_dim_root
            )
            self_attention_matrix = F.softmax(self_attention_matrix, dim=0)
            embedding_attention_matrix = torch.matmul(
                self_attention_matrix, word_embedding_matrix
            ).to(self.device)

            q = self.sigmoid(self.W(embedding_attention_matrix).flatten())
            c_subset = self.c[torch.LongTensor(input_ids)]
            input_ids_selection = (c_subset >= threshold).tolist()

            selected_input_ids = np.asarray(input_ids)[input_ids_selection]
            current_context_keywords = list()
            for selected_input_id in selected_input_ids:
                current_context_keywords.append(self.selector_tokenizer.decode(selected_input_id))
            # current_context_keywords = list(map(str.lower, current_context_keywords))
            context_keywords.append(current_context_keywords)
            loss = F.kl_div(q.log(), c_subset)
            loss_cs.append(loss)

        loss_cs = torch.stack(loss_cs).mean()
        return loss_cs, context_keywords

    def forward(self, passages, preprocess=False, threshold=0.5):
        if self.selector_mode == "whole-word":
            return self.forward_whole_word_selection(passages, preprocess, threshold)
        elif self.selector_mode == "token":
            return self.forward_token_selection(passages, preprocess, threshold)
        else:
            raise ValueError(f"Unknown mode: {self.selector_mode}")
