import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize


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

    def push_to_hub(self, repo_name, commit_message, auth_token):
        self.selector_model.push_to_hub(
            repo_name,
            commit_message=f"Model: {commit_message}",
            use_auth_token=auth_token
        )
        self.selector_tokenizer.push_to_hub(
            repo_name,
            commit_message=f"Tokenizer: {commit_message}",
            use_auth_token=auth_token
        )

    def self_attentive_selection(self, word_embedding_matrices, input_encodings, threshold=0.5):
        loss_cs = list()
        generated_sequences = list()
        num_words_list = list()

        for i, word_embedding_matrix in enumerate(word_embedding_matrices):
            current_threshold = threshold
            embedding_dim = word_embedding_matrix.shape[1]
            embedding_dim_root = embedding_dim ** 0.5
            self_attention_matrix = torch.matmul(word_embedding_matrix, word_embedding_matrix.T) / embedding_dim_root
            self_attention_matrix = F.softmax(self_attention_matrix, dim=0)
            embedding_attention_matrix = torch.matmul(self_attention_matrix, word_embedding_matrix).to(self.device)

            input_ids = input_encodings["input_ids"][i]
            q = self.sigmoid(self.W(embedding_attention_matrix).flatten())
            c_subset = self.c[torch.LongTensor(input_ids.detach().clone().tolist())]
            input_ids_selection = (c_subset >= current_threshold).tolist()
            while not input_ids_selection and current_threshold > 0:
                input_ids_selection = (c_subset >= threshold).tolist()
                current_threshold -= 0.05

            selected_input_ids = input_ids[input_ids_selection]
            generated_sequence = self.selector_tokenizer.decode(
                selected_input_ids.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            num_words_list.append(selected_input_ids.shape[0])
            generated_sequences.append(generated_sequence)
            loss = F.kl_div(q.log(), c_subset)
            loss_cs.append(loss)

        loss_cs = torch.stack(loss_cs).to(self.device)
        num_words_list = torch.tensor(num_words_list).to(self.device)
        loss_cs = (loss_cs * num_words_list).mean()
        return loss_cs, generated_sequences

    def forward_whole_word_selection(self, passages, threshold=0.5):
        passage_words = list(map(word_tokenize, passages))
        input_encodings = self.selector_tokenizer.batch_encode_plus(
            passage_words,
            add_special_tokens=False,
            return_tensors="pt",
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.selector_tokenizer.model_max_length,
            return_attention_mask=True
        ).to(self.device)

        word_embedding_matrices = self.selector_model(**input_encodings).last_hidden_state
        loss_cs, generated_sequences = self.self_attentive_selection(
            word_embedding_matrices,
            input_encodings,
            threshold
        )

        # del input_encodings
        return loss_cs, generated_sequences

    def forward_token_selection(self, passages, threshold=0.5):
        input_encodings = self.selector_tokenizer.batch_encode_plus(
            passages,
            add_special_tokens=False,
            return_tensors="pt",
            is_split_into_words=False,
            padding=True,
            truncation=True,
            max_length=self.selector_tokenizer.model_max_length,
            return_attention_mask=True,
        ).to(self.device)

        word_embedding_matrices = self.selector_model(**input_encodings).last_hidden_state
        loss_cs, generated_sequences = self.self_attentive_selection(
            word_embedding_matrices,
            input_encodings,
            threshold
        )
        return loss_cs, generated_sequences

    def forward(self, passages, threshold=0.5):
        if self.selector_mode == "whole-word":
            return self.forward_whole_word_selection(passages, threshold)
        elif self.selector_mode == "token":
            return self.forward_token_selection(passages, threshold)
        else:
            raise ValueError(f"Unknown mode: {self.selector_mode}")
