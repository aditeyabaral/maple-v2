import nltk
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

nltk.download('punkt')
nltk.download('stopwords')
stopwords = list(nltk.corpus.stopwords.words('english'))

class TokenSelector(nn.Module):
    def __init__(self, gpt_model, gpt_tokenizer, device='cpu'):
        super(TokenSelector, self).__init__()

        self.gpt_model = gpt_model
        self.gpt_tokenizer = gpt_tokenizer
        self.device = device

        self.c = nn.Parameter(torch.rand(len(self.gpt_tokenizer)))
        self.W = nn.Linear(self.gpt_model.transformer.wte.weight.shape[1], 1)
        self.sigmoid = nn.Sigmoid()

    def filter_text(self, tokens):
        tokens = [word for word in tokens if word not in stopwords and word not in string.punctuation]
        return tokens

    def forward(self, tokens, preprocess=True, threshold=0.5):
        if preprocess:
            tokens = self.filter_text(tokens)

        token_indices = self.gpt_tokenizer.encode(tokens, is_split_into_words=True)
        token_labels = [self.gpt_tokenizer.decode(w) for w in token_indices]

        embedding_matrix = self.gpt_model.transformer.wte.weight[token_indices,:].to(self.device)
        embedding_dim = embedding_matrix.shape[1]
        embedding_dim_root = embedding_dim ** 0.5

        self_attention_matrix = torch.matmul(embedding_matrix, embedding_matrix.T)/embedding_dim_root
        self_attention_matrix = F.softmax(self_attention_matrix, dim=0)
        embedding_attention_matrix = torch.matmul(self_attention_matrix, embedding_matrix).to(self.device)

        self.q = self.W(embedding_attention_matrix).flatten()
        self.q = self.sigmoid(self.q)
        self.c_subset = self.c[torch.LongTensor(token_indices)]

        word_selection = (self.c_subset >= threshold)
        context_keywords = [token_labels[i] for i in range(len(token_labels)) if word_selection[i]]
        context_keywords_index = [token_indices[i] for i in range(len(token_indices)) if word_selection[i]]
        # context_word_index = text_index[word_selection]   # TODO: This will eliminate the for loop above if fixed
        # context_keywords = [self.gpt_tokenizer.decode(w) for w in context_word_index]
        
        context_keywords = list(map(str.strip, context_keywords))
        loss = F.kl_div(self.q.log(), self.c_subset)
        return context_keywords, context_keywords_index, loss

