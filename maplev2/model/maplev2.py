from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from .context_selector import ContextSelector


class MAPLEv2Output:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MAPLEv2(nn.Module):
    MODEL_NAME = "Multi-task Approach to blackout Poetry generation using Linguistic Evaluation"

    def __init__(
            self,
            selector_type='v2',
            selector_model_path="roberta-base",
            selector_mode="whole-word",
            gpt_model_path="gpt2",
            freeze_gpt=True,
            grammar_checker_model_path=None,
            freeze_grammar_checker=True,
            device='cuda'
    ):
        super(MAPLEv2, self).__init__()
        self.device = device
        self.selector_type = selector_type
        self.selector_model_path = selector_model_path
        self.selector_mode = selector_mode
        self.gpt_model_path = gpt_model_path
        self.freeze_gpt = freeze_gpt
        self.grammar_checker_model_path = grammar_checker_model_path
        self.freeze_grammar_checker = freeze_grammar_checker

        if self.gpt_model_path is not None:
            self.gpt_tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_path)
            self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
            self.gpt_model = AutoModelForCausalLM.from_pretrained(self.gpt_model_path).to(self.device)
            if self.freeze_gpt:
                for param in self.gpt_model.parameters():
                    param.requires_grad = False

        if self.selector_type == "v2":
            self.token_selector = ContextSelector(
                selector_model_path=self.selector_model_path,
                selector_mode=self.selector_mode,
                device=self.device
            ).to(self.device)
            self.selector_model = self.token_selector.selector_model
            self.selector_tokenizer = self.token_selector.selector_tokenizer
        elif self.selector_type == "v1":
            self.token_selector = AutoModelForTokenClassification.from_pretrained(
                self.selector_model_path).to(self.device)
            self.selector_model = self.token_selector.base_model
            self.selector_tokenizer = AutoTokenizer.from_pretrained(
                self.selector_model_path,
                add_prefix_space=True
            )
        else:
            raise ValueError("Invalid selector type. Use 'context' or 'maple'")

        if self.grammar_checker_model_path is not None:
            self.grammar_checker = AutoModelForSequenceClassification.from_pretrained(
                self.grammar_checker_model_path).to(self.device)
            if self.freeze_grammar_checker:
                for param in self.grammar_checker.parameters():
                    param.requires_grad = False
            self.grammar_checker_tokenizer = AutoTokenizer.from_pretrained(self.grammar_checker_model_path)

    def forward_gpt(self, generated_sequences):
        encoding = self.selector_tokenizer.batch_encode_plus(
            generated_sequences,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        embedding = self.selector_model(**encoding).last_hidden_state
        encoding = self.gpt_tokenizer.batch_encode_plus(
            generated_sequences,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        outputs = self.gpt_model(inputs_embeds=embedding, labels=encoding['input_ids'])
        perplexity = torch.exp(outputs.loss)
        return perplexity

    def forward_grammar(self, generated_sequences):
        encoding = self.selector_tokenizer.batch_encode_plus(
            generated_sequences,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        embedding = self.selector_model(**encoding).last_hidden_state
        logits = self.grammar_checker(inputs_embeds=embedding).logits
        logits = F.softmax(logits, dim=1)
        loss = logits[:, 0].mean() + (1 / (logits[:, 1].mean() + 1e-5))
        return loss

    def forward_maple(self, tokens, labels, max_length=256):
        token_encodings = self.selector_tokenizer.batch_encode_plus(
            tokens,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            add_special_tokens=False,
            is_split_into_words=True,
            return_tensors='pt'
        ).to(self.device)

        for idx, input_ids in enumerate(token_encodings['input_ids']):
            label_length = len(labels[idx])
            labels[idx] = F.pad(labels[idx], (0, max_length - label_length), value=-100).type(torch.long)
        labels = torch.stack(labels).to(self.device)

        outputs = self.token_selector(**token_encodings, labels=labels)
        logits = outputs.logits
        logits = logits.softmax(dim=-1).argmax(dim=-1)
        generated_sequences = list()
        for idx, logit_sequence in enumerate(logits):
            generated_sequence = list()
            for logit_idx, logit in enumerate(logit_sequence):
                if logit and token_encodings['attention_mask'][idx][logit_idx]:
                    generated_sequence.append(token_encodings['input_ids'][idx][logit_idx])
            generated_sequences.append(self.selector_tokenizer.decode(generated_sequence))

        return outputs, generated_sequences

    def forward_context_selector(self, passages):
        loss_cs, context_keywords = self.token_selector(passages)
        generated_sequences = list(map(' '.join, context_keywords))
        return loss_cs, generated_sequences

    def forward(self, **kwargs):
        tokens = kwargs.get('tokens', [])
        labels = kwargs.get('labels', [])
        passages = kwargs.get('passages', [])
        outputs = MAPLEv2Output()

        if self.selector_type == "v2":
            if not passages:
                raise ValueError("Variable 'passages' must be provided for context selector")
            loss_s, generated_sequences = self.forward_context_selector(passages)
            outputs.loss_s = loss_s
            outputs.generated_sequences = generated_sequences

        elif self.selector_type == "v1":
            if not tokens or not labels:
                raise ValueError("Variables 'tokens' and 'labels' must be provided for maple selector")
            maple_outputs, generated_sequences = self.forward_maple(tokens, labels)
            outputs.maple_outputs = maple_outputs
            outputs.loss_s = maple_outputs.loss
            outputs.generated_sequences = generated_sequences

        else:
            raise ValueError("Invalid selector type initialized. Use 'context' or 'maple'")

        if self.gpt_model_path is not None:
            outputs.loss_ppl = self.forward_gpt(generated_sequences)

        if self.grammar_checker_model_path is not None:
            outputs.loss_g = self.forward_grammar(generated_sequences)

        return outputs

    def save(self, path):
        if not Path(path).parent.exists():
            raise ValueError("Invalid path provided. Parent directory does not exist")

        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'selector_type': self.selector_type,
                'selector_model_path': self.selector_model_path,
                'selector_mode': self.selector_mode,
                'gpt_model_path': self.gpt_model_path,
                'grammar_checker_model_path': self.grammar_checker_model_path,
                'freeze_grammar_checker': self.freeze_grammar_checker,
                'freeze_gpt': self.freeze_gpt
            },
            path
        )

    def push_to_hub(self, repo_name, commit_message, auth_token):
        if self.selector_type == "v2":
            self.token_selector.push_to_hub(repo_name, commit_message, auth_token)
        elif self.selector_type == "v1":
            self.token_selector.push_to_hub(
                repo_name,
                commit_message=f"Model: {commit_message}",
                use_auth_token=auth_token
            )
            self.selector_tokenizer.push_to_hub(
                repo_name,
                commit_message=f"Tokenizer: {commit_message}",
                use_auth_token=auth_token
            )
        else:
            raise NotImplementedError

    def load(self, path):
        if not Path(path).exists():
            raise ValueError("Invalid path provided. File does not exist")

        checkpoint = torch.load(path, map_location=self.device)
        self.selector_type = checkpoint['selector_type']
        self.selector_model_path = checkpoint['selector_model_path']
        self.selector_mode = checkpoint['selector_mode']
        self.gpt_model_path = checkpoint['gpt_model_path']
        self.grammar_checker_model_path = checkpoint['grammar_checker_model_path']
        self.freeze_grammar_checker = checkpoint['freeze_grammar_checker']
        self.freeze_gpt = checkpoint['freeze_gpt']
        self.load_state_dict(checkpoint['model_state_dict'])
