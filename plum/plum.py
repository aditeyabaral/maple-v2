import torch
import torch.nn as nn
import torch.nn.functional as F
from .context_selector import ContextSelector
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer


class PLUM(nn.Module):
    def __init__(
            self,
            selector_type='context',
            selector_model_path="roberta-base",
            selector_mode="whole-word",
            gpt_model_path="gpt2",
            freeze_gpt=True,
            device='cuda'
    ):
        super(PLUM, self).__init__()
        self.device = device
        self.selector_type = selector_type
        self.selector_model_path = selector_model_path
        self.selector_mode = selector_mode
        self.gpt_model_path = gpt_model_path
        self.freeze_gpt = freeze_gpt

        if self.gpt_model_path is not None:
            self.gpt_tokenizer = AutoTokenizer.from_pretrained(self.gpt_model_path)
            self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
            self.gpt_model = AutoModelForCausalLM.from_pretrained(self.gpt_model_path).to(self.device)
            if self.freeze_gpt:
                for param in self.gpt_model.parameters():
                    param.requires_grad = False

        if self.selector_type == "context":
            self.token_selector = ContextSelector(
                selector_model_path=self.selector_model_path,
                selector_mode=self.selector_mode,
                device=self.device
            ).to(self.device)
        elif self.selector_type == "maple":
            self.selector_tokenizer = AutoTokenizer.from_pretrained(
                self.selector_model_path,
                add_prefix_space=True
            )
            self.token_selector = AutoModelForTokenClassification.from_pretrained(self.selector_model_path).to(
                self.device)
        else:
            raise ValueError("Invalid selector type. Use 'context' or 'maple'")

    def get_input_encodings(self, tokens, labels, max_length=256):
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

        return token_encodings, labels

    def forward_gpt(self, generated_sequences):
        encodings = self.gpt_tokenizer.batch_encode_plus(
            generated_sequences,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        perplexity_loss = torch.exp(self.gpt_model(**encodings, labels=encodings['input_ids']).loss) \
            .requires_grad_(True)
        return perplexity_loss

    def forward_maple(self, tokens, labels):
        tokens, labels = self.get_input_encodings(tokens, labels)
        outputs = self.token_selector(**tokens, labels=labels)
        logits = outputs.logits

        logits = logits.softmax(dim=-1).argmax(dim=-1)
        generated_sequences = list()
        for idx, logit_sequence in enumerate(logits):
            generated_sequence = list()
            for logit_idx, logit in enumerate(logit_sequence):
                if logit and tokens['attention_mask'][idx][logit_idx]:
                    generated_sequence.append(tokens['input_ids'][idx][logit_idx])
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
        outputs = dict()
        if self.selector_type == "context":
            loss_cs, generated_sequences = self.forward_context_selector(passages)
            outputs['loss_cs'] = loss_cs
            outputs['generated_sequences'] = generated_sequences
        elif self.selector_type == "maple":
            maple_outputs, generated_sequences = self.forward_maple(tokens, labels)
            outputs['maple_outputs'] = maple_outputs
            outputs['loss_m'] = maple_outputs.loss
            outputs['generated_sequences'] = generated_sequences
        else:
            raise ValueError("Invalid selector type initialized. Use 'context' or 'maple'")

        if self.gpt_model_path is not None:
            loss_ppl = self.forward_gpt(generated_sequences)
            outputs['loss_ppl'] = loss_ppl

        return outputs
