import torch
import torch.nn as nn
from token_selector import TokenSelector
from transformers import AutoModelForTokenClassification , AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast


class PLUM(nn.Module):
    def __init__(self, selector_type='roberta-base', use_gpt2=True, freeze_gpt2='partial', device='cpu'):
        super(PLUM, self).__init__()
        self.device = device

        self.use_gpt2 = use_gpt2
        self.freeze_gpt2 = freeze_gpt2
        self.selector_type = selector_type
        if self.use_gpt2 or self.selector_type == 'context':
            self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', add_prefix_space=True)
            if self.freeze_gpt2 != 'none':
                for param in self.gpt2_model.parameters():
                    param.requires_grad = False
                if self.freeze_gpt2 == 'partial':
                    for param in self.gpt2_model.lm_head.parameters():        
                        param.requires_grad = True

        if selector_type == 'context':
            self.selector = TokenSelector(self.gpt2_model, self.gpt2_tokenizer, self.device).to(self.device)
        else:
            self.maple = AutoModelForTokenClassification.from_pretrained(selector_type)
            self.maple_tokenizer = self.get_tokenizer(selector_type)
            self.maple.resize_token_embeddings(len(self.maple_tokenizer))


    def get_tokenizer(self, selector_type):
        if "xlm-roberta" in selector_type:
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(
                selector_type, add_prefix_space=True)
        elif "roberta" in selector_type:
            tokenizer = RobertaTokenizerFast.from_pretrained(
                selector_type, add_prefix_space=True)
        elif "gpt" in selector_type:
            tokenizer = GPT2TokenizerFast.from_pretrained(
                selector_type, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(selector_type)
        
        special_tokens = {
            'bos_token': '<|startoftext|>',
            'pad_token': '<|padtext|>',
            'sep_token': '<|septext|>',
        }
        tokenizer.add_special_tokens(special_tokens)
        return tokenizer


    def get_encodings_and_labels(self, tokens, labels, label_all_tokens=False):
        token_encodings = list()
        attention_masks = list()
        label_encodings = list()
        for idx, t in enumerate(tokens):
            tokenized_input = self.maple_tokenizer.encode_plus(t, is_split_into_words=True, max_length=128, padding='max_length', truncation=True)
            input_ids = torch.tensor(tokenized_input['input_ids'])
            attention_mask = torch.tensor(tokenized_input['attention_mask'])
            token_encodings.append(input_ids)
            attention_masks.append(attention_mask)

            word_ids = tokenized_input.word_ids(batch_index=0)
            previous_word_idx = None
            label_ids = list()
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[idx][word_idx])
                else:
                    label_ids.append(labels[idx][word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            label_encodings.append(torch.tensor(label_ids))

        token_encodings = torch.stack(token_encodings).to(self.device)
        attention_masks = torch.stack(attention_masks).to(self.device)
        label_encodings = torch.stack(label_encodings).to(self.device)
        return token_encodings, attention_masks, label_encodings


    def forward(self, tokens, labels):
        output_sequences = list()
        if self.selector_type != 'context':
            token_encodings, attention_masks, label_encodings = self.get_encodings_and_labels(tokens, labels)
            outputs = self.maple(input_ids=token_encodings, attention_mask=attention_masks, labels=label_encodings)
            loss_m, logits = outputs[:2]
            logits = logits.softmax(dim=2).argmax(dim=2)

            for i in range(token_encodings.shape[0]):
                input_sequence = token_encodings[i]
                logit_sequence = logits[i]
                attention_mask = attention_masks[i]
                output_sequence = list()
                for j in range(input_sequence.shape[0]):
                    if attention_mask[j] and logit_sequence[j]:
                        output_sequence.append(input_sequence[j])
                output_sequence = self.maple_tokenizer.decode(output_sequence).strip()
                if not output_sequence: # if no words are chosen, take entire input sequence
                    output_sequence = ' '.join(tokens[i])
                output_sequences.append(output_sequence)
            
        else:
            loss_m = list()
            for idx, t in enumerate(tokens):
                chosen_words, _, loss = self.selector(t)
                if chosen_words:
                    output_sequence = ' '.join(chosen_words)
                else:
                    output_sequence = ' '.join(t)
                output_sequences.append(output_sequence)
                loss_m.append(loss)
            loss_m = torch.stack(loss_m).mean()

        if self.use_gpt2:
            loss_lm = list()
            for output_sequence in output_sequences:
                input_sequence_ids = self.gpt2_tokenizer.encode(output_sequence)
                input_sequence_ids = torch.tensor(input_sequence_ids).to(self.device)
                loss = self.gpt2_model(input_ids=input_sequence_ids, labels=input_sequence_ids).loss
                loss = torch.exp(loss)
                loss_lm.append(loss)
            loss_lm = torch.stack(loss_lm).mean()
        else:
            loss_lm = 0
        
        return loss_m, loss_lm