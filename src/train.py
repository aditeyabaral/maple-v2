import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import argparse
from nltk.tokenize import word_tokenize
from plum import *

from transformers import AdamW
import argparse
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='logs')

parser = argparse.ArgumentParser(description='Train a PLUM Model')
parser.add_argument('--data', '-d', type=str,
                    help='Path to the dataset', required=True)
parser.add_argument('--selector_model', '-s', type=str,
                    help='Transformer path or word selector type', default='roberta-base')
parser.add_argument('--use_gpt2', '-g', action='store_true', 
                    help='Use GPT2 for perplexity loss')
parser.add_argument('--freeze_gpt2', '-fg', default='partial', choices=['none', 'partial', 'all'],
                    help='Freeze GPT2 weights')
parser.add_argument('--batch_size', '-bs', type=int,
                    help='Batch size', default=4)
parser.add_argument('--epochs', '-e', type=int, 
                    help='Number of epochs', default=10)      
parser.add_argument('--beta', '-b', type=float,
                    help='Beta value for PLUM', default=0.001)              
parser.add_argument('--learning_rate', '-lr', type=float, 
                    help='Learning rate', default=1e-8)     
parser.add_argument('--model_path', '-m', type=str, default='saved_models/',
                    help='Path to save the model')
parser.add_argument('--save_every', '-se', type=int, default=4,
                    help='Save model every n epochs')
parser.add_argument('--resume_checkpoint', '-r', action='store_true',
                    help='Resume from checkpoint')
parser.add_argument('--checkpoint_path', '-cp', default=None, type=str, help='Path to checkpoint')
parser.add_argument('--device', '-dev', default='cuda', type=str, help='Device to use')
args = parser.parse_args()
print(args)

DATASET_PATH = args.data
SELECTOR_MODEL = args.selector_model
USE_GPT2 = args.use_gpt2
FREEZE_GPT2 = args.freeze_gpt2
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
BETA = args.beta
LEARNING_RATE = args.learning_rate
MODEL_PATH = args.model_path
SAVE_EVERY = args.save_every
RESUME_CHECKPOINT = args.resume_checkpoint
CHECKPOINT_PATH = args.checkpoint_path

if torch.cuda.is_available() and args.device == 'cuda':
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

df = pd.read_json(DATASET_PATH)
df = df.drop_duplicates(subset=["passage", "poem"], ignore_index=True)
df["tokens"] = df["passage"].apply(lambda x: word_tokenize(x))
ner_tags = list()
for i in range(df.shape[0]):
    indices = df["indices"][i]
    length = len(df["tokens"][i])
    ner_tag = ['O' for _ in range(length)]
    for idx in indices:
        ner_tag[idx] = 'W'
    ner_tags.append(ner_tag)
df["ner_tags"] = ner_tags

tokens = df["tokens"]
tags = df["ner_tags"]
unique_tags = ["O", "W"]
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
labels = [[tag2id[tag] for tag in tags[i]] for i in range(len(tags))]

model = PLUM(
    selector_type=SELECTOR_MODEL,
    use_gpt2=USE_GPT2,
    freeze_gpt2=FREEZE_GPT2,
    device=DEVICE
)
model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

if RESUME_CHECKPOINT:
    model.load_state_dict(torch.load(CHECKPOINT_PATH)['model_state_dict'])
    print('Model loaded from checkpoint')   
    optimizer.load_state_dict(torch.load(CHECKPOINT_PATH)['optimizer_state_dict'])
    print('Optimizer loaded from checkpoint')
    start_epoch = int(torch.load(CHECKPOINT_PATH)['epoch'])
    print('Resuming training from epoch:', start_epoch)
    BETA = float(torch.load(CHECKPOINT_PATH)['beta'])
else:
    start_epoch = 1

steps = 0
for epoch in tqdm(range(start_epoch, EPOCHS + 1)):
    for i in tqdm(range(0, len(tokens), BATCH_SIZE)):
        model.zero_grad()
        model.train()

        batch_tokens = list(tokens[i:i + BATCH_SIZE])
        batch_labels = list(labels[i:i + BATCH_SIZE])
        loss_m, loss_lm = model(batch_tokens, batch_labels)

        if USE_GPT2:
            loss = loss_lm + BETA * loss_m
        else:
            loss = loss_m
        
        writer.add_scalar('Loss/loss_m', loss_m.item(), steps)
        writer.add_scalar('Loss/loss_lm', loss_lm.item(), steps)
        writer.add_scalar('Loss/loss', loss.item(), steps)
        writer.flush()
        steps += 1

        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch}')
    print(f'Loss: {loss.item()}')
    if epoch % SAVE_EVERY == 0:
        torch.save(
            {
                'epoch': epoch,
                'loss': loss.item(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 
            f'{MODEL_PATH}/model_epoch_{epoch}.pt'
        )

torch.save(
            {
                'epoch': epoch,
                'loss': loss.item(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 
            f'{MODEL_PATH}/model_epoch_{epoch}.pt'
        )

writer.close()
