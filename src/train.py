import sys
import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent))

from plum import PLUM

shutil.rmtree("logs", ignore_errors=True)
writer = SummaryWriter(log_dir='logs')

parser = argparse.ArgumentParser(description='Train a PLUM Model')
parser.add_argument('--data', '-d', type=str,
                    help='Path to the dataset', required=True)
parser.add_argument('--selector-type', '-s', type=str,
                    help='Word Selector Type', default='context', choices=['context', 'maple'])
parser.add_argument('--selector-model-path', '-sm', type=str,
                    help='Path to the selector Transformer model', default='roberta-base')
parser.add_argument('--selector-mode', '-smo', type=str,
                    help='Word Selector Mode', default='whole-word', choices=['whole-word', 'token'])
parser.add_argument('--gpt-model-path', '-gm', type=str,
                    help='Path to the GPT model', default=None)
parser.add_argument('--freeze-gpt', '-fg', default=True, action='store_true',
                    help='Freeze the GPT model')
parser.add_argument('--batch-size', '-bs', type=int,
                    help='Batch size', default=4)
parser.add_argument('--epochs', '-e', type=int,
                    help='Number of epochs', default=10)
parser.add_argument('--alpha', '-a', type=float,
                    help='Alpha coefficient for selector', default=1)
parser.add_argument('--beta', '-b', type=float,
                    help='Beta coefficient for perplexity', default=0.001)
parser.add_argument('--use-selector-loss', '-usl', default=False, action='store_true',
                    help='Use Maple or Context Selector Loss during Training')
parser.add_argument('--learning-rate', '-lr', type=float,
                    help='Learning rate', default=0.0001)
parser.add_argument('--save-model-path', '-smp', type=str, default='./saved_models',
                    help='Path to save the model')
parser.add_argument('--save-every', '-se', type=int, default=1,
                    help='Save model every n epochs')
parser.add_argument('--save-local', '-sl', default=False, action='store_true',
                    help='Save model locally')
parser.add_argument("--hub-model-name", "-hmn", type=str, help="Name of the HuggingFace Hub model")
parser.add_argument("--auth-token", "-at", type=str, help="Huggingface Auth token")
parser.add_argument('--resume', '-r', default=None, type=str, help='Path to checkpoint')
parser.add_argument('--device', '-dev', default='cuda', type=str, help='Device to use')
args = parser.parse_args()
print(args)

SELECTOR_TYPE = args.selector_type
SELECTOR_MODEL_PATH = args.selector_model_path
SELECTOR_MODE = args.selector_mode
GPT_MODEL_PATH = args.gpt_model_path
FREEZE_GPT = args.freeze_gpt

DATASET_PATH = args.data
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
USE_SELECTOR_LOSS = args.use_selector_loss
ALPHA = args.alpha
BETA = args.beta
LEARNING_RATE = args.learning_rate
SAVE_MODEL_PATH = args.save_model_path
SAVE_EVERY = args.save_every
SAVE_LOCAL = args.save_local
CHECKPOINT_PATH = args.resume
HUB_MODEL_NAME = args.hub_model_name
AUTH_TOKEN = args.auth_token
DEVICE = args.device

if torch.cuda.is_available() and args.device == 'cuda':
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def read_dataset(path):
    df = pd.read_json(path)
    df = df.drop_duplicates(subset=["passage", "poem"], ignore_index=True)
    passages = df['passage'].tolist()
    tokens = df["passage"].apply(lambda x: word_tokenize(x))
    labels = list()
    for i in range(df.shape[0]):
        indices = df["indices"][i]
        indices_length = len(tokens[i])
        selection_list = torch.zeros(indices_length)
        for idx in indices:
            selection_list[idx] = 1
        labels.append(selection_list)
    return passages, tokens, labels


def save_model(current_epoch):
    if SAVE_LOCAL:
        torch.save(
            {
                'selector_type': SELECTOR_TYPE,
                'selector_model_path': SELECTOR_MODEL_PATH,
                'gpt_model_path': GPT_MODEL_PATH,
                'freeze_gpt': FREEZE_GPT,
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },
            f'{SAVE_MODEL_PATH}/model_epoch_{current_epoch}.pt'
        )
    if AUTH_TOKEN is not None and HUB_MODEL_NAME is not None:
        if model.selector_type == 'maple':
            model.token_selector.push_to_hub(
                f"{HUB_MODEL_NAME}",
                use_auth_token=AUTH_TOKEN,
                commit_message=f"Epoch {current_epoch}",
            )
        else:
            model.token_selector.selector_model.push_to_hub(
                f"{HUB_MODEL_NAME}",
                use_auth_token=AUTH_TOKEN,
                commit_message=f"Epoch {current_epoch}",
            )


passages, tokens, labels = read_dataset(DATASET_PATH)
total_examples = len(tokens)

model = PLUM(
    selector_type=SELECTOR_TYPE,
    selector_model_path=SELECTOR_MODEL_PATH,
    selector_mode=SELECTOR_MODE,
    gpt_model_path=GPT_MODEL_PATH,
    freeze_gpt=FREEZE_GPT,
    device=DEVICE
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

if CHECKPOINT_PATH:
    model.load_state_dict(torch.load(CHECKPOINT_PATH)['model_state_dict'])
    print('Model loaded from checkpoint')
    optimizer.load_state_dict(torch.load(CHECKPOINT_PATH)['optimizer_state_dict'])
    print('Optimizer loaded from checkpoint')

steps = 0
model.zero_grad()
model.train()
torch.cuda.empty_cache()
Path(SAVE_MODEL_PATH).mkdir(parents=True, exist_ok=True)

for epoch in tqdm(range(1, EPOCHS + 1)):
    total_epoch_loss = 0
    num_batches = 0
    for i in tqdm(range(0, total_examples, BATCH_SIZE)):
        optimizer.zero_grad()

        batch_passages = list(passages[i:i + BATCH_SIZE])
        batch_tokens = list(tokens[i:i + BATCH_SIZE])
        batch_labels = list(labels[i:i + BATCH_SIZE])
        outputs = model(
            passages=batch_passages,
            tokens=batch_tokens,
            labels=batch_labels
        )

        for key in outputs:
            if key.startswith('loss') and outputs[key] is not None:
                writer.add_scalar(f"Loss/{key}", outputs[key].item(), steps)

        for idx, generated_sequence in enumerate(outputs.get("generated_sequences", [])):
            writer.add_text(batch_passages[idx][:30], generated_sequence, steps)

        loss = torch.tensor(0.0).to(DEVICE)
        if USE_SELECTOR_LOSS:
            if SELECTOR_TYPE == 'maple':
                loss += ALPHA * outputs['loss_m']
            else:
                loss += ALPHA * outputs['loss_cs']
        if GPT_MODEL_PATH:
            loss += BETA * outputs['loss_ppl']

        writer.add_scalar("Loss/loss", loss.item(), steps)

        total_epoch_loss += loss
        loss.backward()
        steps += 1
        num_batches += 1

    total_epoch_loss /= num_batches
    scheduler.step(total_epoch_loss)
    print(f"Epoch {epoch} completed\nCurrent lr: {optimizer.param_groups[0]['lr']}\nTotal loss: {total_epoch_loss}")

    if epoch % SAVE_EVERY == 0:
        save_model(epoch)

save_model(EPOCHS)
writer.close()
