# MAPLEv2 -- Multi-task Approach for generating blackout Poetry with Linguistic Evaluation

MAPLEv2 is an improvement over our previous work, [MAPLE](https://github.com/aditeyabaral/maple), which is a Transformer
based blackout poetry generator that uses token classification to learn to write poetry. However, MAPLE is constrained
to only the poems it has seen while learning and hence does not generalise well to unseen passages, thus in most cases
requiring human post-processing to make the poem readable.

This work proposes MAPLEv2, a multi-task approach to improve the generalisation of the model. Unlike v1, MAPLEv2 uses
perplexity and grammatical correctness along with keyword selection to generate blackout poetry. This allows the model
to pick better keywords and generate more readable and coherent sequences. MAPLEv2 is trained on
the same [Blackout Poetry Dataset](https://www.kaggle.com/datasets/aditeyabaral/blackout-poetry-dataset) introduced in
MAPLE.

You can find some of our models on [HuggingFace Hub](https://huggingface.co/maple-v2).

# How to use MAPLEv2

## Setup MAPLEv2 Environment

1. Clone this repository
2. Setup your environment with:
    ```bash
    conda create -n maple-v2 python=3.9
    ```

## Dataset Format

MAPLEv2 requires a dataset in the following format:

| passage                                           | poem                                  | indices           |
|---------------------------------------------------|---------------------------------------|-------------------|
| Did the CIA tell the FBI that it knows the wor... | cia fbi the biggest weapon            | [2, 5, 9, 24, 25] |
| A vigilante lacking of heroic qualities that\n... | lacking qualities that damn criminals | [2, 5, 6, 11, 12] |

The passage is the text from which the poem is generated. The poem is the generated poem. The indices are the indices of
the words in the text that are chosen for the poem. MAPLE's Blackout Poetry dataset has been provided in the `data/`
folder.

## Create Training Data

If you do not have a training dataset, you can follow the same instructions as in
the [MAPLE documentation](https://github.com/aditeyabaral/maple/blob/main/README.md#create-training-data) to create one.

## Train MAPLEv2

To train the MAPLEv2, use the `src/train.py` script. The arguments are explained below.

```bash
usage: python src/train_v1.py [-h] --data DATA [--selector-type {context,maple}] [--selector-model-path SELECTOR_MODEL_PATH]
                [--selector-mode {whole-word,token}] [--gpt-model-path GPT_MODEL_PATH] [--freeze-gpt]
                [--batch-size BATCH_SIZE] [--grammar-checker-model-path GRAMMAR_CHECKER_MODEL_PATH]
                [--freeze-grammar-checker] [--epochs EPOCHS] [--alpha ALPHA] [--beta BETA] [--gamma GAMMA]
                [--use-selector-loss] [--learning-rate LEARNING_RATE] [--save-model-path SAVE_MODEL_PATH]
                [--save-every SAVE_EVERY] [--save-local] [--generate-every GENERATE_EVERY]
                [--hub-model-name HUB_MODEL_NAME] [--auth-token AUTH_TOKEN] [--resume RESUME] [--device DEVICE]

Train a MAPLEv2 Model

optional arguments:
  -h, --help            show this help message and exit
  --data DATA, -d DATA  Path to the dataset
  --selector-type {context,maple}, -s {context,maple}
                        Word Selector Type
  --selector-model-path SELECTOR_MODEL_PATH, -sm SELECTOR_MODEL_PATH
                        Path to the selector Transformer model
  --selector-mode {whole-word,token}, -smo {whole-word,token}
                        Word Selector Mode
  --gpt-model-path GPT_MODEL_PATH, -gm GPT_MODEL_PATH
                        Path to the GPT model
  --freeze-gpt, -fg     Freeze the GPT model
  --batch-size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size
  --epochs EPOCHS, -e EPOCHS
                        Number of epochs
  --alpha ALPHA, -a ALPHA
                        Alpha coefficient for selector
  --beta BETA, -b BETA  Beta coefficient for perplexity
  --use-selector-loss, -usl
                        Use Maple or Context Selector Loss during Training
  --learning-rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate
  --save-model-path SAVE_MODEL_PATH, -smp SAVE_MODEL_PATH
                        Path to save the model
  --save-every SAVE_EVERY, -se SAVE_EVERY
                        Save model every n epochs
  --save-local, -sl     Save model locally
  --hub-model-name HUB_MODEL_NAME, -hmn HUB_MODEL_NAME
                        Name of the HuggingFace Hub model
  --auth-token AUTH_TOKEN, -at AUTH_TOKEN
                        Huggingface Auth token
  --resume RESUME, -r RESUME
                        Path to checkpoint
  --device DEVICE, -dev DEVICE
                        Device to use
```

MAPLEv2 can use 2 methods to select tokens from a passage:

- **MAPLE-style token classification**: The MAPLE-style token classification is the same as the one used in MAPLE. You
  can
  use a MAPLE token classifier as the
  selector model by specifying the `selector_type` as `maple`.
  In this scenario it is advisable to not use the MAPLE loss returned (`loss_m`) since it will train the token
  classifier
  to pick the same tokens as in the labels. Instead, use only the perplexity loss (`loss_ppl`) and do not set
  the `--use-selector-loss` flag.


- **Context-based token selection (Whole-word/Token)**: The context-based token selection is a new
  method that uses a simple self-attention matrix and a learnable selector vector to select the best tokens. This method
  can select either whole words or individual tokens. The `selector_mode` argument specifies the mode to use. The
  loss (`loss_cs`) should be used otherwise the model will not learn how to select tokens from the passage. The
  `--use-selector-loss` argument should be set.

Here is a minimal example to train a MAPLEv2 model:

```py
from maple

-v2
import MAPLEv2

# Initialise a MAPLEv2 model with RoBERTa Whole Word Context Selector as Token Selector and GPT2 as Perplexity Evaluator
model = MAPLEv2(
    selector_type="context",
    selector_model_path="roberta-base",
    selector_mode="whole-word",
    gpt_model_path="gpt2",
    freeze_gpt=True,
    device="cuda"
).to("cuda")

# Perform a forward pass and return losses and generated sequences
outputs = model(passages=passages, tokens=tokens, labels=labels)
outputs.keys()
# dict_keys(['loss_cs', 'loss_ppl', 'generated_sequences'])

# Perform a backward pass and update the model parameters
ALPHA = 1
BETA = 0.001
loss = ALPHA * outputs["loss_cs"] + BETA * outputs["loss_ppl"]
loss.backward()
```

## Train MAPLE

You can set the arguments such that a MAPLE model is trained. Set the following arguments:

```py
from maple

-v2
import MAPLEv2

# Initialise a MAPLE model with RoBERTa Token Classifier as Token Selector and no GPT2 Perplexity Evaluator
model = MAPLEv2(
    selector_type="maple",
    selector_model_path="roberta-base",
    gpt_model_path=None,
    device="cuda"
).to("cuda")

# Perform a forward pass and return losses and generated sequences
outputs = model(passages=passages, tokens=tokens, labels=labels)
outputs.keys()
# dict_keys(['loss_m', 'generated_sequences'])

# Perform backward pass and update the model parameters
outputs["loss_m"].backward()
```