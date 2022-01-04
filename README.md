# PLUM - Perplexity guided Learning for Understanding better sequences and improving Maple

Implementation of **PLUM - Perplexity guided Learning for Understanding better sequences and improving Maple**.

PLUM is an improvement over our previous work, [MAPLE](https://github.com/aditeyabaral/maple), which is a deep learning based blackout poetry generator that uses token classification to learn to write poetry. However, MAPLE is constrained to only the poems it has seen while learning and hence does not generalise well to unseen passages.

PLUM discards poem sequences and instead leverages perplexity as a metric to learn how to pick words that form the best sequences, resulting in more probable and better quality poetry.

[[PLUM MODELS]](https://huggingface.co/plum)

# How to use PLUM

## Setup PLUM Environment

1. Clone this repository
2. Setup your environment with:
```bash
conda env create -f environment.yml
```

## Dataset Format

PLUM required a dataset in the following format:

| passage                                           | poem                                  | indices           |
|---------------------------------------------------|---------------------------------------|-------------------|
| Did the CIA tell the FBI that it knows the wor... | cia fbi the biggest weapon            | [2, 5, 9, 24, 25] |
| A vigilante lacking of heroic qualities that\n... | lacking qualities that damn criminals | [2, 5, 6, 11, 12] |

The passage is the text from which the poem is generated. The poem is the generated poem. The indices are the indices of the words in the text that are chosen for the poem. A sample dataset has been provided in the `data/` folder.

## Create Training Data

If you do not have a training dataset, you can follow the same instructions as in the [MAPLE documentation](https://github.com/aditeyabaral/maple/blob/main/README.md#create-training-data) to create one.


## Train PLUM

To train the PLUM, use the `plum.py` script. The arguments are self explanatory.

```bash
usage: python plum.py [-h] --model MODEL --dataset DATASET [--hub HUB] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
               [--epochs EPOCHS] [--username USERNAME] [--password PASSWORD] [--output OUTPUT] [--hub_name HUB_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Transformer model name/path to train
  --dataset DATASET, -d DATASET
                        Path to dataset in required format
  --hub HUB, -hf HUB    Push model to HuggingFace Hub
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate
  --epochs EPOCHS, -e EPOCHS
                        Number of epochs
  --username USERNAME, -u USERNAME
                        Username for HuggingFace Hub
  --password PASSWORD, -p PASSWORD
                        Password for HuggingFace Hub
  --output OUTPUT, -o OUTPUT
                        Output directory path
  --hub_name HUB_NAME, -hn HUB_NAME
                        Name of the model in the HuggingFace Hub
```