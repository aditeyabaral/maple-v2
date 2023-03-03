# MAPLEv2 - Multi-task Approach for generating blackout Poetry with Linguistic Evaluation

MAPLEv2 is an improvement over our previous work, [MAPLE](https://github.com/aditeyabaral/maple), a Transformer
based blackout poetry generator that uses token classification to learn to write poetry. However, MAPLE is constrained
to only the poems it has seen while learning and hence does not generalise well to unseen passages, thus in most cases
requiring human post-processing to make the poem readable.

This work proposes MAPLEv2, a multi-task approach to improve the generalisation of the model. Unlike the former, MAPLEv2
uses perplexity and grammatical correctness along with keyword selection to generate blackout poetry. This allows the
model to pick better keywords and generate more readable and coherent sequences.

You can find some of our models on [HuggingFace Hub](https://huggingface.co/maple-v2).

## Setup MAPLEv2 Environment

1. Clone this repository
2. Setup your environment with:
    ```bash
    conda create -n maplev2 python=3.10
    conda activate maplev2
    pip install -r requirements.txt
    ```

# How to use MAPLEv2

MAPLEv2 has been packaged into an easy-to-use library that can be imported into your Python code and consists of trainer
and model classes. You can find the code in the `maplev2/` folder. You can find example training code in the `src/`
folders which will show you how to train MAPLEv1 as well as MAPLEv2.

## Setting up the Dataset

MAPLEv2 is trained on the same
[Blackout Poetry Dataset](https://www.kaggle.com/datasets/aditeyabaral/blackout-poetry-dataset) introduced in MAPLE,
which is shown below.

| passage                                           | poem                                  | indices           |
|---------------------------------------------------|---------------------------------------|-------------------|
| Did the CIA tell the FBI that it knows the wor... | cia fbi the biggest weapon            | [2, 5, 9, 24, 25] |
| A vigilante lacking of heroic qualities that\n... | lacking qualities that damn criminals | [2, 5, 6, 11, 12] |

The passage is the text from which the poem is generated. The poem is the generated poem. The indices are the indices of
the words in the text that are chosen for the poem. MAPLE's Blackout Poetry dataset has been provided in the `data/`
folder.

### Create Training Data

If you do not have a training dataset, you can follow the same instructions as in
the [MAPLE documentation](https://github.com/aditeyabaral/maple/blob/main/README.md#create-training-data) to create one.

## Initialise the Model

First, initialise the MAPLEv2 model. This requires an embedding model, which can be any model from
HuggingFaceTransformers. We recommend any of the BERT models.

MAPLEv2 can use 2 methods to select tokens from a passage:

- **MAPLEv1-style token classification**: The MAPLE-style token classification is the same as the one used in MAPLE. You
  can use a MAPLE token classifier as the selector model by specifying the `selector_type` as `v1`. In this scenario it
  is advisable to *not use the selector loss (`loss_s`) returned* since it will train the token
  classifier to pick the same tokens as in the labels. Instead, use only the perplexity loss (`loss_ppl`) and the
  grammar-checker loss `loss_g` and set the `use_selector_loss` flag in the trainer to `False`.


- **Context-based token selection (Whole-word/Token)**: The context-based selection is a new
  method that uses a simple self-attention matrix and a learnable selector vector to select the best tokens and be used
  by setting the `selector_type` as `v2`. This method
  can select either whole words or individual tokens. The `selector_mode` argument specifies the mode to use(`token`
  or `whole-word`). The loss (`loss_s`) should be used otherwise the model will not learn how to select tokens from the
  passage. The`use_selector_loss` argument should be set to `True` in the trainer.

The `gpt_model_path` is used to set the GPT model to be used for perplexity calculation. We recommend using the `gpt2`.
Finally, the `grammar_checker_model_path` is used to set the grammar-checker model to be used for grammatical
correctness calculation. We recommend a model that has been trained on the GLUE CoLa dataset (
like `jxuhf/roberta-base-finetuned-cola`). The labels of the grammar-checker model should return a `0` for the incorrect
sentence and a `1` for the correct sentence. Both the GPT and grammar-checker models should be frozen as well.

```python
from maplev2 import MAPLEv2

model = MAPLEv2(
    selector_type="v2",
    selector_model_path="roberta-base",
    selector_mode="whole-word",
    gpt_model_path="gpt2",
    freeze_gpt=True,
    grammar_checker_model_path="jxuhf/roberta-base-finetuned-cola",
    freeze_grammar_checker=True,
    device="cuda",
)
```

## Load the dataset

A dataset loader class has been included to load and process any custom MAPLE-style dataset. The dataset loader class
takes in a dataset path.

```python
from maplev2 import MAPLEDataset

dataset = MAPLEDataset()
dataset.load("data/data.json")
```

## Initialise the Trainer

The trainer class is used to train the model. The trainer takes in the model, learning rate and other arguments about
the optimizer and scheduler.

```python
from maplev2 import MAPLEv2Trainer

trainer = MAPLEv2Trainer(
    model=model,
    learning_rate=1e-8,
    optimizer_class="adamw",
    scheduler_class="plateau",
    factor=0.5,
    patience=1,
)
```

## Train the model

The `train` method of the trainer class is used to train the model. The `train` method takes in the dataset, model and
other arguments which are self-explanatory.

```python
trainer.train(
    dataset=dataset,  # The dataset to train on
    model=model,  # The model to train
    batch_size=8,  # The batch size to use
    epochs=10,  # The number of epochs to train for
    alpha=10,  # selector loss weight
    beta=0.0002,  # perplexity loss weight
    gamma=0.05,  # grammar-checker loss weight
    use_tensorboard=True,  # Whether to use tensorboard for logging losses and generated poems
    generate_every=1000,  # sample every n steps
    upload_model_to_hub=True,  # upload model to HuggingFace Hub
    use_selector_loss=True,  # use selector loss
    use_absolute_selector_loss=False,  # use absolute of selector loss
    save_every=1,  # save model every epoch
    save_latest=True,  # save the latest model only
    save_dir="./saved_models",  # save directory
    hub_model_name="roberta-base-maplev2",  # HuggingFace Hub model name
    hub_organization="maple-v2",  # HuggingFace Hub organization
    auth_token="your-auth-token"  # HuggingFace Hub auth token
)
```

## Putting it all together

```python
from maplev2 import MAPLEv2, MAPLEDataset, MAPLEv2Trainer

model = MAPLEv2(
    selector_type="v2",
    selector_model_path="roberta-base",
    selector_mode="whole-word",
    gpt_model_path="gpt2",
    freeze_gpt=True,
    grammar_checker_model_path="jxuhf/roberta-base-finetuned-cola",
    freeze_grammar_checker=True,
    device="cuda",
)

dataset = MAPLEDataset()
dataset.load("data/data.json")

trainer = MAPLEv2Trainer(
    model=model,
    learning_rate=1e-8,
    optimizer_class="adamw",
    scheduler_class="plateau",
    factor=0.5,
    patience=1,
)

trainer.train(
    dataset=dataset,
    model=model,
    batch_size=8,
    epochs=10,
    alpha=10,
    beta=0.0002,
    gamma=0.05,
    use_tensorboard=True,
    generate_every=1000,
    upload_model_to_hub=True,
    use_selector_loss=True,
    use_absolute_selector_loss=False,
    save_every=1,
    save_latest=True,
    save_dir="./saved_models",
    hub_model_name="roberta-base-maplev2",
    hub_organization="maple-v2",
    auth_token="your-auth-token"
)
```

# Train MAPLEv1 models

You can specify the model initialisation arguments in such a way to train only MAPLE v1 models. By specifying `None` for
the `gpt_model_path` and `grammar_checker_model_path` arguments, the model will not use perplexity and grammatical
correctness loss respectively, thus training only the MAPLE v1 model. Note that the `use_selector_loss` argument should
be set to `True` in the trainer. The remaining steps are the same as above.

```python
from maplev2 import MAPLEv2, MAPLEDataset, MAPLEv2Trainer

model = MAPLEv2(
    selector_type="v1",
    selector_model_path="roberta-base",
    gpt_model_path=None,
    grammar_checker_model_path=None,
    device="cuda",
)

dataset = MAPLEDataset()
dataset.load("data/data.json")

trainer = MAPLEv2Trainer(model=model, learning_rate=1e-8)

trainer.train(
    model=model,
    dataset=dataset,
    use_selector_loss=True,
    ...
)
```
