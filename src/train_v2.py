# Train a MAPLEv2 model

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from maplev2 import MAPLEv2, MAPLEDataset, MAPLEv2Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MAPLEDataset()
dataset.load("data/data.json")

model = MAPLEv2(
    selector_type="v2",
    selector_model_path="roberta-base",
    selector_mode="whole-word",
    gpt_model_path="gpt2",
    freeze_gpt=True,
    grammar_checker_model_path="jxuhf/roberta-base-finetuned-cola",
    freeze_grammar_checker=True,
    device=device
).to(device)

trainer = MAPLEv2Trainer(
    model=model,
    learning_rate=1e-8
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
    upload_model_to_hub=False,
    use_selector_loss=True,
    use_absolute_selector_loss=False,
    save_every=1,
    save_latest=True,
    save_dir="./saved_models",
    hub_model_name="roberta-base-maplev2",
    hub_organization="maple-v2",
    auth_token="your-auth-token"
)
