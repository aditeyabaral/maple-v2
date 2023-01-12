# Train a MAPLEv2 model

import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from maplev2 import MAPLEv2, MAPLEDataset, MAPLEv2Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MAPLEDataset()
dataset.load("data/data.json")

model = MAPLEv2(
    selector_type="context",
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
    alpha=1,
    beta=0.0002,
    gamma=0.5,
    use_tensorboard=True,
    generate_every=1000,
    upload_model_to_hub=True,
    use_context_selector_loss=True,
    use_absolute_context_selector_loss=False,
    save_every=1,
    save_latest=True,
    save_dir="./saved_models",
    hub_model_name="roberta-base-maple",
    hub_organization="maple",
    auth_token="your_auth_token"
)
