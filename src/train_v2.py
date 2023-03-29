# Train a MAPLEv2 model
# TODO: Update this file
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from maplev2 import MAPLEv2, MAPLEDataset, MAPLEv2Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MAPLEDataset()
dataset.load("data/poetry_foundation.csv")

model = MAPLEv2(
    selector_type="v2",
    selector_model_path="allenai/longformer-base-4096",
    selector_mode="whole-word",
    gpt_model_path="gpt2",
    freeze_gpt=True,
    grammar_checker_model_path="jxuhf/roberta-base-finetuned-cola",
    freeze_grammar_checker=True,
    device=device
).to(device)

trainer = MAPLEv2Trainer(
    model=model,
    learning_rate=1e-5
)

trainer.train(
    dataset=dataset,
    model=model,
    batch_size=2,
    epochs=5,
    alpha=10,
    beta=0.002,
    gamma=0.05,
    use_tensorboard=True,
    generate_every=500,
    upload_model_to_hub=True,
    use_selector_loss=True,
    use_absolute_selector_loss=True,
    save_every=1,
    save_latest=True,
    save_dir="./saved_models",
    hub_model_name="roberta-base-maplev2",
    hub_organization="maple-v2",
    auth_token="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"
)
