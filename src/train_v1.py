# Train a MAPLEv1 model

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from maplev2 import MAPLEv2, MAPLEDataset, MAPLEv2Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MAPLEDataset()
dataset.load("data/data.json")

model = MAPLEv2(
    selector_type="v1",
    selector_model_path="roberta-base",
    gpt_model_path=None,
    grammar_checker_model_path=None,
    device=device
).to(device)

trainer = MAPLEv2Trainer(
    model=model,
    learning_rate=1e-5
)

trainer.train(
    dataset=dataset,
    model=model,
    batch_size=3,
    epochs=10,
    alpha=10,
    use_tensorboard=True,
    generate_every=500,
    upload_model_to_hub=True,
    use_selector_loss=True,
    use_absolute_selector_loss=False,
    save_every=1,
    save_latest=True,
    save_dir="./saved_models",
    hub_model_name="roberta-base-maplev1",
    hub_organization="maple-v2",
    auth_token="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"
)
