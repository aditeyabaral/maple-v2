import os
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AdamW

from .model import MAPLEv2
from .dataset import MAPLEDataset


class MAPLEv2Trainer:
    def __init__(self, model, learning_rate, **kwargs):
        self.optimizer = None
        self.scheduler = None
        self.optimizer_class = kwargs.get("optimizer_class", "adamw")
        self.scheduler_class = kwargs.get("scheduler_class", "plateau")
        self.tensorboard_writer = None

        self.init_optimizer(model, learning_rate, **kwargs)

    def init_optimizer(self, model, learning_rate, **kwargs):
        optimizer_class = kwargs.get("optimizer_class", "adamw")
        if optimizer_class == "adamw":
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_class == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_class == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_class == 'adagrad':
            self.optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        elif optimizer_class == 'adadelta':
            self.optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
        elif optimizer_class == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        elif optimizer_class == 'sparse_adam':
            self.optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError

        self.init_scheduler(**kwargs)

    def init_scheduler(self, **kwargs):
        scheduler_class = kwargs.get("scheduler_class", "plateau")
        if scheduler_class == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=1,
                verbose=True,
            )
        elif scheduler_class == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get("step_size", 1),
                gamma=kwargs.get("gamma", 0.1),
            )
        elif scheduler_class == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get("T_max", 10),
                eta_min=kwargs.get("eta_min", 0.0001),
            )
        elif scheduler_class == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=kwargs.get("gamma", 0.1),
            )
        else:
            raise NotImplementedError

    def init_tensorboard(self, tensorboard_log_dir):
        if Path(tensorboard_log_dir).exists():
            for file in Path(tensorboard_log_dir).iterdir():
                file.unlink()
        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir)

    def write_losses_to_tensorboard(self, losses, step):
        for key, value in losses.items():
            if key.startswith("loss") and value is not None and isinstance(value, torch.Tensor):
                self.tensorboard_writer.add_scalar(f"Loss/{key}", value, step)

    def write_generated_poems_to_tensorboard(self, passages, poems, step):
        for passage, poem in zip(passages, poems):
            self.tensorboard_writer.add_text(f"{passage[:50]}", poem, step)

    @staticmethod
    def compute_loss(outputs, **kwargs):
        # loss = alpha * loss_s + beta * loss + gamma * loss
        loss = (
                kwargs.get("alpha", 1) * outputs.__dict__.get("loss_s", 0) +
                kwargs.get("beta", 0.0002) * outputs.__dict__.get("loss_ppl", 0) +
                kwargs.get("gamma", 0.5) * outputs.__dict__.get("loss_g", 0)
        )
        return {"loss": loss}

    def train(
            self,
            dataset: MAPLEDataset,
            model: MAPLEv2,
            batch_size,
            epochs,
            save_dir="./saved_models",
            **kwargs

    ):
        if epochs <= 0:
            return

        if kwargs.get("use_tensorboard", True):
            self.init_tensorboard(kwargs.get("tensorboard_log_dir", "./logs"))

        if kwargs.get("learning_rate", None) is not None:
            learning_rate = kwargs.get("learning_rate")
            self.optimizer.param_groups[0]["lr"] = learning_rate
            self.init_scheduler(**kwargs)

        if kwargs.get("upload_model_to_hub", False):
            self.clone_hub_repository_into_save_dir(save_dir, **kwargs)

        generate_every = kwargs.get("generate_every", 100)

        print(f"Training for {epochs} epochs with learning rate {self.optimizer.param_groups[0]['lr']}")

        model.train()
        model.zero_grad()
        torch.cuda.empty_cache()
        total_examples = len(dataset)
        steps = 0
        for epoch in tqdm(range(1, epochs + 1)):
            num_batches = 0
            total_epoch_loss = 0
            for batch_idx in tqdm(range(0, total_examples, batch_size)):
                self.optimizer.zero_grad()
                batch_passages, batch_tokens, batch_labels = dataset.get_batch(batch_idx, batch_size)
                outputs = model(
                    passages=batch_passages,
                    tokens=batch_tokens,
                    labels=batch_labels
                )
                if hasattr(outputs, "loss_s") and \
                        outputs.loss_s is not None and \
                        kwargs.get("use_absolute_context_selector_loss", True) and \
                        model.selector_type == "context":
                    outputs.loss_s = torch.abs(outputs.loss_s)

                generate_poems_flag = generate_every if \
                    isinstance(generate_every, bool) else (
                        batch_idx % (generate_every * batch_size) == 0)
                if generate_poems_flag:
                    self.write_losses_to_tensorboard(outputs.__dict__, steps)
                    if hasattr(outputs, "generated_sequences"):
                        self.write_generated_poems_to_tensorboard(batch_passages, outputs.generated_sequences, steps)

                if kwargs.get("use_context_selector_loss", True):
                    del outputs.loss_s

                losses = self.compute_loss(outputs, **kwargs)
                if self.tensorboard_writer is not None:
                    self.write_losses_to_tensorboard(losses, steps)

                for loss in losses.values():
                    if isinstance(loss, torch.Tensor):
                        loss.backward()

                total_epoch_loss += sum(losses.values())
                self.optimizer.step()
                steps += 1
                num_batches += 1
            total_epoch_loss = total_epoch_loss / num_batches
            self.scheduler.step(total_epoch_loss)
            print(f"Epoch {epoch} completed. Training Loss: {total_epoch_loss}")
            print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")

            self.push_training_state_to_hub(model, epoch, **kwargs)
            if epoch % kwargs.get("save_every", 1) == 0:
                self.save_training_state_locally(save_dir, model, epoch, **kwargs)
                if kwargs.get("upload_model_to_hub", False):
                    self.upload_model_to_hub(save_dir, epoch, **kwargs)

        self.push_training_state_to_hub(model, epochs + 1, **kwargs)
        self.save_training_state_locally(save_dir, model, epochs + 1, **kwargs)
        if kwargs.get("upload_model_to_hub", False):
            self.upload_model_to_hub(save_dir, epochs + 1, **kwargs)

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    def save(self, path):
        if not Path(path).parent.exists():
            raise FileNotFoundError
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "optimizer_class": self.optimizer_class,
                "scheduler_class": self.scheduler_class,
            }, path
        )

    def load(self, path):
        if not Path(path).exists():
            raise FileNotFoundError
        checkpoint = torch.load(path)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer_class = checkpoint["optimizer_class"]
        self.scheduler_class = checkpoint["scheduler_class"]

    def save_training_state_locally(self, path, model, epoch, **kwargs):
        if not Path(path).exists():
            Path(path).mkdir(parents=True)

        save_model_name = kwargs.get("save_model_name", "model.pt")
        if not kwargs.get("save_latest", False):
            base_model_name = kwargs.get("save_model_name", "model.pt")
            epoch = str(epoch).zfill(2)
            save_model_name = f"{base_model_name}_{epoch}.pt"
        model_save_path = Path(path).joinpath(save_model_name)
        trainer_save_path = Path(path).joinpath("trainer.pt")

        model.save(model_save_path)
        self.save(trainer_save_path)

    @staticmethod
    def push_training_state_to_hub(model: MAPLEv2, epoch, **kwargs):
        if kwargs.get("auth_token", None) is None or kwargs.get("hub_model_name", None) is None:
            return

        repo_name = kwargs["hub_model_name"]
        if kwargs.get("hub_organization", None):
            repo_name = f"{kwargs['hub_organization']}/{kwargs['hub_model_name']}"

        model.push_to_hub(
            repo_name=repo_name,
            commit_message=f"Epoch {epoch}",
            auth_token=kwargs.get("auth_token")
        )

    @staticmethod
    def upload_model_to_hub(path, epoch, **kwargs):
        if not Path(path).exists() or kwargs.get("auth_token", None) is None:
            return
        path = str(Path(path).absolute().resolve())
        command = f"cd \"{path}\" && git lfs install && huggingface-cli lfs-enable-largefiles . && git-lfs pull && git gc && git add . && git commit -m \"Model: Epoch {epoch}\" && git push"
        os.system(command)

    @staticmethod
    def clone_hub_repository_into_save_dir(save_dir, **kwargs):
        if kwargs.get("auth_token", None) is None or kwargs.get("hub_model_name", None) is None:
            return
        auth_token = kwargs.get("auth_token")
        hub_model_name = kwargs.get("hub_model_name")
        hub_username = kwargs.get("hub_username", None)
        if hub_username is None:
            hub_username = os.popen("huggingface-cli whoami").read().strip().split('\n')[0]
            print(f"Using username {hub_username} for hub model {hub_model_name}")
        hub_organization = kwargs.get("hub_organization", "")
        hub_organization_flag = ""
        if hub_organization:
            hub_organization_flag = f"--organization {hub_organization}"

        print(f"Creating repository {hub_model_name}")
        p = subprocess.run(
            [
                "huggingface-cli",
                "repo",
                "create",
                hub_model_name,
                *hub_organization_flag.split(),
                "-y",
            ],
            capture_output=True,
        )
        print(p.stdout.decode("utf-8"))

        print(f"Cloning repository {hub_model_name} to {save_dir}")
        if hub_organization:
            clone_url = f"https://{hub_organization}:{auth_token}@huggingface.co/{hub_organization}/{hub_model_name}"
        else:
            clone_url = f"https://{hub_username}:{auth_token}@huggingface.co/{hub_username}/{hub_model_name}"

        p = subprocess.run(
            [
                "git",
                "clone",
                clone_url,
                save_dir
            ],
            capture_output=True
        )
        print(p.stdout.decode("utf-8"))

        while not Path(save_dir).exists():
            time.sleep(10)
