import torch
import os
import hydra
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import wandb
from torch import nn
from models.model import MyAwesomeModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import logging
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config_training")
def train(cfg):
    torch.manual_seed(cfg.seed)
    log.info(f"Training with config: {cfg}")

    train_set = torch.load(os.path.join(cfg.dataset_path, "train_images.pt"))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size)

    test_set = torch.load(os.path.join(cfg.dataset_path, "test_images.pt"))
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size)


    model = MyAwesomeModel()
    trainer = Trainer(max_epochs=cfg.num_epochs,
                      default_root_dir=os.getcwd(),
                      logger=pl.loggers.WandbLogger(project="mnist_mlops"),
                      limit_train_batches=0.2,
                      )
    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)

    torch.save(model, f"{os.getcwd()}/trained_model.pt")
    log.info("Model saved")

if __name__ == '__main__':
    train()
