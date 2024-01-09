import torch
import os
import hydra
from torch import nn
from models.model import myawesomemodel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import logging
log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config_training")
def train(cfg):
    torch.manual_seed(cfg.seed)
    log.info(f"Training with config: {cfg}")
    model = myawesomemodel.to(device)
    train_set = torch.load(cfg.dataset_path)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch} Loss {loss}")

    torch.save(model, f"{os.getcwd()}/trained_model.pt")
    log.info("Model saved")

if __name__ == '__main__':
    train()