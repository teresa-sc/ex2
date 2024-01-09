from torch import nn, optim
import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule

cfg = OmegaConf.load('ex2/config/config_model.yaml')
cfg_train = OmegaConf.load('ex2/config/config_training.yaml')
torch.manual_seed(cfg.seed)

class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                    nn.Conv2d(1, cfg.num_filters_lay1, cfg.filter_size),  # [B, 1, 28, 28] -> [B, 32, 26, 26]
                    nn.LeakyReLU(),
                    nn.Conv2d(cfg.num_filters_lay1, cfg.num_filters_lay2, cfg.filter_size), # [B, 32, 26, 26] -> [B, 64, 24, 24]
                    nn.LeakyReLU(),
                    nn.MaxPool2d(2),      # [B, 64, 24, 24] -> [B, 64, 12, 12]
                    nn.Flatten(),        # [B, 64, 12, 12] -> [B, 64 * 12 * 12]
                    nn.Linear(cfg.num_filters_lay2 * cfg.flatten_dim * cfg.flatten_dim, cfg.out_dim), # [B, 64 * 12 * 12] -> [B, 10]
                    )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg_train.lr)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')

        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss)
        acc = (y == y_pred.argmax(dim=-1)).float().mean()
        self.log('train_acc', acc)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer