from torch import nn
import hydra
import torch
from omegaconf import OmegaConf

cfg = OmegaConf.load('ex2/config/config_model.yaml')
torch.manual_seed(cfg.seed)
myawesomemodel = nn.Sequential(
    nn.Conv2d(1, cfg.num_filters_lay1, cfg.filter_size),  # [B, 1, 28, 28] -> [B, 32, 26, 26]
    nn.LeakyReLU(),
    nn.Conv2d(cfg.num_filters_lay1, cfg.num_filters_lay2, cfg.filter_size), # [B, 32, 26, 26] -> [B, 64, 24, 24]
    nn.LeakyReLU(),
    nn.MaxPool2d(2),      # [B, 64, 24, 24] -> [B, 64, 12, 12]
    nn.Flatten(),        # [B, 64, 12, 12] -> [B, 64 * 12 * 12]
    nn.Linear(cfg.num_filters_lay2 * cfg.flatten_dim * cfg.flatten_dim, cfg.out_dim), # [B, 64 * 12 * 12] -> [B, 10]
)

