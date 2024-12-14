from dataset.prepare import DatasetPreparer
from transformers import VideoLlavaProcessor, VideoLlavaImageProcessor
from dataset.collate import DataCollatorWithPadding
from torch import tensor
import numpy as np
import torch
from datasets import load_from_disk
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from torch.utils.data import DataLoader
import os

def train():
    pass

if __name__ == "__main__":
    print(os.getcwd())
    train()