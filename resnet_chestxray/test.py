import os
from tqdm import tqdm, trange
import logging
from scipy.stats import logistic
import numpy as np
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
import csv
from scipy.special import softmax
from scipy.special import expit
import time
import cv2

import torch
import torchvision
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .main_utils import build_training_dataset

dataset = build_training_dataset(data_dir='/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_256/', img_size=256, dataset_metadata='data/test_edema.csv', label_key='edema_severity')

print(dataset, len(dataset))

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
# import ipdb
#ipdb.set_trace()

thing = next(iter(dataloader))
print(thing)
