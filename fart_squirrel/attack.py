import os
import urllib

from torchvision.datasets.utils import download_and_extract_archive
import numpy as np
import torch
import csv

from torch.autograd import Variable
from sklearn import metrics
from tqdm.notebook import tqdm
from torch.distributions import normal
from torch.utils.data import DataLoader, Dataset
from mico_competition import ChallengeDataset, load_purchase100, load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from mico_competition.scoring import tpr_at_fpr, score, generate_roc, generate_table
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import zipfile


SCENARIOS = ["purchase100_inf", "purchase100_hi", "purchase100_lo"]


def pack_results_as_file(experiment_name: str):
    phases = ['dev', 'final']

    with zipfile.ZipFile(f"submissions/{experiment_name}.zip", 'w') as zipf:
        for scenario in tqdm(SCENARIOS, desc="scenario"): 
            for phase in tqdm(phases, desc="phase"):
                root = os.path.join("purchase100", scenario, phase)
                for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
                    path = os.path.join(root, model_folder)
                    file = os.path.join(path, "prediction.csv")
                    if os.path.exists(file):
                        zipf.write(file)
                    else:
                        raise FileNotFoundError(f"`prediction.csv` not found in {path}. You need to provide predictions for all challenges")
