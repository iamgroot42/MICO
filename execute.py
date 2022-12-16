import os
import urllib

from torchvision.datasets.utils import download_and_extract_archive
import numpy as np
import torch
import csv

from fart_squirrel.constants import SCENARIOS, CHALLENGE, LEN_TRAINING, LEN_CHALLENGE

from torch.autograd import Variable
from sklearn import metrics
from tqdm import tqdm
from torch.distributions import normal
from torch.utils.data import DataLoader, Dataset
from mico_competition import ChallengeDataset, load_purchase100, load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from mico_competition.scoring import tpr_at_fpr, score, generate_roc, generate_table
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt
import pandas as pd
import zipfile


from fart_squirrel.attacks import ValueAttack, MetaClassifierAttack
from fart_squirrel.features import get_loss_values, get_gradient_update_norms, neighborhood_loss_robustness_weighted, gradient_updates_and_robustness


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


def check_train_performance():
    all_scores = {}
    phases = ['train']

    for scenario in tqdm(SCENARIOS, desc="scenario"):
        all_scores[scenario] = {}
        for phase in tqdm(phases, desc="phase"):
            predictions = []
            solutions  = []

            root = os.path.join(CHALLENGE, scenario, phase)
            for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
                path = os.path.join(root, model_folder)
                predictions.append(np.loadtxt(os.path.join(path, "prediction.csv"), delimiter=","))
                solutions.append(np.loadtxt(os.path.join(path, "solution.csv"),   delimiter=","))

            predictions = np.concatenate(predictions)
            solutions = np.concatenate(solutions)
        
            scores = score(solutions, predictions)
            all_scores[scenario][phase] = scores
    
    return all_scores


def check_train_performance_meta(attacker):
    all_scores = {}
    for scenario in tqdm(SCENARIOS, desc="scenario"):
        all_scores[scenario] = {}
        predictions = attacker.test_preds[scenario]
        solutions = attacker.y_test[scenario]
        
        scores = score(solutions, predictions)
        all_scores[scenario]["train_val"] = scores

    return all_scores


def plot_allscores(all_scores, focus='train'):
    for scenario in SCENARIOS:
        fpr = all_scores[scenario][focus]['fpr']
        tpr = all_scores[scenario][focus]['tpr']
        fig = generate_roc(fpr, tpr)
        fig.suptitle(f"{scenario}", x=-0.1, y=0.5)
        fig.tight_layout(pad=1.0)
        fig.savefig(f"plots/{scenario}.png")


def print_allscores_tables(all_scores, focus='train'):
    for scenario in SCENARIOS:
        print(scenario)
        scores = all_scores[scenario][focus]
        scores.pop('fpr', None)
        scores.pop('tpr', None)
        print(pd.DataFrame([scores]))


if __name__ == "__main__":
    phases = ['dev', 'final', 'train']

    dataset = load_purchase100(dataset_dir="/u/as9rw/work/MICO/data")
    # feature_extractor = get_gradient_update_norms
    feature_extractor = gradient_updates_and_robustness
    # attacker = MetaClassifierAttack(dataset, feature_extractor)
    # feature_extractor = neighborhood_loss_robustness_weighted
    attacker = ValueAttack(dataset, feature_extractor)
    attacker.train()

    for scenario in tqdm(SCENARIOS, desc="scenario"):
        for phase in tqdm(phases, desc="phase"):
            root = os.path.join(CHALLENGE, scenario, phase)
            for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
                path = os.path.join(root, model_folder)
                challenge_dataset = ChallengeDataset.from_path(
                    path, dataset=dataset, len_training=LEN_TRAINING)
                challenge_points = challenge_dataset.get_challenges()

                model = load_model('purchase100', path)
                model.cuda()
                challenge_dataloader = torch.utils.data.DataLoader(
                    challenge_points, batch_size=2*LEN_CHALLENGE)
                features, labels = next(iter(challenge_dataloader))
                features, labels = features.cuda(), labels.cuda()

                predictions = attacker.get_predictions(scenario, model, features, labels)
                assert np.all((0 <= predictions) & (predictions <= 1))

            with open(os.path.join(path, "prediction.csv"), "w") as f:
                csv.writer(f).writerow(predictions)

    pack_results_as_file("loss_test")
    scores = check_train_performance()
    plot_allscores(scores)
    print_allscores_tables(scores)
    # scores = check_train_performance_meta(attacker)
    # plot_allscores(scores, focus='train_val')
    # print_allscores_tables(scores, focus='train_val')
