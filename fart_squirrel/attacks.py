from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from fart_squirrel.constants import CHALLENGE, LEN_CHALLENGE, LEN_TRAINING, SCENARIOS
from mico_competition import ChallengeDataset, load_purchase100, load_model
from tqdm import tqdm
from mico_competition.scoring import tpr_at_fpr, score, generate_roc, generate_table
from sklearn.metrics import roc_curve, roc_auc_score
import os
from typing import List
import torch.nn as nn
import torch as ch
import numpy as np

from fart_squirrel.features import normalize_preds


class Attack:
    def __init__(self, dataset, feature_extractor):
        self.dataset = dataset
        self.feature_extractor = feature_extractor

    def train(self):
        pass
    
    def get_predictions(self, scenario, model, features, labels):
        raise NotImplementedError


class ValueAttack(Attack):
    def __init__(self, dataset, feature_extractor):
        super().__init__(dataset, feature_extractor)
    
    def get_predictions(self, scenario, model, features, labels):
        processed_features = self.feature_extractor(model, features, labels)
        # Normalize scores
        preds = normalize_preds(processed_features)
        return 1 - preds
        return preds


class MetaClassifierAttack(Attack):
    def collect_training_data(self):
        collected_features = {x:[] for x in SCENARIOS}
        collected_labels = {x:[] for x in SCENARIOS}
        phase = "train"
        for scenario in tqdm(SCENARIOS, desc="scenario"):
            root = os.path.join(CHALLENGE, scenario, phase)
            for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
                path = os.path.join(root, model_folder)
                challenge_dataset = ChallengeDataset.from_path(path, dataset=self.dataset, len_training=LEN_TRAINING)
                challenge_points = challenge_dataset.get_challenges()
            
                model = load_model('purchase100', path)
                model.cuda()
                challenge_dataloader = ch.utils.data.DataLoader(challenge_points, batch_size=2*LEN_CHALLENGE)
                features, labels = next(iter(challenge_dataloader))
                features, labels = features.cuda(), labels.cuda()

                # Get features
                processed_features = self.feature_extractor(model, features, labels)

                # Collect features
                collected_features[scenario].append(processed_features)
            
                # Get labels for membership
                collected_labels[scenario].append(challenge_dataset.get_solutions())

        for sc in SCENARIOS:
            collected_features[sc] = np.concatenate(collected_features[sc], 0)
            collected_labels[sc] = np.concatenate(collected_labels[sc], 0)

        return collected_features, collected_labels

    def train(self):
        X_for_meta, Y_for_meta = self.collect_training_data()
        self.meta_clfs = {x: RandomForestClassifier(max_depth=8) for x in SCENARIOS}
        self.test_preds = {}

        # Make a 80:20 split for training and validation
        self.X_train = {x:models[:80] for x, models in X_for_meta.items()}
        self.X_test = {x:models[80:] for x, models in X_for_meta.items()}
        self.y_train = {x:models[:80] for x, models in Y_for_meta.items()}
        self.y_test = {x:models[80:] for x, models in Y_for_meta.items()}

        for sc in SCENARIOS:
            self.meta_clfs[sc].fit(self.X_train[sc], self.y_train[sc])
            self.test_preds[sc] = self.meta_clfs[sc].predict_proba(self.X_test[sc])[:, 1]

            print(f"{sc} AUC: {roc_auc_score(self.y_test[sc], self.test_preds[sc])}")

    def get_predictions(self, scenario, model, features, labels):
        processed_features = self.feature_extractor(model, features, labels)
        predictions = self.meta_clfs[scenario].predict_proba(processed_features)[:, 1]
        return predictions
