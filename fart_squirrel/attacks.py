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
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from typing import List
import torch.nn as nn
import torch as ch
import numpy as np
import copy

from mico_competition import MLP

from fart_squirrel.features import normalize_preds, get_loss_values
from fart_squirrel.train import train_model, n_datasets_generator


class Attack:
    def __init__(self, dataset, feature_extractor):
        self.dataset = dataset
        self.feature_extractor = feature_extractor

    def train(self):
        pass
    
    def get_predictions(self, scenario, model, features, labels, misc_dict = None):
        raise NotImplementedError


class ValueAttack(Attack):
    def __init__(self, dataset, feature_extractor):
        super().__init__(dataset, feature_extractor)
    
    def get_predictions(self, scenario, model, features, labels, misc_dict=None):
        features, labels = features.cuda(), labels.cuda()
        processed_features = self.feature_extractor(model, features, labels)
        # Normalize scores
        preds = normalize_preds(processed_features)
        # return 1 - preds
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

    def get_predictions(self, scenario, model, features, labels, misc_dict = None):
        features, labels = features.cuda(), labels.cuda()
        processed_features = self.feature_extractor(model, features, labels)
        predictions = self.meta_clfs[scenario].predict_proba(processed_features)[:, 1]
        return predictions


class ShadowAttack(Attack):
    def train(self):
        if os.path.exists("shadow_information.npy"):
            self.shadow_information = np.load("shadow_information.npy", allow_pickle=True).item()
        else:
            self.shadow_information = self._build_shadow_information()
            np.save("shadow_information.npy", self.shadow_information)

    def _build_shadow_information(self):
        phases = ["dev", "final"]
        ranked_lists = {}
        train_model_paths = {}
        # Make note of model paths
        for scenario in SCENARIOS:
            train_root = os.path.join(CHALLENGE, scenario, "train")
            train_model_paths[scenario] = sorted(os.listdir(train_root), key=lambda d: int(d.split('_')[1]))

        # Get loss values for each train model in this scenario
        for scenario in tqdm(SCENARIOS, desc="scenario"):
            ranked_lists[scenario] = {}
            for i, train_model_folder in tqdm(enumerate(train_model_paths[scenario]), desc="model_train", total=len(train_model_paths[scenario])):
                train_root = os.path.join(CHALLENGE, scenario, "train")
                ranked_lists[scenario][i] = {}
                model = load_model('purchase100', path=os.path.join(
                    train_root, train_model_folder))
                model.cuda()
                for phase in phases:
                    ranked_lists[scenario][i][phase] = []
                    root = os.path.join(CHALLENGE, scenario, phase)
                    for j, model_folder in enumerate(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1]))):
                        path = os.path.join(root, model_folder)
                        challenge_dataset = ChallengeDataset.from_path(
                            path, dataset=self.dataset, len_training=LEN_TRAINING)
                        challenge_points = challenge_dataset.get_challenges()

                        # Load challenge points for this phase + scenario
                        challenge_dataloader = ch.utils.data.DataLoader(
                            challenge_points, batch_size=2*LEN_CHALLENGE)
                        features, labels = next(iter(challenge_dataloader))
                        features, labels = features.cuda(), labels.cuda()

                        # Get features
                        processed_features = get_loss_values(
                            model, features, labels)
                        # Store features
                        ranked_lists[scenario][i][phase].append(processed_features)
                    ranked_lists[scenario][i][phase] = np.array(ranked_lists[scenario][i][phase])

        # Conver to useful format
        ranked_lists_other = {}
        for phase in tqdm(phases, desc="phase"):
            ranked_lists_other[phase] = {}
            for scenario in tqdm(SCENARIOS, desc="scenario"):
                ranked_lists_other[phase][scenario] = {}
                # Models per scenario+phase
                n_other_models = ranked_lists[scenario][0][phase].shape[0]
                n_datapoints = ranked_lists[scenario][0][phase].shape[1]
                for i in range(n_other_models):
                    ranked_lists_other[phase][scenario][i] = []
                    # Data within that
                    for j in range(n_datapoints):
                        desired_list = [ranked_lists[scenario][k][phase][i][j] for k in range(len(ranked_lists[scenario]))]
                        desired_list = np.argsort(desired_list)
                        ranked_lists_other[phase][scenario][i].append([train_model_paths[scenario][x] for x in desired_list])
        return ranked_lists_other
    
    def _update_model_weights(self, model, feature, label):
        """
            Perform gradient ascent on given datapoint to minimize loss
            Check reduction in loss.
            Intuition: if already seem by model, reduction would be less than when
            not seen by model
        """
        model_ = copy.deepcopy(model)
        model_.train()
        criterion = nn.CrossEntropyLoss()
        lr = 0.0005
        n_steps = 5
        optimizer = ch.optim.SGD(model_.parameters(), lr=lr, momentum=0)
        for _ in range(n_steps):
            optimizer.zero_grad()
            output = model(feature)
            loss = criterion(ch.unsqueeze(output, 0), ch.unsqueeze(label, 0))
            loss.backward()
            optimizer.step()
        model_.eval()
        return model_

    def get_predictions(self, scenario, model, features, labels, misc_dict=None):
        # Take note of phase, index of model
        phase = misc_dict["phase"]
        model_index = misc_dict["model_index"]
        relevant_dict = self.shadow_information[phase][scenario][model_index]

        features, labels = features.cuda(), labels.cuda()
        m_models = 35 # Use 25 least-likely-containing-member models

        # Get features corresponding to the given model
        processed_features = self.feature_extractor(model, features, labels)
        predictions = []
        for i, (feature, label) in enumerate(zip(features, labels)):
            X_, Y_ = [], []
            relevant_models = np.array(relevant_dict[i])[:m_models]
            # Get relevant train models
            for model_path in relevant_models:
                train_model = MLP.load(os.path.join(
                    CHALLENGE, scenario, "train", model_path, "model.pt"))
                train_model.cuda()
                # Fine-tune 
                train_model_finetuned = self._update_model_weights(train_model, feature, label)
                # Get features
                ex_f = self.feature_extractor(train_model, feature.view(1, -1), label.view(1))
                ex_f_finetuned = self.feature_extractor(train_model_finetuned, feature.view(1, -1), label.view(1))
                X_.append(ex_f)
                X_.append(ex_f_finetuned)
                Y_.append(0)
                Y_.append(1)

            # Train a meta-classifier on this
            X_, Y_ = np.array(X_), np.array(Y_)
            meta_clf = RandomForestClassifier(max_depth=3)
            meta_clf.fit(X_, Y_)
            # Make predictions on the given model
            predictions.append(meta_clf.predict_proba(processed_features[i].reshape(1, -1))[0, 1])

        return np.array(predictions)


def collect_from_batch(loader):
    X, Y = [], []
    for batch in loader:
        X.append(batch[0])
        Y.append(batch[1])
    X = ch.cat(X, 0)
    Y = ch.cat(Y, 0)
    return X, Y
