"""
    Helper functions for feature extraction.
"""
import torch as ch
import torch.nn as nn
import numpy as np
from torch.distributions import normal
from typing import List, Tuple
from torch.autograd import Variable


def normalize_preds(preds):
    # Normalize to unit interval
    min_prediction = np.min(preds)
    max_prediction = np.max(preds)
    preds = (preds - min_prediction) / (max_prediction - min_prediction)
    return preds


def ascent_recovery(model, features, labels):
    """
        Perform gradient ascent on given datapoint to minimize loss
        Check reduction in loss.
        Intuition: if already seem by model, reduction would be less than when
        not seen by model
    """
    criterion = ch.nn.CrossEntropyLoss(reduction='none')
    n_times = 10
    step_size = 0.001
    final_losses = []
    for feature, label in zip(features, labels):
        model.zero_grad()
        feature_var = Variable(feature.clone().detach(), requires_grad=True)
        for _ in range(n_times):
            feature_var = Variable(feature_var.clone().detach(), requires_grad=True)
            output = model(feature_var)
            loss = criterion(ch.unsqueeze(output, 0), ch.unsqueeze(label, 0))
            loss.backward(ch.ones_like(loss), retain_graph=True)
            with ch.no_grad():
                feature_var.data -= step_size * feature_var.data
                loss_new = criterion(ch.unsqueeze(model(feature_var), 0), ch.unsqueeze(label, 0))
        final_losses.append(loss.item() - loss_new.item())
    final_losses = np.array(final_losses)
    return final_losses


# def get_gradient_update_norms(model, features, labels, n_steps: int=10, lr:float=0.0005):
def get_gradient_update_norms(model, features, labels, n_steps: int=5, lr:float=0.0005):
    """
        Perform plain GD steps for given point on model and make note of 
        layer-wise gradient norms. These
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    features_collected = []
    for feature, label in zip(features, labels):
        features_inside = []
        optimizer = ch.optim.SGD(model.parameters(), lr=lr, momentum=0)
        for _ in range(n_steps):
            optimizer.zero_grad()
            output = model(feature)
            loss = criterion(ch.unsqueeze(output, 0), ch.unsqueeze(label, 0))
            loss.backward()
            optimizer.step()
            features_inside.append([ch.linalg.norm(x.grad.detach().cpu()).item() for x in model.parameters()])
        features_collected.append(np.array(features_inside))
    features_collected = np.array(features_collected)
    return features_collected.reshape(features_collected.shape[0], -1)


def get_gradient_norm(model, features, labels):
    """
        Compute gradient for model weights for given data
    """
    return get_gradient_update_norms(model, features, labels, n_steps=1, lr=1.0)


@ch.no_grad()
def neighborhood_robustness(model, features, epsilon: float, n_neighbors: int):
    """
        Sample neighbors of given data within given radius, and compute
        robustness in model predictions (based on norm)
    """
    noise = normal.Normal(0, epsilon)
    l2_diffs = []
    base_preds = model(features).cpu().numpy()
    for i, feature in enumerate(features):
        neighbors = []
        for _ in range(n_neighbors):
            neighbors.append(feature + noise.sample(feature.shape).to(feature.device))
        neighbors = ch.stack(neighbors, 0)
        prediction = model(neighbors).cpu().numpy()
        # Use L2 to check robustness
        predictions_diff = np.linalg.norm(prediction - base_preds[i], axis=1)
        l2_diffs.append(np.mean(predictions_diff))
    return np.array(l2_diffs)


@ch.no_grad()
def neighborhood_loss_robustness_weighted(model, features, labels, epsilon: float = 0.2, n_neighbors: int = 100):
# def neighborhood_loss_robustness_weighted(model, features, labels, epsilon: float = 0.2, n_neighbors: int = 20):
    """
        Got 0.1217 score with 0.2, 20 neighbors
        Local AUCs were 0.69, 0.69, 0.72

        Got 0.124 score with 0.2, 20 negibhors and focusing only on logit corresponding to ground truth
        Local AUCs were 0.69, 0.69, 0.72

        # Got 0.124 score with 0.2, 100 neighbors
        Local AUCs were 0.69, 0.69, 0.71

        # Got 0.119 score with 0.01, 100 neighbors
        Local AUCs were 0.69, 0.69, 0.72
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    noise = normal.Normal(0, epsilon)
    l2_diffs = []
    base_preds = model(features)
    base_losses = criterion(base_preds, labels).cpu().numpy()
    base_preds = base_preds.cpu().numpy()
    for i, feature in enumerate(features):
        neighbors = []
        distances = []
        for _ in range(n_neighbors):
            sampled_noise = noise.sample(feature.shape).to(feature.device)
            neighbors.append(feature + sampled_noise)
            distances.append(sampled_noise.mean().cpu().item())
        neighbors = ch.stack(neighbors, 0)
        prediction = model(neighbors)
        pred_diff = prediction.cpu().numpy() - base_preds[i]

        # Overall L2 norm of difference in logits
        # predictions_diff = np.linalg.norm(pred_diff, axis=1)

        # Only focus on logit corresponding to ground truth
        predictions_diff = np.linalg.norm(pred_diff[:, labels[i]])

        distances = np.array(distances)
        l2_diffs.append(np.mean(predictions_diff * distances)) # Weighted with distances
    return np.array(l2_diffs)


def get_loss_values(model, features, labels):
    """
        Get classification loss for given points
    """
    criterion = ch.nn.CrossEntropyLoss(reduction='none')
    output = model(features)
    predictions = -criterion(output, labels).detach().cpu().numpy()
    return predictions


def feature_combine(model, features, labels, extraction_functions):
    """
        Generic function to combine features from different functions
    """
    features_collect = []

    for extraction_function in extraction_functions:
        features_collect.append(extraction_function(model, features, labels))
    
    # Return collected features
    features_collect = np.array(features_collect).T

    return features_collect


def neighborhood_and_loss(model, features, labels, epsilon_list: List[float] = [0.01], neighbor_list: List[int] = [20]):
    """
        Get neighborhood robustness and loss values for given model
    """
    functors = [get_loss_values]

    # Get neighborhood robustness for given range
    for epsilon, neighbors in zip(epsilon_list, neighbor_list):
        def functor(model, features, labels):
            return neighborhood_robustness(model, features, epsilon, neighbors)
        functors.append(functor)

    # Return collected features
    return feature_combine(model, features, labels, functors)


def gradient_updates_and_robustness(model, features, labels):
    """
        Get neighborhood robustness and loss values for given model
    """
    features_1 = neighborhood_loss_robustness_weighted(model, features, labels)
    # features_2 = get_gradient_update_norms(model, features, labels)
    
    # Got 0.120 score when combining directly
    features_2 = get_gradient_norm(model, features, labels)
    features_2, features_3 = features_2[:, 0], features_2[:, 2]
    features_1 = normalize_preds(features_1)
    features_2 = normalize_preds(features_2)
    features_3 = normalize_preds(features_3)
    return features_1 + features_2 + features_3

    features_1 = features_1.reshape(-1, 1)
    combined_feratures = np.concatenate((features_1, features_2), 1)
    return combined_feratures


def ascent_loss_norm(model, features, labels):
    features_1 = ascent_recovery(model, features, labels)
    features_2 = get_gradient_norm(model, features, labels)
    features_2 = features_2[:, 0].reshape(-1, 1)
    features_3 = get_loss_values(model, features, labels).reshape(-1, 1)
    combined_feratures = np.concatenate((features_1, features_2, features_3), 1)
    return combined_feratures
