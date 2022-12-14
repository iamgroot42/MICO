"""
    Helper functions for feature extraction.
"""
import torch as ch
import numpy as np
from torch.distributions import normal
from typing import List


def get_gradient_update_norms(model, features, labels, n_steps: int=10, lr:float=0.001):
    """
        Perform plain GD steps for given point on model and make note of 
        layer-wise gradient norms. These
    """
    lr = 0.001
    criterion = ch.nn.CrossEntropyLoss(reduction='none')
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
            features_inside.append([ch.linalg.norm(x.grad.detach()).item() for x in model.parameters()])
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


def get_loss_values(model, features, labels):
    """
        Get classification loss for given points
    """
    criterion = ch.nn.CrossEntropyLoss(reduction='none')
    output = model(features)
    predictions = -criterion(output, labels).detach().numpy()
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
