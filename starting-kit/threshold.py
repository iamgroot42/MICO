import numpy as np
import torch as ch


def linear_itp_threshold_func(
        distribution: List[float],
        alpha: List[float],
        signal_min=0,
        signal_max=1000,
        **kwargs
) -> float:
    distribution = np.append(distribution, signal_min)
    distribution = np.append(distribution, signal_max)
    threshold = np.quantile(distribution, q=alpha, interpolation='linear',**kwargs)

    return threshold


def logit_rescale_threshold_func(
        distribution: List[float],
        alpha: List[float],
        **kwargs,
) -> float:
    distribution = np.log(
        np.divide(np.exp(- distribution), (1 - np.exp(- distribution))))
    len_dist = len(distribution)
    loc, scale = norm.fit(distribution)

    threshold = norm.ppf(1 - np.array(alpha), loc=loc, scale=scale)
    threshold = np.log(np.exp(threshold) + 1) - threshold
    return threshold


def min_linear_logit_threshold_func(
        distribution: List[float],
        alpha: List[float],
        signal_min=0,
        signal_max=1000,
        **kwargs,
) -> float:
    distribution_linear = np.append(distribution, signal_min)
    distribution_linear = np.append(distribution_linear, signal_max)
    threshold_linear = np.quantile(distribution_linear, q=alpha, interpolation='linear',**kwargs,)

    # Clip loss values (should not be too close to zero)
    min_loss = 1e-7
    distribution[distribution < min_loss] = min_loss
    distribution = np.log(np.divide(np.exp(- distribution), (1 - np.exp(- distribution))))
    len_dist = len(distribution)
    loc, scale = norm.fit(distribution,**kwargs,)
    threshold_logit = norm.ppf(1 - alpha, loc=loc, scale=scale)
    threshold_logit = np.log(np.exp(threshold_logit) + 1) - threshold_logit

    threshold = min(threshold_logit, threshold_linear)

    return threshold
