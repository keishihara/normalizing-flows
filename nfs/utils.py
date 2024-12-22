import random

import numpy as np
import torch
from scipy.stats import entropy, logistic, multivariate_normal
from sklearn.neighbors import KernelDensity


def compute_histogram_2d(samples, bins=50, range=None):
    """Convert a 2D distribution into a histogram"""
    hist, edges = np.histogramdd(samples, bins=bins, range=range, density=True)
    return hist, edges


def compute_kl_divergence_2d(p_samples, q_samples, bins=50, range=None):
    """Calculate the KL divergence of 2D distributions"""
    # Approximate the distributions with histograms
    p_hist, _ = compute_histogram_2d(p_samples, bins=bins, range=range)
    q_hist, _ = compute_histogram_2d(q_samples, bins=bins, range=range)

    # Normalize the histograms to convert them into probability distributions
    p_hist = p_hist / np.sum(p_hist)
    q_hist = q_hist / np.sum(q_hist)

    # Calculate KL divergence (using scipy.stats.entropy)
    kl_div = entropy(p_hist.flatten(), q_hist.flatten())

    return kl_div


def estimate_gaussian_params(samples):
    """Estimate parameters of a Gaussian distribution"""
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=False)
    return mean, cov


def compute_kl_divergence_gaussian(zs, ps):
    """Calculate KL divergence assuming Gaussian distributions"""
    # Parameters for each distribution
    zs_mean, zs_cov = estimate_gaussian_params(zs)
    ps_mean, ps_cov = estimate_gaussian_params(ps)

    # Define the distributions
    p_dist = multivariate_normal(mean=zs_mean, cov=zs_cov)
    q_dist = multivariate_normal(mean=ps_mean, cov=ps_cov)

    # Calculate KL divergence using sampling
    z_samples = zs  # Samples used for KL calculation
    p_log_prob = p_dist.logpdf(z_samples)
    q_log_prob = q_dist.logpdf(z_samples)
    kl_div = np.mean(p_log_prob - q_log_prob)

    return kl_div


def fit_kde(samples, bandwidth=0.1):
    """Calculate probability density function using Kernel Density Estimation (KDE)"""
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(samples)
    return kde


def compute_kl_divergence_kde(zs, ps, bandwidth=0.1):
    """Calculate KL divergence using Kernel Density Estimation (KDE)"""
    # Estimate distributions using KDE
    p_kde = fit_kde(zs, bandwidth=bandwidth)
    q_kde = fit_kde(ps, bandwidth=bandwidth)

    # Calculate KL divergence
    z_samples = zs  # Samples used for KL calculation
    p_log_prob = p_kde.score_samples(z_samples)
    q_log_prob = q_kde.score_samples(z_samples)
    kl_div = np.mean(p_log_prob - q_log_prob)

    return kl_div


def estimate_logistic_params(samples):
    """Estimate parameters of a logistic distribution"""
    loc = np.mean(samples, axis=0)
    scale = np.std(samples, axis=0) * np.sqrt(3) / np.pi  # Convert standard deviation to scale
    return loc, scale


def compute_kl_divergence_logistic(zs, ps):
    """Calculate KL divergence using logistic distributions"""
    # Parameters for each distribution
    loc_zs, scale_zs = estimate_logistic_params(zs)
    loc_ps, scale_ps = estimate_logistic_params(ps)

    # Calculate KL divergence
    z_samples = zs  # Samples used for KL calculation
    p_log_prob = logistic.logpdf(z_samples, loc=loc_zs, scale=scale_zs).sum(axis=1)
    q_log_prob = logistic.logpdf(z_samples, loc=loc_ps, scale=scale_ps).sum(axis=1)
    kl_div = np.mean(p_log_prob - q_log_prob)

    return kl_div


def set_seed_everywhere(seed: int, deterministic: bool = False) -> None:
    """Set the seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
