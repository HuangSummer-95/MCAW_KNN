# -*- coding: utf-8 -*-
"""
MCAW-KNN: Multi-Class Adaptive Weighted K-Nearest Neighbors

A classification algorithm that combines adaptive local region construction
with class-specific feature weighting to improve KNN performance.

Author: Meimei Huang
Date: January 2026
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from scipy.linalg import eigh
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.utils.validation import check_X_y
from scipy import stats
from collections import Counter

__all__ = [
    'SmoothLocalRegionBuilder',
    'GlobalWeightMatrix',
    'BinaryClassWeightCorrector',
    'WeightedKNNClassifier',
    'run_mcaw_knn_classification'
]


class SmoothLocalRegionBuilder:
    """
    Local region builder based on Bayesian smoothing, Wilson smoothing, and Gaussian kernel.

    Provides smooth class representativeness evaluation and distance adjustment
    for constructing adaptive local regions around query points.

    Parameters
    ----------
    k_region : int, default=15
        Candidate set size for each class
    region_size : int, default=15
        Target local region size
    alpha : float, default=1.0
        Bayesian prior parameter α
    beta : float, default=1.0
        Bayesian prior parameter β
    confidence_level : float, default=0.95
        Wilson interval confidence level
    sigma : float, default=1.0
        Gaussian kernel bandwidth parameter
    prior_strength : int, default=5
        Prior strength for Bayesian estimation

    Attributes
    ----------
    sample_results : dict
        Detailed results for each sample after building a region
    class_tightness : dict
        Computed tightness values for each class
    class_candidates : dict
        Candidate sets for each class
    """

    def __init__(self, k_region=15, region_size=15, alpha=1.0, beta=1.0,
                 confidence_level=0.95, sigma=1.0, prior_strength=5):
        self.k_region = k_region
        self.region_size = region_size
        self.alpha = alpha
        self.beta = beta
        self.confidence_level = confidence_level
        self.sigma = sigma
        self.prior_strength = prior_strength

        # Storage for intermediate computation results
        self.sample_results = {}
        self.class_tightness = {}
        self.class_candidates = {}

    def fit(self, X, y, row_ids):
        """
        Fit the region builder with training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix
        y : array-like of shape (n_samples,)
            Training labels
        row_ids : array-like of shape (n_samples,)
            Row identifiers for tracking samples

        Returns
        -------
        self : object
            Fitted estimator
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.row_ids = np.array(row_ids)
        return self

    def compute_class_tightness(self, class_label, indices):
        """
        Compute class tightness (inverse of average distance to centroid).

        Tighter classes (smaller spread) have higher tightness values.
        """
        class_indices = indices[self.y[indices] == class_label]
        if len(class_indices) == 0:
            return 1.0

        class_samples = self.X[class_indices]
        if len(class_samples) <= 1:
            return 1.0

        centroid = np.mean(class_samples, axis=0)
        avg_distance = np.mean(np.linalg.norm(class_samples - centroid, axis=1))
        tightness = 1.0 / (avg_distance + 1e-8)
        return tightness

    def compute_backward_rank(self, target_point, sample_point, candidate_points):
        """
        Compute backward rank (rank of target from sample's perspective).

        A low backward rank indicates the sample considers the target as a close neighbor,
        suggesting high representativeness.
        """
        if len(candidate_points) == 0:
            return 1

        distances_to_sample = [np.linalg.norm(candidate - sample_point)
                              for candidate in candidate_points]
        target_to_sample_dist = np.linalg.norm(target_point - sample_point)
        all_distances = distances_to_sample + [target_to_sample_dist]
        sorted_indices = np.argsort(all_distances)
        rank = np.where(sorted_indices == len(all_distances) - 1)[0][0] + 1
        return rank

    def bayesian_smoothing(self, k, n, alpha=None, beta=None):
        """
        Compute Bayesian smoothed probability.

        Uses Beta-Binomial conjugate prior to handle small sample sizes.

        Parameters
        ----------
        k : int
            Number of successes
        n : int
            Total number of trials
        alpha : float, optional
            Prior parameter α (default: self.alpha)
        beta : float, optional
            Prior parameter β (default: self.beta)

        Returns
        -------
        float
            Bayesian smoothed probability estimate
        """
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta

        if n == 0:
            return 0.5  # Default prior

        p_bayesian = (k + alpha) / (n + alpha + beta)
        return p_bayesian

    def wilson_score_interval(self, p, n, z_score=None):
        """
        Compute Wilson score interval lower bound.

        Provides a conservative estimate that accounts for sampling uncertainty.

        Parameters
        ----------
        p : float
            Observed proportion
        n : int
            Sample size
        z_score : float, optional
            Z-score for confidence level

        Returns
        -------
        float
            Lower bound of Wilson confidence interval
        """
        if n == 0:
            return p

        if z_score is None:
            z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)

        denominator = 1 + z_score**2 / n
        centre = (p + z_score**2 / (2 * n)) / denominator
        width = (z_score * np.sqrt((p * (1 - p)) / n + z_score**2 / (4 * n**2))) / denominator

        p_wilson = centre - width
        return max(0, min(1, p_wilson))

    def gaussian_kernel_penalty(self, penalty, sigma=None):
        """Compute Gaussian kernel adjustment factor."""
        if sigma is None:
            sigma = self.sigma
        adjustment_factor = np.exp(-0.5 * (penalty / sigma) ** 2)
        return adjustment_factor

    def calculate_penalty(self, p_wilson, method='linear'):
        """
        Calculate penalty term based on Wilson-smoothed probability.

        Parameters
        ----------
        p_wilson : float
            Wilson-smoothed probability
        method : str
            Penalty method: 'linear', 'quadratic', or 'sigmoid'
        """
        if method == 'linear':
            penalty = 1.0 - p_wilson
        elif method == 'quadratic':
            penalty = (1.0 - p_wilson) ** 2
        elif method == 'sigmoid':
            penalty = 1.0 / (1.0 + np.exp(-10 * (p_wilson - 0.5)))
        else:
            penalty = 1.0 - p_wilson
        return penalty

    def compute_representativeness(self, R, r, N, sample_id):
        """
        Compute sample's class representativeness.

        Combines Bayesian smoothing, Wilson smoothing, and Gaussian kernel
        to produce a robust representativeness score.

        Parameters
        ----------
        R : float
            Relative tightness (class tightness / average tightness)
        r : float
            Normalized backward rank
        N : int
            Number of samples in candidate set
        sample_id : int
            Sample identifier

        Returns
        -------
        tuple
            (adjustment_factor, results_dict)
        """
        # Step 1: Initial representativeness assessment
        initial_representativeness = r

        # Step 2: Bayesian smoothing
        k_bayesian = max(1, int(r * N))
        n_bayesian = N
        p_bayesian = self.bayesian_smoothing(k_bayesian, n_bayesian)

        # Step 3: Wilson smoothing
        p_wilson = self.wilson_score_interval(initial_representativeness, N)

        # Step 4: Tightness-weighted adjustment
        tightness_weight = np.tanh(R - 1)
        weighted_p_wilson = p_wilson * (1 + 0.3 * tightness_weight)
        weighted_p_wilson = max(0, min(1, weighted_p_wilson))

        # Step 5: Calculate penalty
        penalty = self.calculate_penalty(weighted_p_wilson, method='sigmoid')

        # Step 6: Gaussian kernel adjustment
        adjustment_factor = self.gaussian_kernel_penalty(penalty)

        results = {
            'sample_id': sample_id,
            'R': R,
            'r': r,
            'N': N,
            'initial_representativeness': initial_representativeness,
            'p_bayesian': p_bayesian,
            'p_wilson': p_wilson,
            'weighted_p_wilson': weighted_p_wilson,
            'penalty': penalty,
            'adjustment_factor': adjustment_factor,
            'tightness_weight': tightness_weight
        }
        return adjustment_factor, results

    def build_local_region(self, target_point, verbose=False):
        """
        Build local region based on smooth representativeness evaluation.

        Parameters
        ----------
        target_point : array-like
            The query point for which to build the local region
        verbose : bool, default=False
            Whether to print detailed progress information

        Returns
        -------
        list
            Indices of samples in the local region
        """
        if verbose:
            print("=== Building Smooth Local Region ===")

        # Reset storage
        self.sample_results = {}
        self.class_tightness = {}
        self.class_candidates = {}

        unique_classes = np.unique(self.y)

        # Step 1: Build candidate set for each class
        candidate_indices = []
        for cls in unique_classes:
            class_indices = np.where(self.y == cls)[0]
            if len(class_indices) == 0:
                continue

            distances = [np.linalg.norm(self.X[i] - target_point) for i in class_indices]
            sorted_indices = np.argsort(distances)[:self.k_region]
            cls_candidates = [class_indices[i] for i in sorted_indices]

            tightness = self.compute_class_tightness(cls, class_indices)

            self.class_candidates[cls] = cls_candidates
            self.class_tightness[cls] = tightness
            candidate_indices.extend(cls_candidates)

        # Step 2: Compute adjusted distances
        all_candidates = list(set(candidate_indices))
        adjusted_distances = []

        for candidate_idx in all_candidates:
            cls = self.y[candidate_idx]
            row_id = self.row_ids[candidate_idx]

            if cls not in self.class_candidates:
                continue

            cls_candidates = self.class_candidates[cls]
            candidate_points = self.X[cls_candidates]
            sample_point = self.X[candidate_idx]

            original_distance = np.linalg.norm(sample_point - target_point)
            backward_rank = self.compute_backward_rank(target_point, sample_point, candidate_points)

            N = len(cls_candidates)
            r = 1 - (backward_rank - 1) / (N - 1) if N > 1 else 1.0

            avg_tightness = np.mean(list(self.class_tightness.values()))
            R = self.class_tightness[cls] / (avg_tightness + 1e-8)

            adjustment_factor, results = self.compute_representativeness(R, r, N, row_id)
            adjusted_distance = original_distance * adjustment_factor

            results.update({
                'row_id': row_id,
                'class': cls,
                'original_distance': original_distance,
                'adjusted_distance': adjusted_distance,
                'backward_rank': backward_rank,
                'R': R,
                'r': r
            })
            self.sample_results[row_id] = results
            adjusted_distances.append((candidate_idx, adjusted_distance, cls, row_id))

        # Step 3: Select samples by class proportion
        local_region = []
        class_distribution = Counter(self.y)
        total_samples = sum(class_distribution.values())

        class_targets = {}
        for cls, count in class_distribution.items():
            target_count = max(1, int(self.region_size * count / total_samples))
            class_targets[cls] = min(target_count, len(self.class_candidates.get(cls, [])))

        for cls, target_count in class_targets.items():
            if cls not in self.class_candidates or target_count == 0:
                continue

            cls_candidates = [idx for idx, _, cls_val, _ in adjusted_distances if cls_val == cls]
            if not cls_candidates:
                continue

            cls_candidates_sorted = sorted(
                cls_candidates,
                key=lambda idx: next(adj_dist for i, adj_dist, c, _ in adjusted_distances
                                   if i == idx and c == cls)
            )[:target_count]
            local_region.extend(cls_candidates_sorted)

        # Step 4: Supplement if needed
        if len(local_region) < self.region_size:
            remaining = self.region_size - len(local_region)
            used_set = set(local_region)
            available_candidates = [idx for idx, _, _, _ in adjusted_distances
                                  if idx not in used_set]
            if available_candidates:
                available_sorted = sorted(
                    available_candidates,
                    key=lambda idx: next(adj_dist for i, adj_dist, _, _ in adjusted_distances
                                       if i == idx)
                )[:remaining]
                local_region.extend(available_sorted)

        return local_region[:self.region_size]

    def get_detailed_results(self, sort_by='p_bayesian'):
        """Get detailed results sorted by specified metric."""
        if not self.sample_results:
            return []

        sort_keys = {
            'p_bayesian': lambda x: x[1]['p_bayesian'],
            'p_wilson': lambda x: x[1]['p_wilson'],
            'penalty': lambda x: x[1]['penalty'],
            'adjustment_factor': lambda x: x[1]['adjustment_factor'],
            'adjusted_distance': lambda x: x[1]['adjusted_distance']
        }

        reverse = sort_by not in ['penalty', 'adjustment_factor', 'adjusted_distance']

        if sort_by in sort_keys:
            return sorted(self.sample_results.items(), key=sort_keys[sort_by], reverse=reverse)
        return list(self.sample_results.items())


class GlobalWeightMatrix:
    """
    Computes global feature weights for each class.

    Supports multiple weight computation methods including LDA-based
    Rayleigh quotient, F-score, centroid-based, and inter-class difference.

    Parameters
    ----------
    method : str, default='lda'
        Weight computation method:
        - 'lda': Linear Discriminant Analysis via Rayleigh quotient
        - 'f_score': ANOVA F-statistics
        - 'centroid': Point-to-centroid ratio based
        - 'inter_class_difference': Precision matrix weighted differences

    Attributes
    ----------
    global_weights : ndarray
        Weight matrix of shape (n_classes, n_features)
    class_weight_dict : dict
        Dictionary mapping class labels to weight vectors
    """

    def __init__(self, method='lda'):
        self.method = method
        self.global_weights = None
        self.class_weight_dict = None
        self.feature_names = None
        self.class_names = None
        self.scaler = StandardScaler()

    def fit(self, X, y=None, feature_names=None, class_names=None):
        """
        Fit the weight calculator on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix
        y : array-like of shape (n_samples,), optional
            Class labels
        feature_names : list, optional
            Names of features
        class_names : list, optional
            Names of classes

        Returns
        -------
        self : object
            Fitted estimator
        """
        X = np.array(X)
        if y is not None:
            y = np.array(y)

        self.feature_names = feature_names
        self.class_names = class_names
        X_scaled = self.scaler.fit_transform(X)

        if y is None:
            self.global_weights = self._compute_global_weights(X_scaled)
            self.global_weights = self.global_weights.reshape(1, -1)
            self.class_weight_dict = {'global': self.global_weights[0]}
        else:
            unique_classes = np.unique(y)
            n_classes = len(unique_classes)
            n_features = X.shape[1]

            self.global_weights = np.zeros((n_classes, n_features))
            self.class_weight_dict = {}

            for i, class_label in enumerate(unique_classes):
                weight_vector = self._compute_global_weights(X_scaled, y, class_label)
                self.global_weights[i] = weight_vector
                self.class_weight_dict[class_label] = weight_vector

        return self

    def get_weight_matrix(self):
        """Return the weight matrix."""
        return self.global_weights

    def get_class_weight_dict(self):
        """Return the class weight dictionary."""
        return self.class_weight_dict

    def get_weight_by_class(self, class_identifier):
        """Get weight vector by class identifier."""
        if self.class_weight_dict is None:
            raise ValueError("Please call fit() method first")

        if class_identifier in self.class_weight_dict:
            return self.class_weight_dict[class_identifier]

        str_identifier = str(class_identifier)
        if str_identifier in self.class_weight_dict:
            return self.class_weight_dict[str_identifier]

        n_features = self.global_weights.shape[1] if self.global_weights is not None else 1
        return np.ones(n_features) / n_features

    def _compute_global_weights(self, X, y=None, current_class=None):
        """Dispatch to appropriate weight computation method."""
        if y is not None and current_class is not None:
            methods = {
                'inter_class_difference': self._inter_class_difference_weights,
                'f_score': self._f_score_weights,
                'centroid': self._centroid_weights,
                'lda': self._rayleigh_quotient
            }
            method_func = methods.get(self.method, self._inverse_covariance_weights)
            return method_func(X, y, current_class)
        return self._inverse_covariance_weights(X, y, current_class)

    def _rayleigh_quotient(self, X, y, current_class):
        """LDA weight computation based on generalized Rayleigh quotient."""
        try:
            y_binary = (y == current_class).astype(int)

            X1 = X[y_binary == 1]
            X0 = X[y_binary == 0]

            n1, n0 = len(X1), len(X0)
            n_features = X.shape[1]

            if n1 < 2 or n0 < 1:
                return np.ones(n_features) / n_features

            mu1 = np.mean(X1, axis=0)
            mu0 = np.mean(X0, axis=0)
            mean_diff = mu1 - mu0

            S1 = np.cov(X1.T) * (n1 - 1) if n1 > 1 else np.zeros((n_features, n_features))
            S0 = np.cov(X0.T) * (n0 - 1) if n0 > 1 else np.zeros((n_features, n_features))
            Sw = S1 + S0

            Sw_reg = Sw + 1e-6 * np.eye(n_features)

            try:
                Sw_inv = np.linalg.pinv(Sw_reg)
                weights = np.dot(Sw_inv, mean_diff)
            except np.linalg.LinAlgError:
                weights = mean_diff

            weights = np.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
            weights = np.abs(weights)

            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n_features) / n_features

            return weights

        except Exception:
            return np.ones(X.shape[1]) / X.shape[1]

    def _centroid_weights(self, X, y, current_class):
        """Centroid method: weights based on ratio to global centroid."""
        try:
            X_class = X[y == current_class]
            n_samples_class = len(X_class)
            n_features = X.shape[1]

            if n_samples_class == 0:
                return np.ones(n_features) / n_features

            centroid_all = np.mean(X, axis=0) + 1e-8

            ratio_vectors = []
            for i in range(n_samples_class):
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.abs(np.divide(X_class[i], centroid_all))
                ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=0.0)
                ratio_vectors.append(ratio)

            mean_ratio = np.mean(ratio_vectors, axis=0) if ratio_vectors else np.ones(n_features)

            if np.sum(mean_ratio) > 0:
                return mean_ratio / np.sum(mean_ratio)
            return np.ones(n_features) / n_features

        except Exception:
            return np.ones(X.shape[1]) / X.shape[1]

    def _inter_class_difference_weights(self, X, y, current_class):
        """Compute weights based on inter-class differences."""
        try:
            X_current = X[y == current_class]
            X_other = X[y != current_class]

            if len(X_current) < 2 or len(X_other) < 1:
                return np.ones(X.shape[1]) / X.shape[1]

            cov_estimator = LedoitWolf().fit(X_current)
            precision = np.linalg.pinv(cov_estimator.covariance_)

            mean_diff = np.mean(X_current, axis=0) - np.mean(X_other, axis=0)
            weights = np.abs(np.dot(precision, mean_diff))

            if np.sum(weights) > 0:
                return weights / np.sum(weights)
            return np.ones(X.shape[1]) / X.shape[1]

        except Exception:
            return np.ones(X.shape[1]) / X.shape[1]

    def _f_score_weights(self, X, y, current_class):
        """Compute weights using F-score."""
        try:
            from sklearn.feature_selection import f_classif
            y_binary = (y == current_class).astype(int)
            f_scores, _ = f_classif(X, y_binary)
            f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=1.0, neginf=0.0)

            if np.sum(f_scores) > 0:
                return f_scores / np.sum(f_scores)
            return np.ones(X.shape[1]) / X.shape[1]

        except Exception:
            return np.ones(X.shape[1]) / X.shape[1]

    def _inverse_covariance_weights(self, X, y=None, current_class=None):
        """Compute weights using inverse covariance matrix."""
        try:
            if current_class is not None and y is not None:
                X_used = X[y == current_class]
                if len(X_used) < 2:
                    return np.ones(X.shape[1]) / X.shape[1]
            else:
                X_used = X

            cov_estimator = LedoitWolf().fit(X_used)
            precision = np.linalg.pinv(cov_estimator.covariance_)

            weights = np.abs(np.dot(precision, np.ones(X_used.shape[1])))

            if np.sum(weights) > 0:
                return weights / np.sum(weights)
            return np.ones(X_used.shape[1]) / X_used.shape[1]

        except Exception:
            return np.ones(X.shape[1]) / X.shape[1]


class BinaryClassWeightCorrector(BaseEstimator):
    """
    Weight vector corrector based on binary global manifold constraints.

    For each class c, treats it as a binary classification problem (c vs. not c),
    then corrects the local weight vector via generalized Rayleigh quotient.

    Parameters
    ----------
    reg_param : float, default=1e-6
        Regularization parameter for numerical stability
    alpha : float, default=0.5
        Balance between local weight preservation and global constraint
    method : str, default='projection'
        Correction method: 'projection' or 'optimization'

    Attributes
    ----------
    class_stats_ : dict
        Statistics for each class (scatter matrices, means, etc.)
    """

    def __init__(self, reg_param=1e-6, alpha=0.5, method='projection'):
        self.reg_param = reg_param
        self.alpha = alpha
        self.method = method
        self.class_stats_ = {}

    def fit(self, X_train, y_train):
        """
        Fit on training data, compute statistics for each class.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix
        y_train : array-like of shape (n_samples,)
            Training labels

        Returns
        -------
        self : object
            Fitted estimator
        """
        X_train, y_train = check_X_y(X_train, y_train)
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.n_features_ = X_train.shape[1]
        self.classes_ = np.unique(y_train)

        self._compute_class_statistics()
        return self

    def _compute_class_statistics(self):
        """Compute statistics for each class."""
        for cls in self.classes_:
            X_c = self.X_train_[self.y_train_ == cls]
            X_not_c = self.X_train_[self.y_train_ != cls]
            n_c, n_not_c = X_c.shape[0], X_not_c.shape[0]

            mean_c = np.mean(X_c, axis=0)
            mean_not_c = np.mean(X_not_c, axis=0)
            global_mean = (n_c * mean_c + n_not_c * mean_not_c) / (n_c + n_not_c)

            S_w_c = self._compute_within_class_scatter(X_c, mean_c)
            S_w_not_c = self._compute_within_class_scatter(X_not_c, mean_not_c)
            S_w = S_w_c + S_w_not_c

            S_b = self._compute_between_class_scatter_binary(
                mean_c, mean_not_c, n_c, n_not_c, global_mean
            )

            self.class_stats_[cls] = {
                'S_w': S_w, 'S_b': S_b,
                'mean_c': mean_c, 'mean_not_c': mean_not_c,
                'global_mean': global_mean,
                'n_c': n_c, 'n_not_c': n_not_c
            }

    def _compute_within_class_scatter(self, X, mean):
        """Compute within-class scatter matrix."""
        n_samples, n_features = X.shape
        S_w = np.zeros((n_features, n_features))
        for i in range(n_samples):
            diff = (X[i] - mean).reshape(-1, 1)
            S_w += diff @ diff.T
        return S_w

    def _compute_between_class_scatter_binary(self, mean_c, mean_not_c, n_c, n_not_c, global_mean):
        """Compute between-class scatter matrix for binary classification."""
        diff_c = (mean_c - global_mean).reshape(-1, 1)
        diff_not_c = (mean_not_c - global_mean).reshape(-1, 1)
        return n_c * (diff_c @ diff_c.T) + n_not_c * (diff_not_c @ diff_not_c.T)

    def correct_weight_vector(self, w_local, class_label):
        """
        Correct local weight vector using manifold projection.

        Parameters
        ----------
        w_local : array-like of shape (n_features,)
            Local weight vector to correct
        class_label : int
            Class label for the binary classification problem

        Returns
        -------
        w_corrected : ndarray of shape (n_features,)
            Corrected weight vector
        """
        if not hasattr(self, 'X_train_'):
            raise ValueError("Model not fitted. Call fit() first.")

        if class_label not in self.class_stats_:
            raise ValueError(f"Class {class_label} not in training data")

        w_local = np.array(w_local).flatten()
        if len(w_local) != self.n_features_:
            raise ValueError(f"Dimension mismatch: expected {self.n_features_}, got {len(w_local)}")

        stats = self.class_stats_[class_label]
        S_w = stats['S_w']
        S_b = stats['S_b']
        S_w_reg = S_w + self.reg_param * np.eye(self.n_features_)

        if self.method == 'projection':
            w_corrected = self._project_to_manifold(w_local, S_b, S_w_reg)
        elif self.method == 'optimization':
            w_corrected = self._optimize_with_constraints(w_local, S_b, S_w_reg)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self._normalize_weight_vector(w_corrected)

    def _project_to_manifold(self, w_local, S_b, S_w):
        """Project onto generalized Rayleigh quotient manifold."""
        try:
            eigenvalues, eigenvectors = eigh(S_b, S_w)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]

            w_manifold = eigenvectors[:, 0]
            coeff = np.dot(w_manifold, w_local) / np.dot(w_manifold, w_manifold)
            w_projected = coeff * w_manifold

            return self.alpha * w_projected + (1 - self.alpha) * w_local
        except np.linalg.LinAlgError:
            return w_local

    def _optimize_with_constraints(self, w_local, S_b, S_w):
        """Correct weight vector through optimization."""
        def objective(w, w_local, S_b, S_w, alpha):
            w = w.reshape(-1, 1)
            w_local_col = w_local.reshape(-1, 1)

            diff_term = np.sum((w - w_local_col)**2)
            denominator = w.T @ S_w @ w
            rayleigh = (w.T @ S_b @ w) / denominator if denominator > 1e-10 else 0

            return alpha * diff_term - (1 - alpha) * rayleigh

        result = minimize(
            objective, w_local,
            args=(w_local, S_b, S_w, self.alpha),
            method='L-BFGS-B',
            bounds=[(-1, 1)] * len(w_local),
            options={'maxiter': 1000, 'ftol': 1e-8}
        )

        return result.x if result.success else w_local

    def _normalize_weight_vector(self, w):
        """Normalize weight vector."""
        w_norm = np.linalg.norm(w)
        return w / w_norm if w_norm > 1e-10 else w

    def compute_binary_rayleigh_quotient(self, w, class_label):
        """Compute generalized Rayleigh quotient for a weight vector."""
        if class_label not in self.class_stats_:
            raise ValueError(f"Class {class_label} not in training data")

        stats = self.class_stats_[class_label]
        S_w = stats['S_w'] + self.reg_param * np.eye(self.n_features_)
        S_b = stats['S_b']

        w = w.reshape(-1, 1)
        numerator = w.T @ S_b @ w
        denominator = w.T @ S_w @ w

        return (numerator / denominator).flatten()[0] if denominator > 1e-10 else 0.0


class WeightedKNNClassifier:
    """
    Weighted KNN classifier based on local region weights.

    Uses class-specific feature weights to compute distances and
    performs distance-weighted voting for classification.

    Parameters
    ----------
    k : int, default=7
        Number of neighbors
    distance_weight_method : str, default='exponential'
        Method for distance weighting
    inv_cov_matrix : ndarray, optional
        Inverse covariance matrix for Mahalanobis distance
    """

    def __init__(self, k=7, distance_weight_method='exponential', inv_cov_matrix=None):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.X_row_number = None
        self.feature_names = None
        self.distance_weight_method = distance_weight_method
        self.inv_cov_matrix = inv_cov_matrix

    def fit(self, X_train, y_train, X_row_number, feature_names=None):
        """Fit the classifier with training data."""
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_row_number = np.array(X_row_number)
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]

    def calculate_improved_distance_weight(self, distances, method='rank_exponential', top_m=5):
        """Improved distance weight calculation emphasizing nearest neighbors."""
        if len(distances) == 0:
            return np.array([])

        distances = np.array(distances)

        if method == 'rank_exponential':
            ranks = np.arange(1, len(distances) + 1)
            weights = 2.0 ** (-ranks + 1)
            weights = [w * (1/(d+1e-8)) for w, d in zip(weights, distances)]
        elif method == 'rank_power':
            ranks = np.arange(1, len(distances) + 1)
            weights = 1.0 / (ranks ** 3.0)
        elif method == 'relative_distance':
            min_dist = max(np.min(distances), 1e-8)
            weights = np.exp(2.0 * (1 - distances / min_dist))
        elif method == 'top_m_dominant':
            weights = np.zeros(len(distances))
            for i in range(min(top_m, len(distances))):
                weights[i] = 1.0 if i == 0 else 0.1
        else:
            ranks = np.arange(1, len(distances) + 1)
            weights = 2.0 ** (-ranks + 1)

        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

        return weights

    def calculate_weighted_distance(self, target_point, weights):
        """Calculate weighted Euclidean distances."""
        weights = weights / np.sum(weights)

        weighted_distances = []
        for i in range(len(self.X_train)):
            weighted_diff = weights * (self.X_train[i] - target_point)
            distance = np.sqrt(np.sum(weighted_diff ** 2))
            weighted_distances.append((self.X_row_number[i], distance, self.y_train[i]))

        return weighted_distances

    def predict_with_distance_weighted_voting(self, target_point, comprehensive_weights,
                                               local_region_classes, verbose=False):
        """
        KNN prediction with distance-weighted voting.

        Returns
        -------
        tuple
            (predicted_class, max_consistency, knn_results, weighted_votes)
        """
        knn_results = {}
        weighted_votes = {}

        for cls, weights in comprehensive_weights.items():
            if cls not in local_region_classes:
                continue

            weighted_distances = self.calculate_weighted_distance(target_point, weights)
            weighted_distances.sort(key=lambda x: x[1])
            k_neighbors = weighted_distances[:self.k]

            distances = [item[1] for item in k_neighbors]
            neighbor_classes = [item[2] for item in k_neighbors]

            distance_weights = self.calculate_improved_distance_weight(distances)
            class_count = sum(1 for _, _, nc in k_neighbors if nc == cls)
            consistency = class_count / self.k

            weighted_vote = 0.0
            for rank, neighbor_class in enumerate(neighbor_classes):
                if neighbor_class == cls:
                    weighted_vote = consistency * 1/(distances[rank] + 1e-8)
                    break

            weighted_votes[cls] = weighted_vote
            knn_results[cls] = {
                'neighbors': k_neighbors,
                'distances': distances,
                'distance_weights': distance_weights,
                'neighbor_classes': neighbor_classes,
                'consistency': weighted_vote,
                'weights_used': weights
            }

        if weighted_votes:
            predicted_class = max(weighted_votes.items(), key=lambda x: x[1])[0]
            return predicted_class, weighted_votes[predicted_class], knn_results, weighted_votes

        # Fallback to simple KNN
        distances = np.linalg.norm(self.X_train - target_point, axis=1)
        nearest_indices = np.argsort(distances)[:self.k]
        predicted_class = Counter(self.y_train[nearest_indices]).most_common(1)[0][0]
        return predicted_class, 0.0, {}, {}


def run_mcaw_knn_classification(file_path, target_column='quality', test_size=0.2,
                                 k_neighbors=7, region_size=20, verbose=False):
    """
    Run the complete MCAW-KNN classification pipeline.

    Parameters
    ----------
    file_path : str
        Path to the CSV data file
    target_column : str, default='quality'
        Name of the target column
    test_size : float, default=0.2
        Proportion of data to use for testing
    k_neighbors : int, default=7
        Number of neighbors for KNN
    region_size : int, default=20
        Size of local regions
    verbose : bool, default=False
        Whether to print detailed progress

    Returns
    -------
    dict
        Results including accuracy and detailed test information
    """

    # Load and preprocess data
    print("=" * 60)
    print("MCAW-KNN Classification")
    print("=" * 60)

    df = pd.read_csv(file_path)
    df['csv_row_number'] = range(2, len(df) + 2)
    df = df.dropna()

    if 'type' in df.columns:
        df['type'] = df['type'].map({'red': 1, 'white': 2, 'Red': 1, 'White': 2}).fillna(0)

    row_indices = df['csv_row_number'].values
    X = df.drop([target_column, 'csv_row_number'], axis=1).values
    y = df[target_column].values
    feature_names = df.drop([target_column, 'csv_row_number'], axis=1).columns.tolist()

    print(f"Data shape: {X.shape}")
    print(f"Classes: {len(np.unique(y))}")

    # Train-test split
    X_train, X_test, y_train, y_test, row_train, row_test = train_test_split(
        X, y, row_indices, test_size=test_size, random_state=42, stratify=y
    )

    # Standardize
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    minmax = MinMaxScaler()
    X_train_norm = minmax.fit_transform(X_train_scaled)
    X_test_norm = minmax.transform(X_test_scaled)

    print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Initialize components
    corrector = BinaryClassWeightCorrector()
    corrector.fit(X_train, y_train)

    weight_calculator = GlobalWeightMatrix(method='lda')

    region_builder = SmoothLocalRegionBuilder(
        k_region=15, region_size=region_size,
        alpha=1.0, beta=1.0, confidence_level=0.95, sigma=0.5
    )
    region_builder.fit(X_train, y_train, row_train)

    knn_classifier = WeightedKNNClassifier(k=k_neighbors)
    knn_classifier.fit(X_train, y_train, row_train, feature_names)

    # Classification
    print("\nClassifying...")
    correct_count = 0
    results = {'details': {}}

    for idx in range(len(X_test)):
        target_point = X_test[idx]
        target_class = y_test[idx]

        best_region = region_builder.build_local_region(target_point)

        if best_region and len(best_region) > 0:
            X_region = X_train[best_region]
            y_region = y_train[best_region]

            weight_calculator.fit(X_region, y_region)
            weights = weight_calculator.get_class_weight_dict()

            for cls in weights:
                weights[cls] = corrector.correct_weight_vector(weights[cls], cls)

            local_classes = list(set(y_region))
            predicted, consistency, _, _ = knn_classifier.predict_with_distance_weighted_voting(
                target_point, weights, local_classes
            )

            is_correct = (predicted == target_class)
            if is_correct:
                correct_count += 1

            results['details'][idx] = {
                'true': target_class,
                'predicted': predicted,
                'correct': is_correct
            }

            if verbose and idx % 10 == 0:
                print(f"  {idx+1}/{len(X_test)}: {'✓' if is_correct else '✗'}")

    accuracy = correct_count / len(X_test)
    results['accuracy'] = accuracy
    results['correct_count'] = correct_count
    results['total_count'] = len(X_test)

    print(f"\nResults: {correct_count}/{len(X_test)} correct")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return results


if __name__ == "__main__":
    # Example usage
    print("MCAW-KNN Module")
    print("Import this module and use run_mcaw_knn_classification() or individual classes.")
