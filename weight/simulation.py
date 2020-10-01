import pandas as pd 
import numpy as np
from sklearn import datasets

from .config import Config


def lin_sep(dis2origin=Config.dis2origin, axis_limit=Config.axis_limit, size=Config.size, cov_scale=1):
    """Create linear separable 2d data"""
    pos_size = size // 2
    pos_center = [dis2origin, -1 * dis2origin]
    while True:
        pos_cord = np.random.multivariate_normal(mean=pos_center, cov=cov_scale * np.eye(2), size=pos_size * 5)
        pos_cord = pos_cord[np.linalg.norm((pos_cord - np.array(pos_center)), axis=1) < axis_limit]
        pos_cord = pos_cord[:pos_size]
        if len(pos_cord) == pos_size:
            break

    neg_cord = -1 * pos_cord
    X = np.vstack([pos_cord, neg_cord]).astype(np.float32)
    X = (X - X.mean(0)[np.newaxis, :]) / X.std(0)[np.newaxis, :]
    y = np.concatenate([np.ones(pos_size), np.zeros(pos_size)], axis=0)
    return X, y


def non_lin_sep(dis2origin=Config.dis2origin, axis_limit=Config.axis_limit, size=Config.size):
    """Create data with linear separable and non-separable components"""
    x,y = lin_sep(dis2origin, axis_limit, int(size * 0.5), 1)
    sx, sy = lin_sep(dis2origin, axis_limit, int(size * 0.5), 1)
    x *= 1.5
    sx *= 0.5
    sy = 1 - sy #this flip the data
    X = np.concatenate([x, sx], axis=0)
    y = np.concatenate([y, sy], axis=0)
    X = (X - X.mean(0)[np.newaxis, :]) / X.std(0)[np.newaxis, :]
    return X, y


def non_lin_moon(noise=0.05):
    """Create moon shaped non-separable data"""
    X, y = datasets.make_moons(n_samples=Config.size, noise=noise)
    # Increase the gap
    X[y > 0, 1] -= 0.15
    X[y <= 0, 1] += 0.15
    X = (X - X.mean(0)[np.newaxis, :]) / X.std(0)[np.newaxis, :]
    return X, y