import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Helper metrics
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

# Regression metrics
def regression_metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE(%)': mape(y_true, y_pred)
    }

# Small plotting helpers
def plot_histograms(df, cols=None, figsize=(12,8)):
    cols = cols or df.columns.tolist()
    n = len(cols)
    rows = int(np.ceil(n/3))
    fig, axes = plt.subplots(rows, 3, figsize=figsize)
    axes = axes.flatten()
    for i, c in enumerate(cols):
        axes[i].hist(df[c], bins=20)
        axes[i].set_title(c)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
    
def scatter_feature_target(df, feature, target='target'):
    plt.figure(figsize=(6,4))
    plt.scatter(df[feature], df[target], alpha=0.6)
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f'{feature} vs {target}')
    plt.show()
    
def plot_model_fit(x_train, y_train, x_val, y_val, x_test, y_test, model, poly=None, xlabel='BMI', ylabel='Disease Progression'):
    x_all = np.concatenate([x_train, x_val, x_test])
    x_min, x_max = x_all.min(), x_all.max()
    xs = np.linspace(x_min, x_max, 200).reshape(-1,1)
    if poly is not None:
        xs_trans = poly.transform(xs)
        ys = model.predict(xs_trans)
    else:
        ys = model.predict(xs)
    plt.figure(figsize=(8,6))
    plt.scatter(x_train, y_train, label='train', alpha=0.6)
    plt.scatter(x_val, y_val, label='val', alpha=0.6)
    plt.scatter(x_test, y_test, label='test', alpha=0.6)
    plt.plot(xs, ys, color='black', linewidth=2, label='model fit')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
