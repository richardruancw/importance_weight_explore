import numpy as np
from matplotlib import pylab as plt
from .config import Config

def split_cord(X, y):
    return X[y > 0], X[y <= 0]



def extract_margin(X, y, xx, yy, model):
    """Extract the decision boundary, margin and support vectors"""
    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)
    est = model.predict(X).flatten()
    y_signed = -1 * (1 - 2 * y)
    func_margin = y_signed * est
    argmin = func_margin.argmin()
    margin = func_margin[argmin]
    # find the `support vector`
    support_vector = X[argmin]
    
    boundary_pos = np.abs(z) < 0.1
    if len(boundary_pos) < 0:
        target_vector = None
        df = None
    else:
        boundary_vectors = np.c_[xx[boundary_pos], yy[boundary_pos]]
        target_vector = boundary_vectors[np.linalg.norm(boundary_vectors - support_vector, axis=1).argmin()]
        dt = np.linspace(0, 1, 30).repeat(2).reshape(-1, 2)
        df = (support_vector - target_vector).reshape(-1, 2) * dt + target_vector
    return z, margin, support_vector, target_vector, df


def preprocess_deep(X, y, x_span, y_span, model, weights):
    result = []
    xx, yy = np.meshgrid(x_span, y_span)
    for weight in weights:
        model.set_weights(weight)
        result.append(extract_margin(X, y, xx, yy, model))
    return result


def progress_plot(X, y,
                  x_span, y_span,
                  lin_results=None, deep_results=None, margins=None,
                  cmap='Paired'):
    """Create training progress"""
    num_step = len(lin_results)
    assert(len(lin_results) == len(deep_results))
    figure, axes = plt.subplots(ncols=num_step, figsize=(20, 3))
    pos_cord, neg_cord = split_cord(X, y)
    xx, yy = np.meshgrid(x_span, y_span)
    for i, ax in enumerate(axes):          
        ax.scatter(pos_cord[:, 0], pos_cord[:, 1], alpha=0.9)
        ax.scatter(neg_cord[:, 0], neg_cord[:, 1], alpha=0.9)
        ax.set_ylim(min(y_span),  max(y_span))
        ax.set_xlim(min(x_span), max(x_span))
        
        if lin_results is not None:
            weight, bias = lin_results[i]
            slope = -1 * weight[1] /  weight[0]
            lg_y = slope * x_span + bias
            ax.plot(x_span, lg_y)
            
        if deep_results is not None:
            z, margin, support_vector, target_vector, df = deep_results[i]
            zt = z > 0
            ax.plot([support_vector[0]], [support_vector[1]], marker='o', markersize=4, color="red")
            if target_vector is not None:
                ax.plot([target_vector[0]], [target_vector[1]], marker='o', markersize=4, color="yellow")
            if df is not None:
                ax.plot(df[:, 0], df[:, 1], color="grey")
            ax.contourf(xx, yy, zt, cmap=cmap, alpha=0.5)    
            
        
        if margins:
            ax.plot(margins[0], margins[1])
        ax.set_aspect('equal', adjustable='box')
    plt.plot()