import numpy as np
from matplotlib import pylab as plt
from .config import Config

def split_cord(X, y):
    return X[y > 0], X[y <= 0]



class LinModel:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def predict(self, feat):
        return feat.dot(self.weight) + self.bias

    @property
    def weights(self):
        return [self.weight]

def get_linear_margin(X, y, xx, yy, lin_results):
    # A hacky one to reuse the code that designed for model class
    out = []
    for lin_result in lin_results:
        lin_model = LinModel(*lin_result)
        _, margin, _, _, _ = extract_margin(X, y, xx, yy, lin_model)
        out.append(margin)
    return out
    
def get_model_norm(model):
    norm = 0
    for weight in model.weights:
        if not isinstance(weight, np.ndarray):
            weight = weight.numpy()
        norm += np.linalg.norm(weight)
    norm = np.sqrt(norm)
    return norm

def extract_margin(X, y, xx, yy, model):
    """Extract the decision boundary, margin and support vectors"""
    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])
    model_norm = get_model_norm(model)
    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)
    est = model.predict(X).flatten()
    y_signed = 2 * y - 1
    func_margin = y_signed * est
    argmin = func_margin.argmin()

    margin = y_signed[argmin] * model.predict(X/model_norm)[argmin]

    support_vector = X[argmin]
    
    boundary_pos = np.abs(z) < 0.05
    if boundary_pos.sum() < 1:
        target_vector = None
        df = None
        #margin = None
    else:
        boundary_vectors = np.c_[xx[boundary_pos], yy[boundary_pos]]
        target_vector = boundary_vectors[np.linalg.norm(boundary_vectors - support_vector, axis=1).argmin()]
        dt = np.linspace(0, 1, 30).repeat(2).reshape(-1, 2)
        df = (support_vector - target_vector).reshape(-1, 2) * dt + target_vector
        #margin = np.sign(y_signed[argmin] * est[argmin]) * np.linalg.norm(target_vector - support_vector)
    
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
    num_step = len(lin_results) if lin_results else len(deep_results)
    figure, axes = plt.subplots(ncols=num_step, figsize=(20, 3))
    pos_cord, neg_cord = split_cord(X, y)
    xx, yy = np.meshgrid(x_span, y_span)
    for i, ax in enumerate(axes):          
        if deep_results is not None:
            z, margin, support_vector, target_vector, df = deep_results[i]
            zt = z > 0
            ax.contourf(xx, yy, zt, cmap=cmap, alpha=0.5)  
        ax.scatter(pos_cord[:, 0], pos_cord[:, 1], alpha=0.9)
        ax.scatter(neg_cord[:, 0], neg_cord[:, 1], alpha=0.9)
        ax.set_ylim(min(y_span),  max(y_span))
        ax.set_xlim(min(x_span), max(x_span))
        
        if lin_results is not None:
            weight, bias = lin_results[i]
            slope = -1 * weight[1] /  weight[0]
            lg_y = slope * x_span + bias
            ax.plot(x_span, lg_y)
        
        if margins:
            ax.plot(margins[0], margins[1])

        if deep_results is not None:
            ax.plot([support_vector[0]], [support_vector[1]], marker='o', markersize=3, color="red")
            if target_vector is not None:
                ax.plot([target_vector[0]], [target_vector[1]], marker='o', markersize=3, color="red")
            if df is not None:
                ax.plot(df[:, 0], df[:, 1], color="yellow")
        ax.set_aspect('equal', adjustable='box')
    plt.plot()