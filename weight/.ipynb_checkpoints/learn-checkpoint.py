import numpy as np
from sklearn import svm
import tensorflow as tf

from .config import Config

def get_margin(X, y, axis_limit=Config.axis_limit):
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    slope = -1 * clf.coef_[0][1] /  clf.coef_[0][0]
    margin_x = np.linspace(-1 * axis_limit, axis_limit, Config.resolution)
    margin_y = slope * margin_x + clf.intercept_
    return margin_x, margin_y


class LinearCollector(tf.keras.callbacks.Callback):
    def __init__(self, milestones):
        super(LinearCollector, self).__init__()
        self.milestones = milestones
        self.max_epoch = max(milestones)
        self.epoch_cnt = 0
        self.fitted_curves = []
            
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_cnt += 1
        if self.epoch_cnt in self.milestones:
            print(f'doing - {self.epoch_cnt} epoch')
            lg_weight = self.model.weights[0].numpy()
            lg_bias = self.model.weights[1].numpy()[0]
            self.fitted_curves.append((lg_weight, lg_bias))
        if self.epoch_cnt >= self.max_epoch:
            self.model.stop_training = True
            

class NonLinearCollector(tf.keras.callbacks.Callback):
    def __init__(self, milestones):
        super(NonLinearCollector, self).__init__()
        self.milestones = milestones
        self.max_epoch = max(milestones)
        self.epoch_cnt = 0
        self.fitted_weights = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_cnt += 1
        if self.epoch_cnt in self.milestones:
            
            print(f'doing - {self.epoch_cnt} epoch')
            self.fitted_weights.append(self.model.get_weights())

        if self.epoch_cnt >= self.max_epoch:
            self.model.stop_training = True
            
            
def make_class_weight(y, pos, neg):
    sample_weight = np.ones_like(y, dtype=np.float32)
    sample_weight[y > 0] *= pos / (pos + neg)
    sample_weight[y <= 0] *= neg / (pos + neg)
    return sample_weight


def run_linear(X, y, sample_weight, lr, milestones = (1, 10, 100, 500, 1500), model=None, optimizer=None,
        batch_size=Config.batch_size, buffer_size=Config.buffer_size, l2=Config.l2):
    """Fit a linear logistic model and returns the weights at each milestone."""
    X = X.astype(np.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((X, y, sample_weight))
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    
    if model is None:
        model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=1, 
                name='dense_weight', 
                kernel_regularizer=tf.keras.regularizers.l1_l2(l2=l2, l1=0),
                bias_regularizer=tf.keras.regularizers.l2(l2))
            ])
    if optimizer is None:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=lr,
                            decay_steps=10000,
                            decay_rate=0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    curve_callback = LinearCollector(milestones)
    model.compile(optimizer=optimizer,
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    model.fit(train_dataset, epochs=max(milestones) + 1, verbose=0, callbacks=[curve_callback])
    return curve_callback.fitted_curves
    


def run_non_linear(X, y, sample_weight, lr, milestones = (1, 10, 100, 500, 1500), model=None, optimizer=None,
        batch_size=Config.batch_size, buffer_size=Config.buffer_size, l2=Config.l2):
    """Fit a linear logistic model and returns the weights at each milestone."""
    X = X.astype(np.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((X, y, sample_weight))
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    
    if model is None:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, 
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l2=l2, l1=0),
                bias_regularizer=tf.keras.regularizers.l2(l2)),
                tf.keras.layers.Dense(units=1, 
                name='dense_weight', 
                kernel_regularizer=tf.keras.regularizers.l1_l2(l2=l2, l1=0),
                bias_regularizer=tf.keras.regularizers.l2(l2))
            ])
    if optimizer is None:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=lr,
                            decay_steps=10000,
                            decay_rate=0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    curve_callback = NonLinearCollector(milestones)
    model.compile(optimizer=optimizer,
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    model.fit(train_dataset, epochs=max(milestones) + 1, verbose=0, callbacks=[curve_callback])
    return model, curve_callback.fitted_weights
    
        
    
    