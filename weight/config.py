import numpy as np

class Config:
    axis_limit = 4 # Limit for both x and y axis
    size = 1024 # Number of points
    dis2origin = 3 
    batch_size=1024
    buffer_size=50
    l2 = 0.000
    resolution=200 # Number of steps used in linspace
    
    x_span = np.linspace(-1 * axis_limit, axis_limit, resolution)
    y_span = np.linspace(-1 * axis_limit, axis_limit, resolution)