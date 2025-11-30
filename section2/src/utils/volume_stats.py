"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    # Dice = 2 * |A ∩ B| / (|A| + |B|)
    a_binary = (a > 0).astype(np.float32)
    b_binary = (b > 0).astype(np.float32)
    intersection = np.sum(a_binary * b_binary)
    volume_sum = np.sum(a_binary) + np.sum(b_binary)
    
    if volume_sum == 0:
        return 1.0 if np.sum(a_binary) == 0 and np.sum(b_binary) == 0 else 0.0
    
    dice = 2.0 * intersection / volume_sum
    return dice

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    # Jaccard = |A ∩ B| / |A ∪ B|
    a_binary = (a > 0).astype(np.float32)
    b_binary = (b > 0).astype(np.float32)
    intersection = np.sum(a_binary * b_binary)
    union = np.sum(a_binary) + np.sum(b_binary) - intersection
    
    if union == 0:
        return 1.0 if np.sum(a_binary) == 0 and np.sum(b_binary) == 0 else 0.0
    
    jaccard = intersection / union
    return jaccard