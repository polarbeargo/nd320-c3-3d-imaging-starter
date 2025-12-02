"""
Module loads the hippocampus dataset into RAM
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from utils.utils import med_reshape

def process_single_file(args):
    """
    Process a single NIFTI file pair (image and label)
    
    Arguments:
        args: tuple of (filename, image_dir, label_dir, y_shape, z_shape)
    
    Returns:
        Dictionary with processed image, label, and filename
    """
    f, image_dir, label_dir, y_shape, z_shape = args
    
    try:
        # We would benefit from mmap load method here if dataset doesn't fit into memory
        # Images are loaded here using MedPy's load method. We will ignore header 
        # since we will not use it
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f))

        # TASK: normalize all images (but not labels) so that values are in [0..1] range
        # Normalize by subtracting min and dividing by range to get [0, 1]
        image_min = np.min(image)
        image_max = np.max(image)
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        
        # We need to reshape data since CNN tensors that represent minibatches
        # in our case will be stacks of slices and stacks need to be of the same size.
        # In the inference pathway we will need to crop the output to that
        # of the input image.
        # Note that since we feed individual slices to the CNN, we only need to 
        # extend 2 dimensions out of 3. We choose to extend coronal and sagittal here

        # TASK: med_reshape function is not complete. Go and fix it!
        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)

        # TASK: Why do we need to cast label to int?
        # ANSWER: Labels represent discrete class indices (0=background, 1=anterior, 2=posterior).
        # The CrossEntropyLoss function expects integer class labels, not floating point values.
        # Converting to int ensures compatibility with PyTorch's loss function and prevents numerical errors.

        return {"image": image, "seg": label, "filename": f}
    except Exception as e:
        print(f"Error processing {f}: {e}")
        return None

def LoadHippocampusData(root_dir, y_shape, z_shape):
    '''
    This function loads our dataset from disk into memory,
    reshaping output to common size. Uses parallel processing
    for faster loading.

    Arguments:
        root_dir {string} -- path to the dataset directory
        y_shape {int} -- target Y dimension
        z_shape {int} -- target Z dimension

    Returns:
        Array of dictionaries with data stored in seg and image fields as 
        Numpy arrays of shape [AXIAL_WIDTH, Y_SHAPE, Z_SHAPE]
    '''

    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')

    images = [f for f in listdir(image_dir) if (
        isfile(join(image_dir, f)) and f[0] != ".")]

    print(f"Found {len(images)} files to process")
    print(f"Using parallel processing with {cpu_count()} CPU cores...")
    
    process_args = [(f, image_dir, label_dir, y_shape, z_shape) for f in images]
    
    out = []
    
    max_workers = min(cpu_count(), len(images))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, args) for args in process_args]
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                out.append(result)
            
            completed += 1
            if completed % 50 == 0:
                print(f"Processed {completed}/{len(images)} files...")

    out = sorted(out, key=lambda x: x['filename'])

    # Hippocampus dataset only takes about 300 Mb RAM, so we can afford to keep it all in RAM
    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
    return np.array(out)
