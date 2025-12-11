"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

def med_reshape(image, new_shape):
    """
    This function reshapes 3D data to new dimension padding with zeros
    and leaving the content in the top-left corner

    Arguments:
        image {array} -- 3D array of pixel data
        new_shape {3-tuple} -- expected output shape

    Returns:
        3D array of desired shape, padded with zeroes
    """
    reshaped_image = np.zeros(new_shape)
    
    # Copy the original image into the top-left corner of the new array
    x_min = min(image.shape[0], new_shape[0])
    y_min = min(image.shape[1], new_shape[1])
    z_min = min(image.shape[2], new_shape[2])
    
    reshaped_image[:x_min, :y_min, :z_min] = image[:x_min, :y_min, :z_min]
    
    return reshaped_image

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        # Store original shape to crop back later
        original_shape = volume.shape
        
        # Normalize the volume
        volume_min = np.min(volume)
        volume_max = np.max(volume)
        if volume_max > volume_min:
            volume = (volume - volume_min) / (volume_max - volume_min)
        
        # Pad the volume to conform to patch size (keeping X dimension, padding Y and Z)
        # The model expects Y and Z dimensions to be patch_size
        target_shape = (volume.shape[0], self.patch_size, self.patch_size)
        volume_padded = med_reshape(volume, target_shape)
        
        # Run inference on the padded volume
        prediction = self.single_volume_inference(volume_padded)
        
        # Crop the prediction back to original Y and Z dimensions
        prediction_cropped = prediction[:original_shape[0], :original_shape[1], :original_shape[2]]
        
        return prediction_cropped

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size
        Uses batched processing for improved GPU efficiency

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()
        slices = []
        batch_size = 16
        num_slices = volume.shape[0]
        
        with torch.no_grad():
            for start_idx in range(0, num_slices, batch_size):
                end_idx = min(start_idx + batch_size, num_slices)
                batch_slices = volume[start_idx:end_idx, :, :]
                batch_tensor = torch.from_numpy(batch_slices[:, None, :, :]).float().to(self.device)
                prediction = self.model(batch_tensor)
                prediction_mask = torch.argmax(prediction, dim=1)
                prediction_np = prediction_mask.cpu().numpy()

                for i in range(prediction_np.shape[0]):
                    slices.append(prediction_np[i])
        
        prediction_volume = np.stack(slices, axis=0)
        
        return prediction_volume