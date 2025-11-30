"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

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
        
        raise NotImplementedError

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

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        
        batch_size = 16
        num_slices = volume.shape[0]
        
        with torch.no_grad():
            for start_idx in range(0, num_slices, batch_size):
                end_idx = min(start_idx + batch_size, num_slices)
                
                # Extract batch of 2D slices
                batch_slices = volume[start_idx:end_idx, :, :]
                
                # Add channel dimension: [batch, 1, H, W]
                batch_tensor = torch.from_numpy(batch_slices[:, None, :, :]).float().to(self.device)
                prediction = self.model(batch_tensor)
                
                # Get the predicted class for each pixel (argmax over class dimension)
                # prediction shape: [batch, num_classes, H, W]
                prediction_mask = torch.argmax(prediction, dim=1)
                
                # Convert to numpy: [batch, H, W]
                prediction_np = prediction_mask.cpu().numpy()
                
                for i in range(prediction_np.shape[0]):
                    slices.append(prediction_np[i])
        
        prediction_volume = np.stack(slices, axis=0)
        
        return prediction_volume 
