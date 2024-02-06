import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jze_3dfilter import *
from jze_3dfilter_data import *


class ModelFlexibleObject(nn.Module):


    def __init__(self, n, T, data_for_init, device="cpu"):
        """
        Initialize the ModelFlexibleObject class.

        Parameters:
            n (int): Number of nodes.
            T (int): Number of time steps.
            data_for_init (dict): Data for initialization.
            device (str): Device to run the model on (default is "cpu").
        """
        super().__init__()
        
        # Store the number of time steps and nodes
        self.T = T
        self.n = n
        self.device = device
        
        # Convert initialization data to torch tensors
        self.data_for_init = np2torch(data_for_init, device)
        
        # Parameters for rigid components
        self.x_rigid = ParametersMatrix(1, n)
        self.y_rigid = ParametersMatrix(1, n)
        self.z_rigid = ParametersMatrix(1, n)

        # Parameters for flexible components
        self.x_flexible = ParametersMatrix(T, n)
        self.y_flexible = ParametersMatrix(T, n)
        self.z_flexible = ParametersMatrix(T, n)

        # Bias parameters
        self.bias_x = ParametersMatrix(T, 1)
        self.bias_y = ParametersMatrix(T, 1)
        self.bias_z = ParametersMatrix(T, 1)

        # Rotation parameters
        self.rotations_x = ParametersMatrix(T, 1)
        self.rotations_y = ParametersMatrix(T, 1)
        self.rotations_z = ParametersMatrix(T, 1)
        
        # Parameters for probabilities of rotations
        self.p_rotations_x = ParametersMatrix(1, 1)
        self.p_rotations_y = ParametersMatrix(1, 1)
        self.p_rotations_z = ParametersMatrix(1, 1)
        
        # Shifts for mixing
        shifts_delta = 10
        self.shifts_mix = list(range(-shifts_delta, shifts_delta + 1))
        
        # Weight mixing matrix
        self.w_mix = ParametersMatrix(1, len(self.shifts_mix))

        # Parameter for initial scale of z-values
        self.logq_z0 = ParametersMatrix(1, 1)
        
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(1)


    def forward(self, epsilon=1e-10):
        """
        Perform forward pass of the model.
    
        Parameters:
            epsilon (float): A small value to avoid division by zero (default is 1e-10).
        
        Returns:
            list: List containing the forward pass results.
        """
        # Extract initialization data
        x_rec, y_rec, z_rec, w_rec, use_z_rec = self.data_for_init["x_rec"], self.data_for_init["y_rec"], self.data_for_init["z_rec"], self.data_for_init["w_rec"], self.data_for_init["use_z_rec"]
        
        # Get the number of time steps and nodes
        T = x_rec.size()[0]
        n = x_rec.size()[1]
        
        # Softmax activation for mixing weights
        w_mix = self.softmax(self.w_mix())
        w_mix = torch.reshape(w_mix, (len(self.shifts_mix), ))
        mix = lambda x: mix_shifts(x, self.shifts_mix, w_mix)
    
        # Set standard deviations for different components
        sigma = 1e-1
        sigma_bias = sigma
        sigma_rigid = sigma
        sigma_flexible = sigma
        sigma_rotations = sigma
        sigma_p_rotations = sigma
    
        # Mix probabilities for rotations
        p_rotations_x = sigma_p_rotations * mix(self.p_rotations_x())
        p_rotations_y = sigma_p_rotations * mix(self.p_rotations_y())
        p_rotations_z = sigma_p_rotations * mix(self.p_rotations_z())
    
        # Normalize z-values
        sumw = sumsum(w_rec)
        sumwz = sumsum(w_rec * z_rec)
        sumwz2 = sumsum(w_rec * z_rec * z_rec)
        mu_z = sumwz / (sumw + epsilon)
        sigma2_z = sumwz2 / (sumw + epsilon) - mu_z * mu_z
        z_rec = (z_rec - mu_z) / torch.sqrt(sigma2_z + epsilon)
        
        # Scale z-values
        q_z0 = torch.exp(self.logq_z0())
        z_rec = q_z0 * z_rec
        
        # Calculate means for rigid components
        sumw = torch.sum(w_rec, axis=0, keepdims=True)
        sumwx = torch.sum(w_rec * x_rec, axis=0, keepdims=True)
        sumwy = torch.sum(w_rec * y_rec, axis=0, keepdims=True)
        sumwz = torch.sum(w_rec * z_rec, axis=0, keepdims=True)
        x_rigid0 = sumwx / (sumw + 1e-10)
        y_rigid0 = sumwy / (sumw + 1e-10)
        z_rigid0 = sumwz / (sumw + 1e-10)
        
        # Calculate biases for flexible components
        bias_x0 = torch.sum((x_rec - x_rigid0) * w_rec, axis=1, keepdims=True) / (torch.sum(w_rec, axis=1, keepdims=True) + 1e-10)
        bias_y0 = torch.sum((y_rec - y_rigid0) * w_rec, axis=1, keepdims=True) / (torch.sum(w_rec, axis=1, keepdims=True) + 1e-10)
        bias_z0 = torch.sum((z_rec - z_rigid0) * w_rec, axis=1, keepdims=True) / (torch.sum(w_rec, axis=1, keepdims=True) + 1e-10)
        
        # Apply scaling to parameters
        x_rigid = sigma_rigid * self.x_rigid() + x_rigid0
        y_rigid = sigma_rigid * self.y_rigid() + y_rigid0
        z_rigid = sigma_rigid * self.z_rigid() + z_rigid0
        
        # Initialize rotation parameters
        rotation0_x = 0
        rotation0_y = 0
        rotation0_z = 0
        
        # Calculate flexible components without rotation
        x_flexible0 = x_rec - (x_rigid0 + bias_x0)
        y_flexible0 = y_rec - (y_rigid0 + bias_y0)
        z_flexible0 = z_rec - (z_rigid0 + bias_z0)
        
        # Apply scaling to flexible components
        x0_flexible = sigma_flexible * self.x_flexible() + x_flexible0
        y0_flexible = sigma_flexible * self.y_flexible() + y_flexible0
        z0_flexible = sigma_flexible * self.z_flexible() + z_flexible0
        
        # Apply mixing to bias parameters
        bias_x = sigma_bias * mix(self.bias_x()) + bias_x0
        bias_y = sigma_bias * mix(self.bias_y()) + bias_y0
        bias_z = sigma_bias * mix(self.bias_z()) + bias_z0
    
        # Apply mixing to rotation parameters
        rotations_x = sigma_rotations * mix(self.rotations_x()) + rotation0_x
        rotations_y = sigma_rotations * mix(self.rotations_y()) + rotation0_y
        rotations_z = sigma_rotations * mix(self.rotations_z()) + rotation0_z
    
        # Apply mixing to flexible components
        x_flexible = x_rigid + mix(x0_flexible)
        y_flexible = y_rigid + mix(y0_flexible)
        z_flexible = z_rigid + mix(z0_flexible)
    
        # Rotate rigid and flexible components
        x_rigid_rotation, y_rigid_rotation, z_rigid_rotation = rotate3D(x_rigid, y_rigid, z_rigid, rotations_x, rotations_y, rotations_z, p_rotations_x, p_rotations_y, p_rotations_z)
        x_flexible_rotation, y_flexible_rotation, z_flexible_rotation = rotate3D(x_flexible, y_flexible, z_flexible, rotations_x, rotations_y, rotations_z, p_rotations_x, p_rotations_y, p_rotations_z)
            
        # Apply bias to rigid and flexible components
        x_rigid_bias, y_rigid_bias, z_rigid_bias = bias(x_rigid, y_rigid, z_rigid, bias_x, bias_y, bias_z)
        x_flexible_bias, y_flexible_bias, z_flexible_bias = bias(x_flexible, y_flexible, z_flexible, bias_x, bias_y, bias_z)
    
        # Rotate and apply bias to rotated rigid and flexible components
        x_rigid_rotation_bias, y_rigid_rotation_bias, z_rigid_rotation_bias = rotate3D(x_rigid_rotation, y_rigid_rotation, z_rigid_rotation, rotations_x, rotations_y, rotations_z, p_rotations_x, p_rotations_y, p_rotations_z)
        x_flexible_rotation_bias, y_flexible_rotation_bias, z_flexible_rotation_bias = rotate3D(x_flexible_rotation, y_flexible_rotation, z_flexible_rotation, rotations_x, rotations_y, rotations_z, p_rotations_x, p_rotations_y, p_rotations_z)
        
        # Calculate regularization energy
        e_reg = 0
        #e_reg = e_reg + 1.0 * trajectory_length(x_rigid_rotation_bias, y_rigid_rotation_bias, z_rigid_rotation_bias, do_sqrt=False) / (self.T * self.n)
        e_reg = e_reg + 1.0 * trajectory_length(x_flexible_rotation_bias, y_flexible_rotation_bias, z_flexible_rotation_bias) / (self.T * self.n)
    
        # Return the results of the forward pass
        return [
            #x_rigid_bias, y_rigid_bias, z_rigid_bias,
            x_flexible_rotation_bias, y_flexible_rotation_bias, z_flexible_rotation_bias,
            x_rigid_rotation_bias, y_rigid_rotation_bias, z_rigid_rotation_bias,
            #x_flexible_bias, y_flexible_bias, z_flexible_bias,
            e_reg, 
        ]
    
