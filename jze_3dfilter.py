import os
import pathlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametersMatrix(nn.Module):
    """
    Custom Linear layer that represents a matrix of learnable parameters.
    It does not have any input and returns the learnable weights.

    Attributes:
    - size0 (int): Number of rows in the matrix.
    - size1 (int): Number of columns in the matrix.
    - weights (nn.Parameter): Learnable parameter representing the matrix.
    """

    def __init__(self, size0, size1):
        """
        Initializes the ParametersMatrix.

        Parameters:
        - size0 (int): Number of rows in the matrix.
        - size1 (int): Number of columns in the matrix.
        """
        super().__init__()
        self.size0, self.size1 = size0, size1

        # Create a learnable parameter representing the matrix
        weights = torch.Tensor(size0, size1)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.

        # Initialize weights
        # Uncomment one of the following initialization methods based on your preference

        # Initialize weights using Kaiming Uniform initialization with a scaling factor
        #nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        # Initialize weights using Kaiming Uniform initialization with a specific scale (e.g., 1e+2)
        #nn.init.kaiming_uniform_(self.weights, a=1e+3)

        # Initialize weights to zeros
        nn.init.zeros_(self.weights)

    def forward(self):
        """
        Defines the forward pass of the ParametersMatrix.

        Returns:
        - torch.Tensor: Learnable weights representing the matrix.
        """
        return self.weights


def sumsum(x, backend=torch):
    """
    Computes the sum of the input tensor 'x' along both axes using the specified backend.

    Parameters:
    - x (torch.Tensor): Input tensor.
    - backend (module): The backend module for tensor operations (default is torch).

    Returns:
    - torch.Tensor: Sum of the input tensor along both axes.
    """
    sum_along_axis0 = backend.sum(x, axis=0, keepdims=True)
    sum_along_both_axes = backend.sum(sum_along_axis0, axis=1, keepdims=True)
    return sum_along_both_axes
    

def trajectory_length(x, y, z, do_sqrt=True):
    """
    Calculates the length of a 3D trajectory given the coordinates along the x, y, and z axes.

    :param x: array of x coordinates (batch length firts).
    :param y: array of y coordinates (batch length firts).
    :param z: array of z coordinates (batch length firts).
    :return: Length of the trajectory.
    """
    x0 = x[0:(-2), :]
    x1 = x[1:(-1), :]
    y0 = y[0:(-2), :]
    y1 = y[1:(-1), :]
    z0 = z[0:(-2), :]
    z1 = z[1:(-1), :]
    delta_x = x1 - x0
    delta_y = y1 - y0
    delta_z = z1 - z0
    l = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
    if do_sqrt:
        l = torch.sqrt(l)
    l = sumsum(l)
    return l 


def rotate3D(x, y, z, alpha, beta, gamma, p_x, p_y, p_z):
    """
    Rotates 3D coordinates around a specified axis by a given angle.

    :param x: array of x coordinates.
    :param y: array of y coordinates.
    :param z: array of z coordinates.
    :param alpha: array of x angles.
    :param beta: array of y angles.
    :param gamma: array of z angles.
    :return: Rotated x, y, and z coordinates.
    """
    sin_alpha = torch.sin(alpha)
    cos_alpha = torch.cos(alpha)
    sin_beta = torch.sin(beta)
    cos_beta = torch.cos(beta)
    sin_gamma = torch.sin(gamma)
    cos_gamma = torch.cos(gamma)
    A = [
        [cos_beta * cos_gamma, sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma, cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma], 
        [cos_beta * sin_gamma, sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma, cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma],
        [-sin_beta,            sin_alpha * cos_beta,                                     cos_alpha * cos_beta],
    ]

    x2 = x - p_x
    y2 = y - p_y
    z2 = z - p_z

    x_rot = A[0][0] * x2 + A[0][1] * y2 + A[0][2] * z2 + p_x
    y_rot = A[1][0] * x2 + A[1][1] * y2 + A[1][2] * z2 + p_y
    z_rot = A[2][0] * x2 + A[2][1] * y2 + A[2][2] * z2 + p_z
    return x_rot, y_rot, z_rot
    

def bias(x, y, z, bias_x, bias_y, bias_z):
    """
    Adds bias values to corresponding coordinates.

    Parameters:
    - x (torch.Tensor): x-coordinates.
    - y (torch.Tensor): y-coordinates.
    - z (torch.Tensor): z-coordinates.
    - bias_x (torch.Tensor): Bias value to add to x-coordinates.
    - bias_y (torch.Tensor): Bias value to add to y-coordinates.
    - bias_z (torch.Tensor): Bias value to add to z-coordinates.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updated coordinates (x + bias_x, y + bias_y, z + bias_z).
    """

    # Add bias values to corresponding coordinates
    updated_x = x + bias_x
    updated_y = y + bias_y
    updated_z = z + bias_z
    return updated_x, updated_y, updated_z
    

def apply_shift(x, shift):
    """
    Applies a shift to the input tensor along the first axis.

    Parameters:
    - x (torch.Tensor): Input tensor.
    - shift (int): Number of positions to shift. If shift is 0, returns the original tensor.

    Returns:
    - torch.Tensor: Tensor after applying the shift.
    """
    # Check if shift is zero, return the original tensor
    if shift == 0:
        return x
  
    # Get the size of the input tensor
    x_size = x.size()
    
    # Extract the first row and the last row of the input tensor
    x0 = x[0:1, 0:x_size[1]]
    xT = x[(x_size[0] - 1):(x_size[0]), 0:x_size[1]]
    
    # Repeat the first and last rows 'abs(shift)' times
    x0 = x0.repeat(abs(shift), 1)
    xT = xT.repeat(abs(shift), 1)
    
    # Concatenate the repeated rows with the original tensor along the first axis
    y = torch.cat([x0, x, xT], axis=0)
    
    # Adjust the output based on the sign of the shift
    if shift < 0:
        return y[0:x_size[0], 0:x_size[1]]
    return y[(2 * shift):(2 * shift + x_size[0]), (0):(x_size[1])]


def mix_shifts(x, shifts, w_mix):
    """
    Applies a weighted mixture of shifts to the input tensor.

    Parameters:
    - x (torch.Tensor): Input tensor.
    - shifts (list of int): List of shifts to apply to the input tensor.
    - w_mix (list of float): List of weights corresponding to each shift.

    Returns:
    - torch.Tensor: Tensor after applying the mixture of shifts.
    """
    # Initialize the result tensor
    y = 0
    
    # Get the number of shifts
    n_shifts = len(shifts)
    
    # Iterate through each shift
    for i_shift in range(n_shifts):
        # Get the weight for the current shift
        w = w_mix[i_shift]
        # Apply the current shift and add it to the result tensor
        y = y + w * apply_shift(x, shifts[i_shift])
    
    return y


def wmse(x, tar_x, tar_w, epsilon=1e-10):
    """
    Computes the weighted mean squared error between two tensors.

    Parameters:
    - x (torch.Tensor): Input tensor.
    - tar_x (torch.Tensor): Target tensor.
    - tar_w (torch.Tensor): Weights corresponding to the target tensor.
    - epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

    Returns:
    - torch.Tensor: Weighted mean squared error between x and tar_x.
    """
    # Compute the element-wise difference between the input and target tensors
    delta = x - tar_x
    # Compute the squared element-wise difference
    delta2 = delta * delta
    # Compute the sum of squared differences weighted by the target weights
    sum_wdelta2 = sumsum(delta2 * tar_w)
    # Compute the sum of target weights
    sum_w = sumsum(tar_w)
    
    # Compute the weighted mean squared error
    e = sum_wdelta2 / (sum_w + epsilon)
    return e
    

def norm_coordinates(x, y, w, backend=torch, epsilon=1e-10):
    """
    Normalize coordinates using weights and calculate additional statistics.

    Parameters:
    - x (torch.Tensor): Input x-coordinates.
    - y (torch.Tensor): Input y-coordinates.
    - w (torch.Tensor): Weights associated with the coordinates.
    - backend (module, optional): The backend module for tensor operations (default is torch).
    - epsilon (float, optional): Small value to avoid division by zero (default is 1e-10).

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
      - x_norm (torch.Tensor): Normalized x-coordinates.
      - y_norm (torch.Tensor): Normalized y-coordinates.
      - x0 (torch.Tensor): Mean of x-coordinates.
      - y0 (torch.Tensor): Mean of y-coordinates.
      - px0 (torch.Tensor): Constant weighted mean of x-coordinates.
      - py0 (torch.Tensor): Constant weighted mean of y-coordinates.
      - scale (torch.Tensor): Scaling factor based on weighted variance.
    """
    # Calculate weighted x and y
    wx = w * x
    wy = w * y

    # Calculate sums for normalization
    sumw = backend.sum(w, axis=1, keepdims=True)
    sumwx = backend.sum(wx, axis=1, keepdims=True)
    sumwy = backend.sum(wy, axis=1, keepdims=True)

    # Calculate normalized means
    px0 = sumwx / (sumw + epsilon)
    py0 = sumwy / (sumw + epsilon)

    # Subtract normalized means from coordinates
    x = x - px0
    y = y - py0

    # Recalculate weighted x and y
    wx = w * x
    wy = w * y

    # Calculate sums using sumsum function
    sumw = sumsum(w, backend=backend)
    sumwx = sumsum(wx, backend=backend)
    sumwy = sumsum(wy, backend=backend)
    sumwx2 = sumsum(wx * x, backend=backend)
    sumwy2 = sumsum(wy * y, backend=backend)

    # Calculate additional statistics
    x0 = sumwx / (sumw + epsilon)
    y0 = sumwy / (sumw + epsilon)
    sum0 = sumw + sumw
    sum1 = sumwx + sumwy
    sum2 = sumwx2 + sumwy2
    mu = sum1 / (sum0 + epsilon)
    sigma2 = sum2 / (sum0 + epsilon) - mu * mu
    scale = backend.sqrt(sigma2)

    # Normalize coordinates using calculated statistics
    x_norm = (x - x0) / scale
    y_norm = (y - y0) / scale

    return x_norm, y_norm, x0, y0, px0, py0, scale
    

def norm_coordinate(x, x0, px0, scale, backend=torch, epsilon=1e-10):
    """
    Normalize a single coordinate using given statistics.

    Parameters:
    - x (torch.Tensor): Input coordinate to be normalized.
    - x0 (torch.Tensor): Mean of the coordinates.
    - px0 (torch.Tensor): Weighted mean of the coordinates.
    - scale (torch.Tensor): Scaling factor based on weighted variance.
    - backend (module, optional): The backend module for tensor operations (default is torch).
    - epsilon (float, optional): Small value to avoid division by zero (default is 1e-10).

    Returns:
    - torch.Tensor: Normalized coordinate.
    """
    # Subtract weighted mean from coordinate
    x = x - px0

    # Normalize coordinate using mean and scale
    x_norm = (x - x0) / (scale + epsilon)

    return x_norm
    
    
def denorm_coordinates(x_norm, x0, p0, scale, backend=torch):
    """
    Denormalize the normalized coordinates using specified parameters.

    Parameters:
    - x_norm (torch.Tensor): Normalized coordinates.
    - x0 (torch.Tensor): Mean coordinate after normalization.
    - p0 (torch.Tensor): Weighted mean coordinate before normalization.
    - scale (torch.Tensor): Scaling factor used for normalization.

    Returns:
    - torch.Tensor: Denormalized coordinates.
    """
    # Denormalize coordinates using specified parameters
    x_denorm = x_norm * scale + x0 + p0
    return x_denorm


class ModelCameraPlain(nn.Module):
    """
    Plain camera model that does not apply any transformation.

    Attributes:
    - device (str): Device on which the model is executed ("cpu" or "cuda").
    """

    def __init__(self, device="cpu"):
        """
        Initializes the ModelCameraPlain.

        Parameters:
        - device (str, optional): Device on which the model is executed ("cpu" or "cuda"). Defaults to "cpu".
        """
        super().__init__()
        self.device = device
        # Uncomment the following lines if using ParametersMatrix for logzoom
        # self.logzoom = ParametersMatrix(T, 1)

    
    def forward(self, x, y, z, epsilon=1e-10):
        """
        Forward pass of the camera model.

        Parameters:
        - x (torch.Tensor): x-coordinates.
        - y (torch.Tensor): y-coordinates.
        - z (torch.Tensor): z-coordinates.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - x_transformed (torch.Tensor): Transformed x-coordinates.
            - y_transformed (torch.Tensor): Transformed y-coordinates.
            - e_reg (torch.Tensor): Regularization term (always 0 in the plain camera model).
        """
        # Uncomment the following lines if using ParametersMatrix for logzoom
        # logzoom = self.logzoom()
        # zoom = torch.exp(logzoom)
        # e_reg = torch.sum(zoom * zoom, axis=0, keepdims=True)
        # return zoom * x, zoom * y, e_reg

        # If not using any transformation, return original coordinates and zero regularization term
        e_reg = 0
        return x, y, e_reg
        


class ModelScene(nn.Module):
    """
    Combined scene model consisting of an object model and a camera model.

    Attributes:
    - device (str): Device on which the model is executed ("cpu" or "cuda").
    - model_object (nn.Module): Object model representing the scene's objects.
    - model_camera (nn.Module): Camera model representing the scene's camera.
    """

    def __init__(self, model_object, model_camera, device="cpu"):
        """
        Initializes the ModelScene.

        Parameters:
        - model_object (nn.Module): Object model representing the scene's objects.
        - model_camera (nn.Module): Camera model representing the scene's camera.
        - device (str, optional): Device on which the model is executed ("cpu" or "cuda"). Defaults to "cpu".
        """
        super().__init__()
        self.device = device
        self.model_object = model_object
        self.model_camera = model_camera

    def forward(self, epsilon=1e-10):
        """
        Forward pass of the scene model.

        Parameters:
        - epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

        Returns:
        - List[torch.Tensor]: List of tensors representing various outputs of the scene model.
            - For the object model, these are x, y, z, and regularization term.
            - For the camera model, these are transformed x, transformed y, and regularization term.
        """
        # Forward pass through the object model
        results_obj = list(self.model_object(epsilon=epsilon))
        e_reg_obj = results_obj[-1]
        results_obj = results_obj[0:(len(results_obj) - 1)]

        # Initialize camera regularization term
        e_reg_cam = 0
        results_cam = []

        # Forward pass through the camera model for each object result
        for i in range(0, len(results_obj), 3):
            x_obj, y_obj, z_obj = results_obj[i + 0], results_obj[i + 1], results_obj[i + 2]
            x_cam, y_cam, e_reg_cam_key = self.model_camera(x_obj, y_obj, z_obj, epsilon=epsilon)
            results_cam.append(x_cam)
            results_cam.append(y_cam)
            e_reg_cam = e_reg_cam + e_reg_cam_key

        # Return combined results along with regularization terms
        return results_obj + results_cam + [e_reg_obj, e_reg_cam]


class ModelSceneMultipleObjects(nn.Module):
    def __init__(self, models_object, model_camera, device="cpu"):
        """
        Initialize the ModelSceneMultipleObjects.

        Parameters:
            models_object (list): List of models representing objects in the scene.
            model_camera (nn.Module): Model representing the camera.
            device (str): Device to run the model on (default is "cpu").

        """
        super().__init__()
        self.device = device
        self.n_objects = len(models_object)
        
        # Set attributes for object models
        for i_object in range(self.n_objects):
            setattr(self, "model_object_%d" % i_object, models_object[i_object])
        
        # Set attribute for the camera model
        self.model_camera = model_camera

    def forward(self, epsilon=1e-10):
        """
        Forward pass of the ModelSceneMultipleObjects.

        Parameters:
            epsilon (float): A small value to avoid division by zero (default is 1e-10).

        Returns:
            list: List of results for the objects and camera, and a regularization error.

        """
        e_reg_obj = 0
        e_reg_cam = 0
        results_obj = []  # Results for objects
        results_cam = []  # Results for camera
        
        # Forward pass for each object
        for i_object in range(self.n_objects):
            model_object = getattr(self, "model_object_%d" % i_object)
            results_obj_i = list(model_object(epsilon=epsilon))  # Forward pass for object i
            # Append object results to the overall results
            results_obj += results_obj_i[0:(len(results_obj_i) - 1)]  # Exclude regularization error
            e_reg_obj += results_obj_i[-1]  # Accumulate regularization error for objects
            
            # Forward pass for camera given object coordinates
            for i in range(0, len(results_obj_i) - 1, 3):
                x_obj, y_obj, z_obj = results_obj_i[i], results_obj_i[i + 1], results_obj_i[i + 2]
                x_cam, y_cam, e_reg_cam_key = self.model_camera(x_obj, y_obj, z_obj, epsilon=epsilon)
                # Append camera results to the overall results
                results_cam.append(x_cam)
                results_cam.append(y_cam)
                e_reg_cam += e_reg_cam_key  # Accumulate regularization error for camera

        # Return the concatenated results and regularization energies
        return list(results_obj) + results_cam + [e_reg_obj, e_reg_cam]


def np2torch(d, device):
    """
    Convert NumPy arrays in a dictionary to PyTorch tensors and move them to the specified device.

    Parameters:
    - d (dict): Input dictionary containing NumPy arrays.
    - device (torch.device): The device to which the tensors should be moved.

    Returns:
    - dict: Dictionary with NumPy arrays converted to PyTorch tensors and moved to the specified device.
    """
    # Iterate through keys in the dictionary
    for key in d:
        x = d[key]

        # Check if the value is a NumPy array
        if type(x) is np.ndarray:
            # Convert NumPy array to PyTorch tensor and move to the specified device
            d[key] = torch.tensor(x).to(device)

    return d
    

def resolve_scene(model_scene, losses, lr, q_reg_obj, q_reg_cam, n_epochs, device, print_loss=False):
    """
    Resolve the scene using optimization.

    Parameters:
    - model_scene (nn.Module): The scene model to be resolved.
    - losses (list of functions): List of loss functions for each object.
    - lr (float): Learning rate for the optimizer.
    - q_reg_obj (float): Regularization weight for object-related terms.
    - q_reg_cam (float): Regularization weight for camera-related terms.
    - n_epochs (int): Number of optimization epochs.
    - device (str): Device to which the model should be moved.
    - print_loss (bool, optional): Whether to print loss during optimization (default is False).

    Returns:
    - torch.Tensor: Predicted x, y, z coordinates of the scene.

    Raises:
    - Exception: If the provided output type is unknown.
    """
    # Set up the optimizer
    momentum = 0.1
    optimizer = torch.optim.Adam(params=model_scene.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model_scene.parameters(), lr=lr, momentum=momentum)
    
    n_losses = len(losses)
    
    results_obj_min = [None for _ in range(n_losses)]
    results_cam_min = [None for _ in range(n_losses)]
    losses_min = [None for _ in range(n_losses)]
    
    # Optimization loop
    for i_epoch in range(n_epochs):
        results = model_scene()
        n = int((len(results) - 2) / 5)
        results_obj = results[0:(3 * n)]
        results_cam = results[(3 * n):(5 * n)]
        e_reg_obj = results[-2]
        e_reg_cam = results[-1]
        
        loss = 0
        
        # Calculate loss for each object
        losses_vals = []
        for i_loss in range(n):
            x_cam, y_cam = results_cam[2 * i_loss], results_cam[2 * i_loss + 1]
            loss_i = losses[i_loss]
            loss_i_val = loss_i(x_cam, y_cam)
            loss = loss + loss_i_val
            losses_vals.append(loss_i_val)

        # Calculate total loss with regularization terms
        loss = loss + q_reg_obj * e_reg_obj + q_reg_cam * e_reg_cam

        # Perform backpropagation and optimization step
        loss.backward()
        optimizer.step()
        
        # Update minimum loss and corresponding results
        losses_vals_np = [loss_val.detach().cpu().numpy()[0, 0] for loss_val in losses_vals]
        for i_loss in range(n_losses):
            if losses_min[i_loss] is None or losses_min[i_loss] >= losses_vals_np[i_loss]:
                results_obj_min[i_loss] = [xi.detach().cpu().numpy() for xi in results_obj[(3 * i_loss):(3 * i_loss + 3)]] 
                results_cam_min[i_loss] = [xi.detach().cpu().numpy() for xi in results_cam[(2 * i_loss):(2 * i_loss + 2)]]
                losses_min[i_loss] = losses_vals_np[i_loss]
    
        # Print loss if specified
        if print_loss:
            print(i_epoch, losses_vals_np)

    # Print final loss
    if print_loss:
        print()
        print(losses_min)
    
    return results_obj_min, results_cam_min

def compute_center_and_scale(arrays_coordinates, epsilon=1e-10):
    """
    Compute the center and scale of arrays of coordinates.

    Parameters:
        arrays_coordinates (list): List of arrays of coordinates.
            [
                [[x1, wx1], [x2, wx2], ...],
                [[y1, wy1], [y2, wy2], ...],
                ...
            ]
        epsilon (float): A small value to avoid division by zero.

    Returns:
        tuple: A tuple containing the centers and the scale.

    """
    sum_w_all_coordinates = 0
    sum_wx_all_coordinates = 0
    sum_wx2_all_coordinates = 0
    c = []
    
    # Iterate over each array of coordinates
    for arrays_coordinate in arrays_coordinates:
        sum_w = 0
        sum_wx = 0
        sum_wx2 = 0
        
        # Iterate over each coordinate and weight pair in the array
        for x, w in arrays_coordinate:
            # Compute weighted sums
            sum_w += sumsum(w, backend=np)
            sum_wx += sumsum(w * x, backend=np)
            sum_wx2 += sumsum(w * x * x, backend=np)
            
            # Accumulate sums for all coordinates
            sum_w_all_coordinates += sum_w
            sum_wx_all_coordinates += sum_wx
            sum_wx2_all_coordinates += sum_wx2
        
        # Compute the center for the current array of coordinates
        c_x = sum_wx / (sum_w + epsilon)
        c.append(c_x[0, 0]) 
    
    # Compute the mean and variance of all coordinates
    mu = sum_wx_all_coordinates / (sum_w_all_coordinates + epsilon) 
    sigma2 = (sum_wx2_all_coordinates / (sum_w_all_coordinates + epsilon)) - mu * mu
    
    # Ensure non-negative variance
    if sigma2 < 0:
        sigma2 = 0
    
    # Compute the scale as the square root of the variance
    scale = math.sqrt(sigma2)
    
    return c, scale

