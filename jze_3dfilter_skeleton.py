import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from jze_3dfilter import *


def find_order(structure):
    """
    Finds the order of edges to build a directed graph with the given structure.

    Parameters:
    - structure (dict): A dictionary representing the graph structure with the following keys:
        - "root" (int): Index of the root node.
        - "#nodes" (int): Number of nodes in the graph.
        - "edges" (list): List of edges, where each edge is represented as a tuple (i_from, i_to, i_length).

    Returns:
    - order (list): A list of edge indices indicating the order in which edges should be traversed to cover the entire graph.
    
    Raises:
    - Exception: If the order is incomplete, i.e., there are unvisited nodes in the graph.
    """
    # Extract information from the input structure
    i_root = structure["root"]
    n_nodes = structure["#nodes"]
    n_edges = len(structure["edges"])
    
    # Initialize variables for tracking visited nodes and the order
    found = [False for _ in range(n_nodes)]
    found[i_root] = True
    change = True
    order = []

    # Iterate until no further change in the order
    while change:
        change = False
        
        # Iterate through each edge in the structure
        for i_edge in range(n_edges):
            i_from, i_to, i_length = structure["edges"][i_edge]
            
            # Check if the starting node is already visited and the ending node is not visited
            if found[i_from] and not found[i_to]:
                # Mark the ending node as visited, indicating a change in order
                change = True
                found[i_to] = True
                # Add the current edge index to the order list
                order.append(i_edge)

    # Check if there are any nodes left unvisited
    for i_node in range(n_nodes):
        if not found[i_node]:
            raise Exception("The order is incomplete.")

    # Return the order of edges to traverse the graph
    return order


def init_skeleton_lengths(x_rec, y_rec, z_rec, w_rec, structure, length0_q, use_z_rec=False, epsilon=1e-10):
    """
    Initializes the lengths of the edges in a skeleton based on recorded coordinates.

    Parameters:
    - x_rec (torch.Tensor): x-coordinates of the reconstructed nodes.
    - y_rec (torch.Tensor): y-coordinates of the reconstructed nodes.
    - z_rec (torch.Tensor): z-coordinates of the reconstructed nodes (optional if use_z_rec is True).
    - w_rec (torch.Tensor): Weights associated with each reconstructed node.
    - structure (dict): A dictionary representing the skeleton structure with the following keys:
        - "#lengths" (int): Number of lengths in the skeleton.
        - "edges" (list): List of edges, where each edge is represented as a tuple (i_from, i_to, i_length).
    - length0_q (float): Scaling factor for initializing the lengths.
    - use_z_rec (bool, optional): If True, z_rec is used; otherwise, z_rec is ignored. Defaults to False.
    - epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

    Returns:
    - lengths_0 (torch.Tensor): Initialized lengths of the edges in the skeleton.
    """
    # Initialize lists to store intermediate results
    lengths_0 = [0 for i_length in range(structure["#lengths"])] 
    lengths_0_sumw = [0 for i_length in range(structure["#lengths"])] 

    # Iterate through each edge in the structure
    for i_from, i_to, i_length in structure["edges"]:
         # Calculate weights, differences in coordinates, and length for each edge
         w = w_rec[:, i_to:(i_to + 1)] * w_rec[:, i_from:(i_from + 1)]
         d_x = (x_rec[:, (i_to):(i_to + 1)] - x_rec[:, (i_from):(i_from + 1)])
         d_y = (y_rec[:, (i_to):(i_to + 1)] - y_rec[:, (i_from):(i_from + 1)])
         d_z = (z_rec[:, (i_to):(i_to + 1)] - z_rec[:, (i_from):(i_from + 1)]) if use_z_rec else 0
         n = torch.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)
         
         # Accumulate lengths and sum of weights for each length
         lengths_0[i_length] = lengths_0[i_length] + torch.sum(w * n, axis=0, keepdims=True)
         lengths_0_sumw[i_length] = lengths_0_sumw[i_length] + torch.sum(w, axis=0, keepdims=True)

    # Concatenate and normalize lengths based on accumulated weights
    lengths_0 = torch.cat(lengths_0, axis=1)
    lengths_0_sumw = torch.cat(lengths_0_sumw, axis=1)
    lengths_0 = lengths_0 / (lengths_0_sumw + epsilon)

    # Scale the lengths by the specified factor
    lengths_0 = length0_q * lengths_0
    
    return lengths_0


def init_direction(from_x, from_y, from_z, to_x, to_y, to_z, L, use_z_rec, epsilon=1e-10):
    d_x = to_x - from_x
    d_y = to_y - from_y
    if use_z_rec:
        d_z = to_z - from_z
    else:
        #L_xy = torch.sqrt(d_x * d_x + d_y * d_y)
        n = (L * (d_x + d_y)) / (to_x + to_y - from_x - from_y)
        n2 = n * n
        d_z2 = n2 - (d_x * d_x + d_y * d_y)
        d_z2[d_z2 < 0] = 0
        d_z = torch.sqrt(d_z2)         

    n = torch.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)
    direction_x_0 = d_x / (n + epsilon)
    direction_y_0 = d_y / (n + epsilon)
    direction_z_0 = d_z / (n + epsilon)
    return  direction_x_0, direction_y_0, direction_z_0


def init_direction(from_x, from_y, from_z, to_x, to_y, to_z, L, use_z_rec, epsilon=1e-10):
    """
    Initialize the direction vector between two points.

    Parameters:
    - from_x (torch.Tensor): x-coordinate of the starting point.
    - from_y (torch.Tensor): y-coordinate of the starting point.
    - from_z (torch.Tensor): z-coordinate of the starting point.
    - to_x (torch.Tensor): x-coordinate of the ending point.
    - to_y (torch.Tensor): y-coordinate of the ending point.
    - to_z (torch.Tensor): z-coordinate of the ending point.
    - L (torch.Tensor): Length of the edge connecting the two points.
    - use_z_rec (bool): Indicates whether z-coordinates are used for recordings
    - epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Direction vector (x, y, z) between two points.
    """
    # Calculate the displacement vectors
    d_x = to_x - from_x
    d_y = to_y - from_y

    # Calculate the displacement along the z-axis based on reconstruction method
    if use_z_rec:
        d_z = to_z - from_z
    else:
        # Calculate the length along the xy-plane
        n = (L * (d_x + d_y)) / (to_x + to_y - from_x - from_y)
        n2 = n * n

        # Calculate the squared displacement along the z-axis
        d_z2 = n2 - (d_x * d_x + d_y * d_y)
        d_z2[d_z2 < 0] = 0  # Ensure non-negative values
        d_z = torch.sqrt(d_z2)

    # Calculate the length of the displacement vector
    n = torch.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)

    # Calculate the normalized direction vector
    direction_x_0 = d_x / (n + epsilon)
    direction_y_0 = d_y / (n + epsilon)
    direction_z_0 = d_z / (n + epsilon)

    return direction_x_0, direction_y_0, direction_z_0


def init_skeleton(x_rec, y_rec, z_rec, w_rec, structure, order, length0_q, use_z_rec=False, epsilon=1e-10):
    """
    Initializes a skeleton based on recorded coordinates and a given order of edges.

    Parameters:
    - x_rec (torch.Tensor): x-coordinates of the recorded nodes.
    - y_rec (torch.Tensor): y-coordinates of the recorded nodes.
    - z_rec (torch.Tensor): z-coordinates of the recorded nodes (optional if use_z_rec is True).
    - w_rec (torch.Tensor): Weights associated with each recorded node.
    - structure (dict): A dictionary representing the skeleton structure with the following keys:
        - "root" (int): Index of the root node.
        - "#nodes" (int): Number of nodes in the skeleton.
        - "#lengths" (int): Number of lengths in the skeleton.
        - "edges" (list): List of edges, where each edge is represented as a tuple (i_from, i_to, i_length).
    - order (list): A list of edge indices indicating the order in which edges should be traversed.
    - length0_q (float): Scaling factor for initializing the lengths.
    - use_z_rec (bool, optional): If True, z_rec is used; otherwise, z_rec is ignored. Defaults to False.
    - epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

    Returns:
    - root_position_x_0 (torch.Tensor): Initial x-coordinate of the root node.
    - root_position_y_0 (torch.Tensor): Initial y-coordinate of the root node.
    - root_position_z_0 (torch.Tensor): Initial z-coordinate of the root node (optional if use_z_rec is True).
    - directions_x_0 (torch.Tensor): Initial x-directions of edges in the skeleton.
    - directions_y_0 (torch.Tensor): Initial y-directions of edges in the skeleton.
    - directions_z_0 (torch.Tensor): Initial z-directions of edges in the skeleton (optional if use_z_rec is True).
    - loglengths_0 (torch.Tensor): Logarithm of the initial lengths of edges in the skeleton.
    """
    # Extract information from the input structure
    i_root = structure["root"]
    n_nodes = structure["#nodes"]

    # Initialize root positions
    root_position_x_0 = x_rec[:, i_root:(i_root + 1)]
    root_position_y_0 = y_rec[:, i_root:(i_root + 1)]
    root_position_z_0 = z_rec[:, i_root:(i_root + 1)] if use_z_rec else 0

    # Initialize lengths of edges
    lengths_0 = init_skeleton_lengths(x_rec, y_rec, z_rec, w_rec, structure, length0_q, use_z_rec, epsilon)

    # Compute logarithm of initial lengths
    loglengths_0 = torch.log(lengths_0 + epsilon)

    # Initialize lists to store direction vectors and node positions
    directions_x_0 = []
    directions_y_0 = []
    directions_z_0 = []
    pos_x = [None for _ in range(n_nodes)]
    pos_y = [None for _ in range(n_nodes)]
    pos_z = [None for _ in range(n_nodes)]
    pos_x[i_root] = root_position_x_0
    pos_y[i_root] = root_position_y_0
    pos_z[i_root] = root_position_z_0

    # Iterate through edges in the specified order
    for i_edge in order:
        i_from, i_to, i_length = structure["edges"][i_edge]

        # Calculate weights, differences in coordinates, and length for each edge
        w = w_rec[:, i_to:(i_to + 1)] * w_rec[:, i_from:(i_from + 1)]
        from_x = pos_x[i_from]
        from_y = pos_y[i_from]
        from_z = pos_z[i_from]
        to_x = x_rec[:, (i_to):(i_to + 1)]
        to_y = y_rec[:, (i_to):(i_to + 1)]
        to_z = z_rec[:, (i_to):(i_to + 1)] if use_z_rec else 0
        L = lengths_0[0, i_length]
        direction_x_0, direction_y_0, direction_z_0 = init_direction(from_x, from_y, from_z, to_x, to_y, to_z, L, use_z_rec, epsilon)
        directions_x_0.append(direction_x_0)
        directions_y_0.append(direction_y_0)
        directions_z_0.append(direction_z_0)

        # Update positions of the 'to' node
        pos_x[i_to] = pos_x[i_from] + L * direction_x_0
        pos_y[i_to] = pos_y[i_from] + L * direction_y_0
        pos_z[i_to] = pos_z[i_from] + L * direction_z_0

    # Concatenate direction vectors
    directions_x_0 = torch.cat(directions_x_0, axis=1)
    directions_y_0 = torch.cat(directions_y_0, axis=1)
    directions_z_0 = torch.cat(directions_z_0, axis=1)

    return root_position_x_0, root_position_y_0, root_position_z_0, directions_x_0, directions_y_0, directions_z_0, loglengths_0


def construct_skeleton(root_positions_x, root_positions_y, root_positions_z, directions_x, directions_y, directions_z, lengths, structure, order, epsilon=1e-10):
    """
    Constructs a 3D skeleton based on root positions, directions, and lengths.

    Parameters:
    - root_positions_x (torch.Tensor): x-coordinates of the root positions.
    - root_positions_y (torch.Tensor): y-coordinates of the root positions.
    - root_positions_z (torch.Tensor): z-coordinates of the root positions.
    - directions_x (torch.Tensor): x-components of the directions for each edge.
    - directions_y (torch.Tensor): y-components of the directions for each edge.
    - directions_z (torch.Tensor): z-components of the directions for each edge.
    - lengths (torch.Tensor): Lengths of each edge.
    - structure (dict): Dictionary describing the skeleton structure, including nodes, edges, and root information.
    - order (list): List specifying the order in which edges should be constructed.
    - epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Coordinates (x, y, z) of the constructed 3D skeleton.
    """
    n_nodes = structure["#nodes"]
    i_root = structure["root"]
    xs0 = [None for i_node in range(n_nodes)]
    ys0 = [None for i_node in range(n_nodes)]
    zs0 = [None for i_node in range(n_nodes)]
    xs0[i_root] = root_positions_x
    ys0[i_root] = root_positions_y
    zs0[i_root] = root_positions_z

    # Iterate through edges in the specified order and construct the skeleton
    for i_edge in order:
        i_from, i_to, i_length = structure["edges"][i_edge]
        length = lengths[:, i_length]
        direction_x = directions_x[:, i_edge:(i_edge + 1)]
        direction_y = directions_y[:, i_edge:(i_edge + 1)]
        direction_z = directions_z[:, i_edge:(i_edge + 1)]

        # Calculate the normalized direction vector
        norm = torch.sqrt(direction_x * direction_x + direction_y * direction_y + direction_z * direction_z) + epsilon

        # Update coordinates for the "to" node
        xs0[i_to] = xs0[i_from] + direction_x * length / norm
        ys0[i_to] = ys0[i_from] + direction_y * length / norm
        zs0[i_to] = zs0[i_from] + direction_z * length / norm

    # Concatenate the coordinates to form the 3D skeleton
    x = torch.cat(xs0, axis=1)
    y = torch.cat(ys0, axis=1)
    z = torch.cat(zs0, axis=1)

    return x, y, z


class ModelSkeleton(nn.Module):
    def __init__(self, structure, T, data_for_init, device="cpu"):
        """
        Initialize the ModelSkeleton class.

        Parameters:
        - structure (dict): Skeleton structure containing information about nodes, lengths, and edges.
        - T (int): Number of time steps.
        - data_for_init (dict): Data dictionary for initialization containing keys 'x_rec', 'y_rec', 'z_rec', 'w_rec', 'use_z_rec'.
        - device (str, optional): Device to which the model should be moved (default is "cpu").
        """
        super().__init__()
        self.T = T
        self.structure = structure
        self.device = device
        
        # Convert NumPy data to PyTorch and move to the specified device
        self.data_for_init = np2torch(data_for_init, device)
        
        # Extract information about the skeleton structure
        self.n_nodes = self.structure["#nodes"]
        self.n_lengths = self.structure["#lengths"]
        self.n_edges = len(structure["edges"])
        
        # Find the order of the skeleton
        self.order = find_order(structure)
        
        # Define learnable parameters for the model
        self.loglength0_q = ParametersMatrix(1, 1)
        self.logqz = ParametersMatrix(1, 1)
        self.loglengths = ParametersMatrix(1, self.n_lengths)
        self.root_positions_x = ParametersMatrix(T, 1)
        self.root_positions_y = ParametersMatrix(T, 1)
        self.root_positions_z = ParametersMatrix(T, 1)
        self.directions_x = ParametersMatrix(T, self.n_edges)
        self.directions_y = ParametersMatrix(T, self.n_edges)
        self.directions_z = ParametersMatrix(T, self.n_edges)
        
        # Define shifts for mixing
        shifts_delta = 10
        self.shifts_mix = list(range(-shifts_delta, shifts_delta + 1))
        
        # Initialize mixing parameters
        self.w_mix = ParametersMatrix(1, len(self.shifts_mix))
        self.softmax = nn.Softmax(1)

    def forward(self, epsilon=1e-10):
        """
        Forward pass of the model.

        Parameters:
        - epsilon (float, optional): Small value to avoid division by zero (default is 1e-10).

        Returns:
        - torch.Tensor: Predicted x, y, z coordinates of the skeleton.
        - torch.Tensor: Error term for regularization.
        """
        # Extract data for initialization
        x_rec, y_rec, z_rec, w_rec, use_z_rec = self.data_for_init["x_rec"], self.data_for_init["y_rec"], self.data_for_init["z_rec"], self.data_for_init["w_rec"], self.data_for_init["use_z_rec"]
        
        # Calculate softmax weights for mixing
        w_mix = self.softmax(self.w_mix())
        w_mix = torch.reshape(w_mix, (len(self.shifts_mix), ))
        
        # Define mixing function
        mix = lambda x: mix_shifts(x, self.shifts_mix, w_mix)
       
        # Extract learnable parameters for initialization
        length0_q = torch.exp(self.loglength0_q())
        qz = torch.exp(self.logqz())
        z_rec = qz * z_rec
    
        root_position_x_0, root_position_y_0, root_position_z_0, directions_x_0, directions_y_0, directions_z_0, loglengths_0 = init_skeleton(
            x_rec,
            y_rec,
            z_rec,
            w_rec,
            self.structure,
            self.order,
            length0_q,
            use_z_rec,
            epsilon,
        )

        # Parameter for gradual initialization
        q_param = 0.1
        
        # Apply gradual initialization to root positions
        root_positions_x = q_param * self.root_positions_x() + root_position_x_0
        root_positions_y = q_param * self.root_positions_y() + root_position_y_0
        root_positions_z = q_param * self.root_positions_z() + root_position_z_0
        
        # Apply mixing to root positions
        root_positions_x = mix(root_positions_x)
        root_positions_y = mix(root_positions_y)
        root_positions_z = mix(root_positions_z)

        # Apply gradual initialization to log lengths
        loglengths = q_param * self.loglengths() + loglengths_0
        lengths = torch.exp(loglengths)
        
        # Apply mixing to directions
        directions_x = self.directions_x() + directions_x_0
        directions_y = self.directions_y() + directions_y_0
        directions_z = self.directions_z() + directions_z_0
        
        directions_x = mix(directions_x)
        directions_y = mix(directions_y)
        directions_z = mix(directions_z)
        
        # Combine directions and root positions to construct the skeleton
        directions = directions_x, directions_y, directions_z
        root_positions = root_positions_x, root_positions_y, root_positions_z        
        x, y, z = construct_skeleton(*root_positions, *directions, lengths, self.structure, self.order, epsilon)
        
        # An error for regularization
        e_reg = 0
        e_reg = e_reg + 1.0 * trajectory_length(x, y, z, do_sqrt=False) / (x.size()[0] * x.size()[1])
        
        return x, y, z, e_reg
        

def wmse_edges(pred, tar, w_tar, structure, epsilon=1e-10):
    """
    Calculate the weighted mean squared error (WMSE) for each edge in the structure.

    Parameters:
    - pred (torch.Tensor): Predicted coordinates of the edges.
    - tar (torch.Tensor): Target coordinates of the edges.
    - w_tar (torch.Tensor): Weights for the target coordinates.
    - structure (dict): Dictionary containing information about the structure of the scene.
    - epsilon (float, optional): Small value added to denominator for numerical stability (default is 1e-10).

    Returns:
    - torch.Tensor: Weighted mean squared error for all edges.

    Notes:
    - Assumes that each edge is defined by a tuple (i_from, i_to, i_length) in the 'edges' field of the structure dictionary.
    """
    edges = structure["edges"]  # Extract edges from the structure

    e = 0  # Initialize total error

    # Iterate over each edge in the structure
    for edge in edges:
        i_from, i_to, i_length = edge  # Extract indices and length for the current edge
        
        # Calculate weight for the current edge
        w = w_tar[:, i_from:i_from + 1] * w_tar[:, i_to:i_to + 1]
        
        # Calculate difference between predicted and target coordinates for the current edge
        edge_tar = tar[:, i_to:i_to + 1] - tar[:, i_from:i_from + 1]
        edge_pred = pred[:, i_to:i_to + 1] - pred[:, i_from:i_from + 1]
        
        # Calculate squared error, weighted by w
        delta = edge_pred - edge_tar
        delta2 = delta * delta
        sum_wdelta2 = sumsum(delta2 * w)
        sum_w = sumsum(w)
        
        # Accumulate error for the current edge
        e = e + sum_wdelta2 / (sum_w + epsilon)

    # Normalize total error by the number of edges
    return e / len(edges)        
        
