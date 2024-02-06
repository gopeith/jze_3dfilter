import os
import pathlib
import math
import json

import numpy as np


edges_raw = [
	(("pose_landmarks", 0), ("pose_landmarks", 11), 0),
	(("pose_landmarks", 0), ("pose_landmarks", 12), 0),
	(("pose_landmarks", 11), ("pose_landmarks", 13), 1),
	(("pose_landmarks", 12), ("pose_landmarks", 14), 1),
	(("pose_landmarks", 13), ("pose_landmarks", 15), 2),
	(("pose_landmarks", 14), ("pose_landmarks", 16), 2),
    (("pose_landmarks", 15), ("left_hand_landmarks", 0), 3),
    (("left_hand_landmarks", 0), ("left_hand_landmarks", 1), 4),
    (("left_hand_landmarks", 1), ("left_hand_landmarks", 2), 5),
    (("left_hand_landmarks", 2), ("left_hand_landmarks", 3), 6),
    (("left_hand_landmarks", 3), ("left_hand_landmarks", 4), 7),
    (("left_hand_landmarks", 0), ("left_hand_landmarks", 5), 8),
    (("left_hand_landmarks", 5), ("left_hand_landmarks", 6), 9),
    (("left_hand_landmarks", 6), ("left_hand_landmarks", 7), 10),
    (("left_hand_landmarks", 7), ("left_hand_landmarks", 8), 11),
    (("left_hand_landmarks", 0), ("left_hand_landmarks", 9), 12),
    (("left_hand_landmarks", 9), ("left_hand_landmarks", 10), 13),
    (("left_hand_landmarks", 10), ("left_hand_landmarks", 11), 14),
    (("left_hand_landmarks", 11), ("left_hand_landmarks", 12), 15),
    (("left_hand_landmarks", 0), ("left_hand_landmarks", 13), 16),
    (("left_hand_landmarks", 13), ("left_hand_landmarks", 14), 17),
    (("left_hand_landmarks", 14), ("left_hand_landmarks", 15), 18),
    (("left_hand_landmarks", 15), ("left_hand_landmarks", 16), 19),
    (("left_hand_landmarks", 0), ("left_hand_landmarks", 17), 20),
    (("left_hand_landmarks", 17), ("left_hand_landmarks", 18), 21),
    (("left_hand_landmarks", 18), ("left_hand_landmarks", 19), 22),
    (("left_hand_landmarks", 19), ("left_hand_landmarks", 20), 23),
    (("pose_landmarks", 16), ("right_hand_landmarks", 0), 3),
    (("right_hand_landmarks", 0), ("right_hand_landmarks", 1), 4),
    (("right_hand_landmarks", 1), ("right_hand_landmarks", 2), 5),
    (("right_hand_landmarks", 2), ("right_hand_landmarks", 3), 6),
    (("right_hand_landmarks", 3), ("right_hand_landmarks", 4), 7),
    (("right_hand_landmarks", 0), ("right_hand_landmarks", 5), 8),
    (("right_hand_landmarks", 5), ("right_hand_landmarks", 6), 9),
    (("right_hand_landmarks", 6), ("right_hand_landmarks", 7), 10),
    (("right_hand_landmarks", 7), ("right_hand_landmarks", 8), 11),
    (("right_hand_landmarks", 0), ("right_hand_landmarks", 9), 12),
    (("right_hand_landmarks", 9), ("right_hand_landmarks", 10), 13),
    (("right_hand_landmarks", 10), ("right_hand_landmarks", 11), 14),
    (("right_hand_landmarks", 11), ("right_hand_landmarks", 12), 15),
    (("right_hand_landmarks", 0), ("right_hand_landmarks", 13), 16),
    (("right_hand_landmarks", 13), ("right_hand_landmarks", 14), 17),
    (("right_hand_landmarks", 14), ("right_hand_landmarks", 15), 18),
    (("right_hand_landmarks", 15), ("right_hand_landmarks", 16), 19),
    (("right_hand_landmarks", 0), ("right_hand_landmarks", 17), 20),
    (("right_hand_landmarks", 17), ("right_hand_landmarks", 18), 21),
    (("right_hand_landmarks", 18), ("right_hand_landmarks", 19), 22),
    (("right_hand_landmarks", 19), ("right_hand_landmarks", 20), 23),
]

edges_world_raw = [
	(("pose_world_landmarks", 0), ("pose_world_landmarks", 11), 0),
	(("pose_world_landmarks", 0), ("pose_world_landmarks", 12), 0),
	(("pose_world_landmarks", 11), ("pose_world_landmarks", 13), 1),
	(("pose_world_landmarks", 12), ("pose_world_landmarks", 14), 1),
	(("pose_world_landmarks", 13), ("pose_world_landmarks", 15), 2),
	(("pose_world_landmarks", 14), ("pose_world_landmarks", 16), 2),
    (("pose_world_landmarks", 15), ("left_hand_landmarks", 0), 3),
    (("left_hand_landmarks", 0), ("left_hand_landmarks", 1), 4),
    (("left_hand_landmarks", 1), ("left_hand_landmarks", 2), 5),
    (("left_hand_landmarks", 2), ("left_hand_landmarks", 3), 6),
    (("left_hand_landmarks", 3), ("left_hand_landmarks", 4), 7),
    (("left_hand_landmarks", 0), ("left_hand_landmarks", 5), 8),
    (("left_hand_landmarks", 5), ("left_hand_landmarks", 6), 9),
    (("left_hand_landmarks", 6), ("left_hand_landmarks", 7), 10),
    (("left_hand_landmarks", 7), ("left_hand_landmarks", 8), 11),
    (("left_hand_landmarks", 0), ("left_hand_landmarks", 9), 12),
    (("left_hand_landmarks", 9), ("left_hand_landmarks", 10), 13),
    (("left_hand_landmarks", 10), ("left_hand_landmarks", 11), 14),
    (("left_hand_landmarks", 11), ("left_hand_landmarks", 12), 15),
    (("left_hand_landmarks", 0), ("left_hand_landmarks", 13), 16),
    (("left_hand_landmarks", 13), ("left_hand_landmarks", 14), 17),
    (("left_hand_landmarks", 14), ("left_hand_landmarks", 15), 18),
    (("left_hand_landmarks", 15), ("left_hand_landmarks", 16), 19),
    (("left_hand_landmarks", 0), ("left_hand_landmarks", 17), 20),
    (("left_hand_landmarks", 17), ("left_hand_landmarks", 18), 21),
    (("left_hand_landmarks", 18), ("left_hand_landmarks", 19), 22),
    (("left_hand_landmarks", 19), ("left_hand_landmarks", 20), 23),
    (("pose_world_landmarks", 16), ("right_hand_landmarks", 0), 3),
    (("right_hand_landmarks", 0), ("right_hand_landmarks", 1), 4),
    (("right_hand_landmarks", 1), ("right_hand_landmarks", 2), 5),
    (("right_hand_landmarks", 2), ("right_hand_landmarks", 3), 6),
    (("right_hand_landmarks", 3), ("right_hand_landmarks", 4), 7),
    (("right_hand_landmarks", 0), ("right_hand_landmarks", 5), 8),
    (("right_hand_landmarks", 5), ("right_hand_landmarks", 6), 9),
    (("right_hand_landmarks", 6), ("right_hand_landmarks", 7), 10),
    (("right_hand_landmarks", 7), ("right_hand_landmarks", 8), 11),
    (("right_hand_landmarks", 0), ("right_hand_landmarks", 9), 12),
    (("right_hand_landmarks", 9), ("right_hand_landmarks", 10), 13),
    (("right_hand_landmarks", 10), ("right_hand_landmarks", 11), 14),
    (("right_hand_landmarks", 11), ("right_hand_landmarks", 12), 15),
    (("right_hand_landmarks", 0), ("right_hand_landmarks", 13), 16),
    (("right_hand_landmarks", 13), ("right_hand_landmarks", 14), 17),
    (("right_hand_landmarks", 14), ("right_hand_landmarks", 15), 18),
    (("right_hand_landmarks", 15), ("right_hand_landmarks", 16), 19),
    (("right_hand_landmarks", 0), ("right_hand_landmarks", 17), 20),
    (("right_hand_landmarks", 17), ("right_hand_landmarks", 18), 21),
    (("right_hand_landmarks", 18), ("right_hand_landmarks", 19), 22),
    (("right_hand_landmarks", 19), ("right_hand_landmarks", 20), 23),
]


def make_structure(edges):
    nodes = {}
    lengths = {}
    edges2 = []
    is_root = {}
    for node_from, node_to, i_length in edges:
        if not node_from in is_root:
            is_root[node_from] = True
        is_root[node_to] = False
        lengths[i_length] = "gotha"
        if not node_from in nodes:
            nodes[node_from] = len(nodes)
        if not node_to in nodes:
            nodes[node_to] = len(nodes)
        edges2.append((nodes[node_from], nodes[node_to], i_length))      

    root = None
    for node in is_root:
        if is_root[node]:
            root = nodes[node]
            break
        
    structure = {
        "#nodes": len(nodes),
        "#lengths": len(lengths),
        "root": root,
        "edges": edges2,
    }
    return structure, nodes
    


def set_w(x, val):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if len(x[i][j]) > 0:
                x[i][j][3] = val
    return x


def missing_values(x, epsilon=1e-10):
    sum_wx = []
    sum_wy = []
    sum_wz = []
    sum_w = []
    for x_i in x:
        for j in range(len(x_i)):
            x_ij = x_i[j]
            if len(x_ij) == 0:
                continue
            while len(sum_w) <= j:
                sum_wx.append(0)
                sum_wy.append(0)
                sum_wz.append(0)
                sum_w.append(0)
            w_ij = x_i[j][3] + epsilon
            sum_wx[j] += w_ij * x_i[j][0]
            sum_wy[j] += w_ij * x_i[j][1]
            sum_wz[j] += w_ij * x_i[j][2]
            sum_w[j] += w_ij
    y = []
    for x_i in x:
        y_i = []
        if len(x_i) == 0:
            for j in range(len(sum_wx)):
                x_ij = [sum_wx[j] / (sum_w[j]), sum_wy[j] / (sum_w[j]), sum_wz[j] / (sum_w[j]), 0.0]
                y_i.append(x_ij)
        else:
            for j in range(len(x_i)):
                x_ij = x_i[j]
                if len(x_ij) == 0:
                    x_ij = [sum_wx[j] / (sum_w[j] + epsilon), sum_wy[j] / (sum_w[j] + epsilon), sum_wz[j] / (sum_w[j] + epsilon), 0.0]
                y_i.append(x_ij)
        y.append(y_i)
    return y            
            
            
def all_weights_are_zero(x):
    for x_i in x:
        for x_ij in x_i:
            if x_ij[3] > 0:
                return False
    return True


def load_json(fname_json):
    data = json.load(open(fname_json))
    joints = data["joints"]
    results = {}
    for i_frame in joints:
        for key in joints[i_frame]:
            data_key = joints[i_frame][key]
            i = int(i_frame)
            if not key in results:
                results[key] = []
            while len(results[key]) <= i:
                results[key].append(None)
            results[key][i] = data_key

    for key in results:
        if all_weights_are_zero(results[key]):
            results[key] = set_w(results[key], 1.0)
        results[key] = missing_values(results[key])
        results[key] = np.asarray(results[key], dtype="float32")    

    return results    


def check_weights(w):
    """
    Check if the weights array meets certain criteria.

    Parameters:
        w (numpy.ndarray): The weights array to be checked.

    Raises:
        Exception: If the weights are negative or greater than 1.
    """
    # Check if any weight is negative
    if np.min(np.min(w)) < 0:
        raise Exception("Weights cannot be negative.")
    
    # Check if any weight is greater than 1
    if np.max(np.max(w)) > 1:
        raise Exception("Weights must be <= 1.")

