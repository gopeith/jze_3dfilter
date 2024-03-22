import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jze_3dfilter import *
from jze_3dfilter_data import *
from jze_3dfilter_face import *
from jze_3dfilter_skeleton import *


def demo_save_result(fname, x, y, z=None):
    """
    Save the x, y, z coordinates to a file.
    """
    f = open(fname, "w")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if z is None:
                f.write("%e %e " % (x[i, j], y[i, j]))
            else:
                f.write("%e %e %e " % (x[i, j], y[i, j], z[i, j]))
        f.write("\n")
    f.close()


def demo_load_data(fname_json, w100_face=False, w100_skeleton=False):
    """
    Load data from a JSON file and preprocess it.

    Parameters:
        fname_json (str): File name of the JSON file containing the data.

    Returns:
        tuple: A tuple containing arrays representing face data, skeleton data and structure information.

    """

    # Load data from JSON file
    data = load_json(fname_json)

    # Make skeleton structure
    structure_skeleton, nodes = make_structure(edges_raw)

    # Extract face landmarks
    xyzw_face = data["face_landmarks"]

    # Extract face coordinates
    x_rec_face_np = xyzw_face[:, :, 0]
    y_rec_face_np = xyzw_face[:, :, 1]
    z_rec_face_np = -xyzw_face[:, :, 2] # Flip z coordinates
    w_rec_face_np = xyzw_face[:, :, 3]
    
    # Extract skeleton coordinates
    xyzw_skeleton = [None for i in range(len(nodes))]
    for key in nodes:
        data_key, i = key
        xyzw_skeleton[nodes[key]] = data[data_key][:, i:(i + 1), :]
    xyzw_skeleton = np.concatenate(xyzw_skeleton, axis=1)
    x_rec_skeleton_np = xyzw_skeleton[:, :, 0]
    y_rec_skeleton_np = xyzw_skeleton[:, :, 1]
    z_rec_skeleton_np = xyzw_skeleton[:, :, 2]
    w_rec_skeleton_np = xyzw_skeleton[:, :, 3]
    
    if w100_face:
        w_rec_face_np = w_rec_face_np / 100.0
    if w100_skeleton:
        w_rec_skeleton_np = w_rec_skeleton_np / 100.0
    
    # Check weights of face and skeleton data
    check_weights(w_rec_face_np)
    check_weights(w_rec_skeleton_np)

    return x_rec_face_np, y_rec_face_np, z_rec_face_np, w_rec_face_np, x_rec_skeleton_np, y_rec_skeleton_np, z_rec_skeleton_np, w_rec_skeleton_np, structure_skeleton


def demo():

    fname_json = "0AQHLPt8NDk-00_00_03.669-00_00_06.272.json"
    #fname_json = "debug/input_data/_2FBDaOPYig_5-3-rgb_front.json"
    #fname_json = "debug/input_data/_5CV2fIG7qY_5-5-rgb_front.json"
    #fname_json = "debug/input_data/0AQHLPt8NDk-00_00_03.669-00_00_06.272.json"
    #fname_json = "debug/input_data/0pKzG0RRUz4_2-2-rgb_front.json"
    #fname_json = "debug/input_data/0pKzG0RRUz4_3-1-rgb_front.json"
    #fname_json = "debug/input_data/1aJwX9nRlmk_7-2-rgb_front.json"
    
    
    # Load data from JSON file
    x_rec_face_np, y_rec_face_np, z_rec_face_np, w_rec_face_np, x_rec_skeleton_np, y_rec_skeleton_np, z_rec_skeleton_np, w_rec_skeleton_np, structure_skeleton = demo_load_data(fname_json, w100_skeleton=True)
    
    # Normalize position and scale
    # Compute the center and scale of the data for normalization
    center, scale = compute_center_and_scale(
        [
            [[x_rec_skeleton_np, w_rec_skeleton_np]],
            [[y_rec_skeleton_np, w_rec_skeleton_np]],
        ]
    )
    # Normalize the x and y coordinates of both face and skeleton data
    x_rec_skeleton_raw_np = x_rec_skeleton_np
    y_rec_skeleton_raw_np = y_rec_skeleton_np
    x_rec_skeleton_np = (x_rec_skeleton_np - center[0]) / scale
    y_rec_skeleton_np = (y_rec_skeleton_np - center[1]) / scale    

    T_skeleton = x_rec_skeleton_np.shape[0]
    
    # You can also choose "cuda" to utilize GPU computation if available.
    device = "cpu"  # Device selection: "cpu" for CPU computation or "cuda" for GPU computation (if available)

    # Convert numpy arrays into torch tensors and move them to the selected device (CPU or GPU) 
    x_rec_skeleton = torch.tensor(x_rec_skeleton_np).to(device)
    y_rec_skeleton = torch.tensor(y_rec_skeleton_np).to(device)
    z_rec_skeleton = torch.tensor(z_rec_skeleton_np).to(device)
    w_rec_skeleton = torch.tensor(w_rec_skeleton_np).to(device)
    
    # Data needed for initialization. Objects compute initialization of their parameters.    
    data_for_init_skeleton = {
        "x_rec": x_rec_skeleton,
        "y_rec": y_rec_skeleton,
        "z_rec": z_rec_skeleton,
        "w_rec": w_rec_skeleton,
        "use_z_rec": False, # this z-coordinates are useless
    }

    # Model of a scene has:
    #   - model of face - returns two x, y, z coordinates:
    #        - coordinates for flexible face
    #        - coordinates for rigid face. This face serves as sort of regulariyation
    #   - model of sceleton - returns one x, y, z tuples
    
    # loss functions
    loss_skeleton_nodes = lambda x_cam, y_cam: wmse(x_cam, x_rec_skeleton, w_rec_skeleton) + wmse(y_cam, y_rec_skeleton, w_rec_skeleton)
    #loss_skeleton_edges = lambda x_cam, y_cam: wmse_edges(x_cam, x_rec_skeleton, w_rec_skeleton, structure_skeleton) + wmse_edges(y_cam, y_rec_skeleton, w_rec_skeleton, structure_skeleton)
    #loss_skeleton = lambda x_cam, y_cam: loss_skeleton_nodes(x_cam, y_cam) + loss_skeleton_edges(x_cam, y_cam)
    loss_skeleton = lambda x_cam, y_cam: loss_skeleton_nodes(x_cam, y_cam)
    losses = [
        loss_skeleton,  # Loss function for the skeleton
    ]
    
    # Creating models
    model_object_skeleton = ModelSkeleton_2g(
        T=T_skeleton,
        structure=structure_skeleton,
        data_for_init=data_for_init_skeleton,
        device=device,
    )
    model_camera = ModelCameraPlain(device=device)
    model_scene = ModelSceneMultipleObjects(
        models_object=[model_object_skeleton],
        model_camera=model_camera,
        device=device,
    )
    
    # Computing results
    # This function returns two lists:
    #   - List of x, y, z tuples for objects 
    #   - List of x, y tuples for objects visible through camera 
    
    results_obj, results_cam = resolve_scene(
        model_scene,
        losses=losses,
        lr=1e-3,
        q_reg_obj=1.0,
        q_reg_cam=1.0,
        n_epochs=100,
        device=device,
        print_loss=True,
    )
    
    # Result for skeleton, idenx = 2
    x_result_skeleton = results_obj[0][0]
    y_result_skeleton = results_obj[0][1]
    z_result_skeleton = results_obj[0][2]

    # denormalizing coordinates
    x_result_skeleton = scale * x_result_skeleton + center[0]  
    y_result_skeleton = scale * y_result_skeleton + center[1]  
    z_result_skeleton = scale * z_result_skeleton

    # Save results

    demo_save_result("result-skeleton.txt", x_result_skeleton, y_result_skeleton, z_result_skeleton)  
    demo_save_result("tar-skeleton.txt", x_rec_skeleton_raw_np, y_rec_skeleton_raw_np)  
    
    return "OK"
        
        
if __name__ == "__main__":

    print(demo())


