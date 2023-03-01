"""
Code to get the error metrics (MSE, IoU) between the predicted grids and the ground truth grid.

All the predicted grids along with GT are stored in the all_grids folder in the same directory.
Predicted format: all_grids/final_grid_r{}.{}.pt
GT: all_grids/ground_truth.npy

"""

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

def max_pool_bloat(converted_occ, radius_encoded, device = "cpu"):
    # function to convert the occupancy values based on the radius of the object

    if radius_encoded != 0:
        occupancy_radius = math.ceil(radius_encoded * 42.667)
        converted_occ = converted_occ.unsqueeze(0)
        kernel_size = (occupancy_radius*2+1, occupancy_radius*2+1, occupancy_radius*2+1)
        padding = occupancy_radius
        converted_occ = F.max_pool3d(converted_occ, kernel_size=kernel_size, stride=1, padding=padding)
        converted_occ = converted_occ.squeeze(0)

    if device == "cpu":
        return converted_occ.cpu()
    else:
        return converted_occ

def get_iou(A, B):
    # A = A.float()
    # B = B.float()
    intersection = torch.sum(A*B)
    union = torch.sum(A) + torch.sum(B) - intersection
    iou = intersection / union
    return iou


if __name__ == "__main__":
    # Get IOU for all grids
    device = "cpu"


    file_0 = torch.load('all_grids/final_grid_r0.0.pt')
    file_0 = torch.from_numpy(file_0)
    file_1 = torch.load('all_grids/final_grid_r0.1.pt')
    file_1 = torch.from_numpy(file_1)
    file_2 = torch.load('all_grids/final_grid_r0.2.pt')
    file_2 = torch.from_numpy(file_2)
    file_3 = torch.load('all_grids/final_grid_r0.3.pt')
    file_3 = torch.from_numpy(file_3)
    file_4 = torch.load('all_grids/final_grid_r0.4.pt')
    file_4 = torch.from_numpy(file_4)
    file_5 = torch.load('all_grids/final_grid_r0.5.pt')
    file_5 = torch.from_numpy(file_5)
    file_6 = torch.load('all_grids/final_grid_r0.6.pt')
    file_6 = torch.from_numpy(file_6)
    file_7 = torch.load('all_grids/final_grid_r0.7.pt')
    file_7 = torch.from_numpy(file_7)
    file_8 = torch.load('all_grids/final_grid_r0.8.pt')
    file_8 = torch.from_numpy(file_8)
    file_9 = torch.load('all_grids/final_grid_r0.9.pt')
    file_9 = torch.from_numpy(file_9)
    file_10 = torch.load('all_grids/final_grid_r1.0.pt')
    file_10 = torch.from_numpy(file_10)

    file_main = np.load('all_grids/ground_truth.npy')
    file_gt_0 = torch.from_numpy(file_main)
    file_gt_1 = max_pool_bloat(file_gt_0, 0.1, device = device)
    file_gt_2 = max_pool_bloat(file_gt_0, 0.2, device = device)
    file_gt_3 = max_pool_bloat(file_gt_0, 0.3, device = device)
    file_gt_4 = max_pool_bloat(file_gt_0, 0.4, device = device)
    file_gt_5 = max_pool_bloat(file_gt_0, 0.5, device = device)
    file_gt_6 = max_pool_bloat(file_gt_0, 0.6, device = device)
    file_gt_7 = max_pool_bloat(file_gt_0, 0.7, device = device)
    file_gt_8 = max_pool_bloat(file_gt_0, 0.8, device = device)
    file_gt_9 = max_pool_bloat(file_gt_0, 0.9, device = device)
    file_gt_10 = max_pool_bloat(file_gt_0, 1.0, device = device)


    iou_0 = get_iou(file_0, file_gt_0)
    iou_1 = get_iou(file_1, file_gt_1)
    iou_2 = get_iou(file_2, file_gt_2)
    iou_3 = get_iou(file_3, file_gt_3)
    iou_4 = get_iou(file_4, file_gt_4)
    iou_5 = get_iou(file_5, file_gt_5)
    iou_6 = get_iou(file_6, file_gt_6)
    iou_7 = get_iou(file_7, file_gt_7)
    iou_8 = get_iou(file_8, file_gt_8)
    iou_9 = get_iou(file_9, file_gt_9)
    iou_10 = get_iou(file_10, file_gt_10)

    print(f"The IOU is 0:{iou_0}, 1:{iou_1}, 2:{iou_2}, 3:{iou_3}, 4:{iou_4}, 5:{iou_5}", flush = True)
    print(f"Continuing, 6:{iou_6}, 7:{iou_7}, 8:{iou_8}, 9:{iou_9}, 10:{iou_10}", flush = True)




    #### MSE

    file_gt = torch.load('all_grids/mse_gt.pt')
    file_pred = torch.load('all_grids/mse_predicted.pt')

    print("Hello", flush = True)

    mse = nn.MSELoss()

    loss = mse(file_gt, file_pred)
    print(f"The MSE loss is {loss}", flush = True) 