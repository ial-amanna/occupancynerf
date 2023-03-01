"""
Code to get and visualize the ESDF values from the final grids, also Ground Truth if available

All the predicted grids along with GT are stored in the all_grids folder
Predicted format: all_grids/final_grid_r{}.{}.pt
GT: all_grids/ground_truth.npy

"""

import torch
import numpy as np
import math
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from get_metrics import max_pool_bloat

    
# Code to generate heatmap (Predicted ESDF) from different final grids
def get_heatmap():

    main_file = torch.zeros((128,128,128))
    file_list = [torch.load(f'all_grids/final_grid_r{i:.1f}.pt') for i in np.arange(0, 1.6, 0.1)]

    for i in range(128):
        for j in range(128):
            for k in range(128):
                # Set the voxel value based on the file values
                prev_val = file_list[0][i][j][k]
                for idx, file in enumerate(file_list[1:], start=1):
                    if prev_val != file[i][j][k]:
                        main_file[i][j][k] = idx/10
                        break
                prev_val = file[i][j][k]

    # convert main_file to numpy array
    torch.save(main_file, 'all_grids/esdf_predicted.pt')




# Code to generate ground truth heatmap (GT ESDF)
def get_gt_heatmap(file_name = 'all_grids/ground_truth.npy', device = "cpu"):
    files = []
    file_main = np.load(file_name)
    file_0 = torch.from_numpy(file_main).to(device)
    files.append(file_0)
    files.append(max_pool_bloat(file_0, 0.1, device=device))
    files.append(max_pool_bloat(file_0, 0.2, device=device))
    files.append(max_pool_bloat(file_0, 0.3, device=device))
    files.append(max_pool_bloat(file_0, 0.4, device=device))
    files.append(max_pool_bloat(file_0, 0.5, device=device))
    files.append(max_pool_bloat(file_0, 0.6, device=device))
    files.append(max_pool_bloat(file_0, 0.7, device=device))
    files.append(max_pool_bloat(file_0, 0.8, device=device))
    files.append(max_pool_bloat(file_0, 0.9, device=device))
    files.append(max_pool_bloat(file_0, 1.0, device=device))
    files.append(max_pool_bloat(file_0, 1.1, device=device))
    files.append(max_pool_bloat(file_0, 1.2, device=device))
    files.append(max_pool_bloat(file_0, 1.3, device=device))
    files.append(max_pool_bloat(file_0, 1.4, device=device))
    files.append(max_pool_bloat(file_0, 1.5, device=device))

    main_file = torch.zeros((128,128,128))

    for i in range(128):
        for j in range(128):
            for k in range(128):
                # Set the voxel value based on the file values
                prev_val = files[0][i][j][k]
                for idx, file in enumerate(files[1:], start=1):
                    if prev_val != file[i][j][k]:
                        main_file[i][j][k] = idx/10
                        break
                prev_val = file[i][j][k]

    # convert main_file to numpy array
    torch.save(main_file, 'all_grids/esdf_gt.pt')




if __name__ == '__main__':
    get_heatmap()
    get_gt_heatmap(file_name = 'all_grids/ground_truth.npy', device = 'cuda:0')
    
    # check by plotting a slice of the heatmap
    # main_file = torch.load('all_grids/esdf_predicted.pt')
    main_file = torch.load('all_grids/esdf_gt.pt')
    main_file = main_file.numpy()

    color_list = [(0, '#FF0000'), (0.5, '#FFFF00'), (1, '#00FF00')]
    cmap = colors.LinearSegmentedColormap.from_list('custom', color_list)
    norm = colors.Normalize(vmin=0, vmax=1.5)

    plt.pcolormesh(main_file[:,:,64], cmap=cmap, norm=norm)
    plt.colorbar(location = 'bottom')
    plt.savefig('temp.png')
