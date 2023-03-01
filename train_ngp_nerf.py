"""
Main program for Training - args are passed from the command line, two phases of training are performed

Phase 1: Train the radiance field
Phase 2: Train for the Occupancy Grid conditioned on Radius

The grids are saved in the all_grids folder
The other .pt files are saved in the local directory

Parts of the program are taken from the Instant NGP nerfacc repository

"""

import argparse
import math
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from radiance_fields.ngp import NGPradianceField, NewESDF
from utils import render_image, set_random_seed

from datasets.nerf_synthetic import SubjectLoader
from get_metrics import max_pool_bloat

from nerfacc import ContractionType, OccupancyGrid

def get_args():
    parser = argparse.ArgumentParser()
    # Get arguments from argparse: train_split, scene, aabb, test_chunk_size, unbounded (set), auto_aabb (set), cone_angle
    parser.add_argument(
        "--data_root_fp",
        type=str,
        default="/cluster/home/alakshmanan/datasets/",
        help="path to the data root",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="trainval",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=[
            # nerf synthetic
            "chair",
            "drums",
            "ficus",
            "hotdog",
            "lego",
            "materials",
            "mic",
            "ship",
            # mipnerf360 unbounded
            "garden",
            "bicycle",
            "bonsai",
            "counter",
            "kitchen",
            "room",
            "stump",
        ],
        help="which scene to use",
    )
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
        help="delimited list input",
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="whether to use unbounded rendering",
    )
    parser.add_argument(
        "--auto_aabb",
        action="store_true",
        help="whether to automatically compute the aabb",
    )
    parser.add_argument("--cone_angle", type=float, default=0.0)

    parser.add_argument(
        "--radius_encoded",
        type=float,
        default=0,
    )

    args = parser.parse_args()
    return args


###################### PHASE I Training the radiance field ######################

def instant_ngp(args, device):
    set_random_seed(42)

    render_n_samples = 1024

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}


    data_root_fp = args.data_root_fp
    target_sample_batch_size = 1 << 18
    grid_resolution = 128

    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split=args.train_split,
        num_rays=target_sample_batch_size // render_n_samples,
        **train_dataset_kwargs,
    )

    # From dataloader get images, transformation matrix, intrinsics
    train_dataset.images = train_dataset.images.to(device)
    train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    train_dataset.K = train_dataset.K.to(device)

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="test",
        num_rays=None,
        **test_dataset_kwargs,
    )
    test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)

    #If auto-aabb is set
    if args.auto_aabb:
        camera_locs = torch.cat(
            [train_dataset.camtoworlds, test_dataset.camtoworlds]
        )[:, :3, -1]
        args.aabb = torch.cat(
            [camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]
        ).tolist()
        print("Using auto aabb", args.aabb)

    # setup the scene bounding box.
    if args.unbounded:
        print("Using unbounded rendering")
        contraction_type = ContractionType.UN_BOUNDED_SPHERE
        # contraction_type = ContractionType.UN_BOUNDED_TANH
        scene_aabb = None
        near_plane = 0.2
        far_plane = 1e4
        render_step_size = 1e-2
        alpha_thre = 1e-2
    else:
        contraction_type = ContractionType.AABB
        scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        render_step_size = (
            (scene_aabb[3:] - scene_aabb[:3]).max()
            * math.sqrt(3)
            / render_n_samples
        ).item()
        alpha_thre = 0.0

    # setup the radiance field we want to train.
    max_steps = 20000
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    # calls the radiance field class
    radiance_field = NGPradianceField(
        aabb=args.aabb,
        unbounded=args.unbounded,
    ).to(device)
    optimizer = torch.optim.Adam(
        radiance_field.parameters(), lr=1e-2, eps=1e-15
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
        gamma=0.33,
    )

    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)

    # training
    step = 0
    tic = time.time()
    for epoch in range(10000000):
        for i in range(len(train_dataset)):
            radiance_field.train()
            data = train_dataset[i]

            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            def occ_eval_fn(x):
                if args.cone_angle > 0.0:
                    # randomly sample a camera for computing step size.
                    camera_ids = torch.randint(
                        0, len(train_dataset), (x.shape[0],), device=device
                    )
                    origins = train_dataset.camtoworlds[camera_ids, :3, -1]
                    t = (origins - x).norm(dim=-1, keepdim=True)
                    # compute actual step size used in marching, based on the distance to the camera.
                    step_size = torch.clamp(
                        t * args.cone_angle, min=render_step_size
                    )
                    # filter out the points that are not in the near far plane.
                    if (near_plane is not None) and (far_plane is not None):
                        step_size = torch.where(
                            (t > near_plane) & (t < far_plane),
                            step_size,
                            torch.zeros_like(step_size),
                        )
                else:
                    step_size = render_step_size
                # compute occupancy
                density = radiance_field.query_density(x)
                return density * step_size

            # update occupancy grid
            occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)

            # render
            rgb, acc, depth, n_rendering_samples = render_image(
                radiance_field,
                occupancy_grid,
                rays,
                scene_aabb,
                # rendering options
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=args.cone_angle,
                alpha_thre=alpha_thre,
            )
            if n_rendering_samples == 0:
                continue

            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)
            alive_ray_mask = acc.squeeze(-1) > 0

            # compute loss
            loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
            optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            optimizer.step()
            scheduler.step()

            if step % 10000 == 0:
                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                print(
                    f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                    f"loss={loss:.5f} | "
                    f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} |", flush = True
                )

            
            #save the model
            if step == max_steps:
                # evaluation
                radiance_field.eval()
                torch.save(radiance_field.state_dict(), f"temp_radiance_field.pt")
                occupancy_values_main = torch.zeros((grid_resolution,grid_resolution,grid_resolution), dtype=torch.float32, device = device)
                
                to_divide_by = grid_resolution/(2*1.5) # grid_resolution*3/8 
                occupancy_values = occupancy_grid.query_occ(torch.tensor([[length/to_divide_by, width/to_divide_by, height/to_divide_by] for height in range(-int(grid_resolution/2),int(grid_resolution/2)) for width in range(-int(grid_resolution/2),int(grid_resolution/2)) for length in range(-int(grid_resolution/2),int(grid_resolution/2))], dtype=torch.float32, device = device))
                for z in range(grid_resolution):
                    for y in range(grid_resolution):
                        for x in range(grid_resolution):
                                occupancy_values_main[x][y][z] = occupancy_values[x + y*grid_resolution + z*grid_resolution*grid_resolution]
                torch.save(occupancy_values_main, f"temp_occupancy_values.pt")


                psnrs = []
                with torch.no_grad():
                    for i in tqdm.tqdm(range(len(test_dataset))):
                        data = test_dataset[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]

                        # rendering_tic = time.time()
                        # rendering
                        rgb, acc, depth, _ = render_image(
                            radiance_field,
                            occupancy_grid,
                            rays,
                            scene_aabb,
                            # rendering options
                            near_plane=near_plane,
                            far_plane=far_plane,
                            render_step_size=render_step_size,
                            render_bkgd=render_bkgd,
                            cone_angle=args.cone_angle,
                            alpha_thre=alpha_thre,
                            # test options
                            test_chunk_size=args.test_chunk_size,
                        )
                        # final_rendering_time = time.time() - rendering_tic
                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())

                        # Check to see if the model is working
                        # if i == 0:
                            # imageio.imwrite(
                            #     "acc_binary_test.png",
                            #     ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
                            # )
                            # imageio.imwrite(
                            #     "rgb_test.png",
                            #     (rgb.cpu().numpy() * 255).astype(np.uint8),
                            # )
                            # break

                psnr_avg = sum(psnrs) / len(psnrs)
                print(f"evaluation: psnr_avg={psnr_avg}", flush = True)
                train_dataset.training = True

            if step == max_steps:
                print("training stops", flush = True)
                exit()

            step += 1


###################### PHASE II Training for the occupancy conditioned on a radius ######################

def occupancy_on_radius(args, device):

    set_random_seed(42)

    grid_resolution = 128

    # setup the radiance field we want to train.
    max_steps = 2000
    grad_scaler = torch.cuda.amp.GradScaler(2**10)

    # load the new model
    radius_encoded = args.radius_encoded
    
    #new model is the model trained by the new code, inputs = [x,y,z] points, output = occupancy values
    new_model = NewESDF(aabb=args.aabb,
         unbounded=args.unbounded, new_tensor = torch.torch.full((5,1),radius_encoded, dtype=torch.float32).to(device)
     ).to(device)
    old_model = torch.load(f"radiance_field.pt")

    # Get the old occupancy, apply max pooling to get the bloated occupancy - ground truth
    old_occupancy = torch.load(f"occupancy_values.pt")
    old_occupancy = max_pool_bloat(old_occupancy, radius_encoded, device)
    old_occupancy = (old_occupancy > 0.1).float()

    # Freeze the old model
    with torch.no_grad():
        new_model.mlp_base_1.params.copy_(old_model['mlp_base_1.params'])
        new_model.mlp_base_2.params.copy_(old_model['mlp_base_2.params'])
        new_model.mlp_base[0].params.copy_(old_model['mlp_base.0.params'])
        new_model.mlp_base[1].params.copy_(old_model['mlp_base.1.params'])

        new_model.mlp_base_1.requires_grad = False
        new_model.mlp_base_2.requires_grad = False
        new_model.mlp_base[0].requires_grad = False
        new_model.mlp_base[1].requires_grad = False



    optimizer = torch.optim.Adam(
        new_model.parameters(), lr=1e-2, eps=1e-15
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
        gamma=0.33,
    )

    # training
    step = 0
    loss_sum = 0
    tic = time.time()



    for epoch in range(200):
                    
        new_model.train()
        
        #randomly sample 10000 points from the grid (grid dimensions are -1.5 to 1.5), 
        # torch.rand gives random numbers between 0 and 1, so we multiply it by 3 and subtract 1.5 to get numbers between -1.5 and 1.5
        points = 3* torch.rand(10000,3, device = device) - 1.5

        network_outputs = new_model.query_density(points)

        # To get the old occupancy points which are in range 0 to 128, we add 1.5 to points and divide by 3 to get values between 0 and 1
        points_for_old_occupancy = (((points+1.5) / 3) * grid_resolution).int()

        index_tensor = torch.tensor(points_for_old_occupancy, device = device, dtype = torch.long)
        old_occupancy_list = old_occupancy[tuple(index_tensor.T)]


        loss = F.binary_cross_entropy_with_logits(network_outputs.squeeze(), torch.Tensor(old_occupancy_list).to(device))

        loss_sum += loss
        optimizer.zero_grad()

        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            elapsed_time = time.time() - tic
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss_sum:.5f} | ", flush=True)
            loss_sum = 0

        #save the model when 200 epochs are reached. For lego, the convergence is 200 epochs, for other datasets, have to fix a number
        if epoch == 199:
            # evaluation
            new_model.eval()
            
            #get the occupancy values given dimensions (128x128x128) max values) dimensions: (3,3,3)m from -1.5 to 1.5
            to_divide_by = grid_resolution/(2*1.5) #grid_resolution*3/8 

            final_occupancy_grid = new_model.query_density(torch.tensor([[length/to_divide_by, width/to_divide_by, height/to_divide_by] for height in range(-int(grid_resolution/2),int(grid_resolution/2)) for width in range(-int(grid_resolution/2),int(grid_resolution/2)) for length in range(-int(grid_resolution/2),int(grid_resolution/2))], dtype=torch.float32, device = device))

            
            final_grid = np.zeros((grid_resolution,grid_resolution,grid_resolution))
            for z in range(grid_resolution):
                for y in range(grid_resolution):
                    for x in range(grid_resolution):
                            final_grid[x][y][z] = final_occupancy_grid[x + y*grid_resolution + z*grid_resolution*grid_resolution]
            
            
            final_grid /= final_grid.max()
            final_grid = (final_grid > final_grid.max()/2).astype(np.uint8)

            torch.save(final_grid, f"all_grids/final_grid_r{radius_encoded}.pt")
            
            # Visualization of a slice to check if the grid is properly generated
            # imageio.imwrite("predicted_radius_slice.png",(final_grid[:,:,60]*255).astype(np.uint8))

            print("training stops", flush = True)
            exit()

        step += 1



if __name__ == '__main__':
    device = "cuda:0"
    args = get_args()
    # instant_ngp(args, device)
    occupancy_on_radius(args, device)

    print("done", flush = True)