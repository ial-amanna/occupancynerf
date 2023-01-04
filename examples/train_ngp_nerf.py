"""
Main program for Training
"""

import argparse
import math
import os
import time

import imageio
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import tqdm
from radiance_fields.ngp import NGPradianceField, NewESDF
from utils import render_image, set_random_seed

from nerfacc import ContractionType, OccupancyGrid

def get_args():
    parser = argparse.ArgumentParser()
    # Get arguments from argparse: train_split, scene, aabb, test_chunk_size, unbounded (set), auto_aabb (set), cone_angle

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
    args = parser.parse_args()
    return args


def main_model(args, device):
    set_random_seed(42)

    render_n_samples = 1024

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    if args.unbounded:
        from datasets.nerf_360_v2 import SubjectLoader

        data_root_fp = "/cluster/home/alakshmanan/datasets/"
        target_sample_batch_size = 1 << 20
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        grid_resolution = 256
    else:
        from datasets.nerf_synthetic import SubjectLoader

        data_root_fp = "/cluster/home/alakshmanan/datasets/"
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
                # torch.save(radiance_field.state_dict(), "radiance_field.pt")
                
                occupancy_values_0 = occupancy_grid.query_occ(torch.tensor([[m, n, 0] for m in range(1000) for n in range(1000)], dtype=torch.float32, device = device))
                occupancy_values_50 = occupancy_grid.query_occ(torch.tensor([[m, n, 50.0] for m in range(1000) for n in range(1000)], dtype=torch.float32, device = device))
                occupancy_values_100 = occupancy_grid.query_occ(torch.tensor([[m, n, 100.0] for m in range(1000) for n in range(1000)], dtype=torch.float32, device = device))
                occupancy_values_300 = occupancy_grid.query_occ(torch.tensor([[m, n, 300.0] for m in range(1000) for n in range(1000)], dtype=torch.float32, device = device))
                occupancy_values_500 = occupancy_grid.query_occ(torch.tensor([[m, n, 500.0] for m in range(1000) for n in range(1000)], dtype=torch.float32, device = device))


                print(max(occupancy_values_50)*255)
                print(max(occupancy_values_100)*255)

                print(max(occupancy_values_300)*255)
                print(max(occupancy_values_500)*255)
                map_output_0 = np.ones((1000,1000))
                map_output_50 = np.ones((1000,1000))
                map_output_100 = np.ones((1000,1000))

                map_output_300 = np.ones((1000,1000))
                map_output_500 = np.ones((1000,1000))

                count = 0
                for i in range(1000):
                    for j in range(1000):
                        map_output_0[i][j] = occupancy_values_0[count]
                        map_output_50[i][j] = occupancy_values_50[count]
                        map_output_100[i][j] = occupancy_values_100[count]
                        map_output_300[i][j] = occupancy_values_300[count]
                        map_output_500[i][j] = occupancy_values_500[count]
                        count+=1


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

                        if i == 0:
                            imageio.imwrite(
                                "acc_binary_test2.png",
                                ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
                            )
                            imageio.imwrite(
                                "rgb_test2.png",
                                (rgb.cpu().numpy() * 255).astype(np.uint8),
                            )
                            break
                        # if i == 0:
                        #     imageio.imwrite(
                        #         "occupancy_map_0.png",
                        #         ((map_output_0)* 255).astype(np.uint8)
                        #     )
                        #     imageio.imwrite(
                        #         "occupancy_map_50.png",
                        #         ((map_output_50)* 255).astype(np.uint8)
                        #     )
                        #     imageio.imwrite(
                        #         "occupancy_map_100.png",
                        #         ((map_output_100)* 255).astype(np.uint8)
                        #     )
                        #     imageio.imwrite(
                        #         "occupancy_map_300.png",
                        #         ((map_output_300)* 255).astype(np.uint8)
                        #     )
                        #     imageio.imwrite(
                        #         "occupancy_map_500.png",
                        #         ((map_output_500)* 255).astype(np.uint8)
                        #     )
                        #     break

                psnr_avg = sum(psnrs) / len(psnrs)
                print(f"evaluation: psnr_avg={psnr_avg}", flush = True)
                train_dataset.training = True

            if step == max_steps:
                print("training stops", flush = True)
                exit()

            step += 1

def new_render(args, device):

    set_random_seed(42)

    render_n_samples = 1024

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    if args.unbounded:
        from datasets.nerf_360_v2 import SubjectLoader

        data_root_fp = "/cluster/home/alakshmanan/datasets/"
        target_sample_batch_size = 1 << 20
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        grid_resolution = 256
    else:
        from datasets.nerf_synthetic import SubjectLoader

        data_root_fp = "/cluster/home/alakshmanan/datasets/"
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


    # load the new model
    radius_encoded = 2.0

    #new model is the model trained by the new code, try to sample points on a plane and query the distance,
    new_model = NewESDF(aabb=args.aabb,
         unbounded=args.unbounded, new_tensor = torch.torch.full((5,1),radius_encoded, dtype=torch.float32).to(device)
     ).to(device)
    old_model = torch.load("radiance_field.pt")
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
            new_model.train()
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
                density = new_model.query_density(x)
                return density * step_size

            # update occupancy grid
            occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)
            # render
            rgb, acc, depth, n_rendering_samples = render_image(
                new_model,
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
            loss = F.binary_cross_entropy_with_logits(rgb, pixels)

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
                    # f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} |", flush = True
                )

            #save the model
            if step == max_steps:
                # evaluation
                new_model.eval()
                # torch.save(new_model.state_dict(), "new_model.pt")
                
                #get the occupancy values given dimensions (1000x1000)
                
                occupancy_values_0 = occupancy_grid.query_occ(torch.tensor([[m, n, 0] for m in range(128) for n in range(128)], dtype=torch.float32, device = device))
                occupancy_values_20 = occupancy_grid.query_occ(torch.tensor([[m, n, 20.0] for m in range(128) for n in range(128)], dtype=torch.float32, device = device))
                occupancy_values_40 = occupancy_grid.query_occ(torch.tensor([[m, n, 40.0] for m in range(128) for n in range(128)], dtype=torch.float32, device = device))
                occupancy_values_60 = occupancy_grid.query_occ(torch.tensor([[m, n, 60.0] for m in range(128) for n in range(128)], dtype=torch.float32, device = device))
                occupancy_values_80 = occupancy_grid.query_occ(torch.tensor([[m, n, 80.0] for m in range(128) for n in range(128)], dtype=torch.float32, device = device))
                
                map_output_0 = np.ones((128,128))
                map_output_20 = np.ones((128,128))
                map_output_40 = np.ones((128,128))

                map_output_60 = np.ones((128,128))
                map_output_80 = np.ones((128,128))

                count = 0
                for i in range(128):
                    for j in range(128):
                        map_output_0[i][j] = occupancy_values_0[count]
                        map_output_20[i][j] = occupancy_values_20[count]
                        map_output_40[i][j] = occupancy_values_40[count]
                        map_output_60[i][j] = occupancy_values_60[count]
                        map_output_80[i][j] = occupancy_values_80[count]
                        count+=1

                print(max(map_output_0))
                print(max(map_output_20))
                
                print(max(map_output_40))
                print(max(map_output_60))
                print(max(map_output_80))
                
                map_output_0 /= max(map_output_0)
                map_output_20 /= max(map_output_20)
                map_output_40 /= max(map_output_40)
                map_output_60 /= max(map_output_60)
                map_output_80 /= max(map_output_80)


                with torch.no_grad():
                    for i in tqdm.tqdm(range(len(test_dataset))):
                        data = test_dataset[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]

                        # rendering
                        rgb, acc, depth, _ = render_image(
                            new_model,
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


                        if i == 0:
                            imageio.imwrite(
                                "occupancy_map_0.png",
                                ((map_output_0)* 255).astype(np.uint8)
                            )
                            imageio.imwrite(
                                "occupancy_map_20.png",
                                ((map_output_20)* 255).astype(np.uint8)
                            )
                            imageio.imwrite(
                                "occupancy_map_40.png",
                                ((map_output_40)* 255).astype(np.uint8)
                            )
                            imageio.imwrite(
                                "occupancy_map_60.png",
                                ((map_output_60)* 255).astype(np.uint8)
                            )
                            imageio.imwrite(
                                "occupancy_map_80.png",
                                ((map_output_80)* 255).astype(np.uint8)
                            )
                            break

                # psnr_avg = sum(psnrs) / len(psnrs)
                # print(f"evaluation: psnr_avg={psnr_avg}", flush = True)
                train_dataset.training = True

            if step == max_steps:
                print("training stops", flush = True)
                exit()

            step += 1




if __name__ == '__main__':
    device = "cuda:0"
    args = get_args()
    # main_model(args, device)
    new_render(args, device)
    print("done", flush = True)