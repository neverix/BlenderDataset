# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import argparse
import json
import time
import subprocess
import tempfile
import cv2
import numpy as np
import torch


class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_list="./dataset_list.json",
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 blender_root="blender",
                 shapenet_version="3",
                 engine="EEVEE",
                 quiet_mode=False,
                 headless=False,
                 camera_root=None,
                 data_camera_mode="shapenet_car",
                 save_folder=None,
                 model_name="model.obj"
                 ):
        super().__init__()
        self.engine = engine
        self.headless = headless
        self.quiet_mode = quiet_mode
        self.save_folder = save_folder
        self.model_name = model_name
        self.dataset_list = dataset_list
        self.blender_root = blender_root
        self.shapenet_version = shapenet_version
        self.num_views = 1  # TODO?
        self.camera_root = camera_root
        self.data_camera_mode = data_camera_mode
       
        if shapenet_version == "3":  # no model directory, I made this up
            model_name = None
        
        if self.save_folder is None:
            self.temp = tempfile.TemporaryDirectory()
            self.save_folder = self.temp.name
        
        if self.camera_root is None:
            self.camera_root = os.path.join(self.save_folder, "camera")
        
        if self.headless and self.engine == "EEVEE":
            from pyvirtualdisplay import Display
            Display().start()

        # check if dataset_list exists, throw error if not
        if not os.path.exists(dataset_list):
            raise ValueError("dataset_list does not exist!")
        
        scale_list = []
        path_list = []

        # read and parse json file at dataset_list.json
        with open(dataset_list, "r") as f:
            dataset = json.load(f)

        for entry in dataset:
            scale_list.append(entry["scale"])
            path_list.append(entry["directory"])
        
        # for shapenet v2, we normalize the model location
        if shapenet_version == '2':
            for obj_scale, dataset_folder in zip(scale_list, path_list):
                file_list = sorted(os.listdir(os.path.join(dataset_folder)))
                for file in file_list:
                    # check if file_list+'/models' exists
                    if os.path.exists(os.path.join(dataset_folder, file, 'models')):
                        # move all files in file_list+'/models' to file_list
                        os.system('mv ' + os.path.join(dataset_folder, file, 'models/*') + ' ' + os.path.join(dataset_folder, file))
                        # remove file_list+'/models' if it exists
                        os.system('rm -rf ' + os.path.join(dataset_folder, file, 'models'))
                    material_file = os.path.join(dataset_folder, file, 'model_normalized.mtl')
                    # read material_file as a text file, replace any instance of '../images' with './images'
                    with open(material_file, 'r') as f:
                        material_file_text = f.read()
                    material_file_text = material_file_text.replace('../images', './images')
                    # write the modified text to material_file
                    with open(material_file, 'w') as f:
                        f.write(material_file_text)

        self.suffix = ""
        if self.quiet_mode:
            self.suffix = " >> tmp.out"
        
        self.files = []
        self.scales = []
        for obj_scale, dataset_folder in zip(scale_list, path_list):
            file_list = [
                os.path.join(dataset_folder, name, *((model_name,) * int(model_name is not None)))
                for name in sorted(os.listdir(os.path.join(dataset_folder)))]
            self.files += file_list
            self.scales += [obj_scale for _ in file_list]   
        
        
        self.img_size = resolution 

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path, scale = self.files[idx], self.scales[idx]
        render_cmd = "%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views %s --engine %s%s" % (
            self.blender_root, self.save_folder, path, scale, self.num_views, self.engine, self.suffix
        )
        out = subprocess.check_output(render_cmd, shell=True).decode("utf-8")
        if "Saved: '" not in out:
            return None
        fname = out.rpartition("Saved: '")[-1].partition("'\n")[0]
        
        if self.data_camera_mode == 'shapenet_car' or self.data_camera_mode == 'shapenet_chair' \
                or self.data_camera_mode == 'renderpeople' \
                or self.data_camera_mode == 'shapenet_motorbike' or self.data_camera_mode == 'ts_house' or self.data_camera_mode == 'ts_animal' \
                :
            ori_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            img = ori_img[:, :, :3][..., ::-1]
            mask = ori_img[:, :, 3:4]
            condinfo = np.zeros(2)
            fname_list = fname.split('/')
            img_idx = int(fname_list[-1].split('.')[0])
            obj_idx = fname_list[-2]
            syn_idx = fname_list[-3]

            if self.data_camera_mode == 'shapenet_car' or self.data_camera_mode == 'shapenet_chair' \
                    or self.data_camera_mode == 'renderpeople' or self.data_camera_mode == 'shapenet_motorbike' \
                    or self.data_camera_mode == 'ts_house' or self.data_camera_mode == 'ts_animal':
                if not os.path.exists(os.path.join(self.camera_root, syn_idx, obj_idx, 'rotation.npy')):
                    print('==> not found camera root')
                else:
                    rotation_camera = np.load(os.path.join(self.camera_root, syn_idx, obj_idx, 'rotation.npy'))
                    elevation_camera = np.load(os.path.join(self.camera_root, syn_idx, obj_idx, 'elevation.npy'))
                    condinfo[0] = rotation_camera[img_idx] / 180 * np.pi
                    condinfo[1] = (90 - elevation_camera[img_idx]) / 180.0 * np.pi
        else:
            raise NotImplementedError

        if self.img_size is not None:
            resize_img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        else:
            resize_img = img
        if not mask is None:
            mask = cv2.resize(mask, resize_img.shape[:2], interpolation=cv2.INTER_NEAREST)  ########
        else:
            mask = np.ones(1)
        img = resize_img.transpose(2, 0, 1)
        background = np.zeros_like(img)
        img = img * (mask > 0).astype(np.float) + background * (1 - (mask > 0).astype(np.float))
        return np.ascontiguousarray(img), condinfo, np.ascontiguousarray(mask)


if __name__ == "__main__":  # Test
    ds = BlenderDataset()
    img, condinfo, mask = ds[28]
    img = img.transpose(1, 2, 0).astype(np.uint8)
    cv2.imwrite("img.png", img)

