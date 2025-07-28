import time

from shapely import points

import os,sys
import argparse
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available*")
import logging
#import imageio
import cv2 as cv
import open3d as o3d
from matplotlib import pyplot as plt
import torch
import numpy as np
try:
    from .foundation_stereo.Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
    from .foundation_stereo.core.foundation_stereo import FoundationStereo
    from omegaconf import OmegaConf

except ImportError:
    from foundation_stereo.Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
    from foundation_stereo.core.foundation_stereo import FoundationStereo
    from omegaconf import OmegaConf
#from core.utils.utils import InputPadder



class ModelManager:
    DIM_FACTOR = 32

    def __init__(self, config_path, model_path, device='cuda', calib=None):
        self.device = device
        self.calib = calib
        self.config = OmegaConf.create(OmegaConf.load(config_path))
        self.foundation_stereo_model = FoundationStereo(self.config)

        self.foundation_stereo_model.load_state_dict(torch.load(model_path, weights_only=False)['model'])
        self.foundation_stereo_model.to(self.device)
        self.foundation_stereo_model.eval()
        #set_logging_format()

    def clean(self, data, min, max):
        data[data < min] = 0
        data[data > max] = 0  # Remove outliers
        return data

    def crop_to_dim_factor(self, image):
        h, w = image.shape[:2]
        new_h = h - (h % self.DIM_FACTOR)
        new_w = w - (w % self.DIM_FACTOR)
        return image[:new_h, :new_w]
    
    def predict(self, left_image, right_image):
        left_image = self.crop_to_dim_factor(left_image)
        right_image = self.crop_to_dim_factor(right_image)
        print(left_image.shape)
        H, W = left_image.shape[:2]
        with torch.no_grad():
            left_image = cv.cvtColor(left_image, cv.COLOR_BGR2RGB)
            right_image = cv.cvtColor(right_image, cv.COLOR_BGR2RGB)
            left_tensor = torch.as_tensor(left_image).cuda().float()[None].permute(0, 3, 1, 2)
            right_tensor = torch.as_tensor(right_image).cuda().float()[None].permute(0, 3, 1, 2)

            disparity_map = self.foundation_stereo_model.forward(left_tensor, right_tensor, iters=32, test_mode=True)
            disp = disparity_map.data.cpu().numpy().reshape(H, W)
            return disp    
        
    def predict_depth(self, left_image, right_image, baseline, focal_length, remove_outliers=True):
        disp = self.predict(left_image, right_image)
        depth_map =( focal_length * baseline )/ (disp + 1e-6)
        
        depth_map = self.clean(depth_map, 0, 20)
        depth_average = np.mean(depth_map)
        depth_std = np.std(depth_map)

        depth_map = self.clean(depth_map, depth_average - 2 * depth_std, depth_average + 2 * depth_std)
        return depth_map
    
    def reverse_normals(self, mesh):
        normals = np.asarray(mesh.vertex_normals)
        mesh.vertex_normals = o3d.utility.Vector3dVector(-normals)

        # Optionally reverse triangle winding too
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])
        return mesh

    def create_mesh(self, pcd):
    # Create a mesh from the point cloud
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=30)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        densities = np.asarray(densities)
        density_threshold = np.percentile(densities, 5)
        vertices_to_keep = densities > density_threshold
        mesh.remove_vertices_by_mask(~vertices_to_keep)
        mesh.compute_vertex_normals()
        colors = np.asarray(mesh.vertex_colors)
        if colors.max() > 1.0:
            colors = colors / 255.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        return mesh, densities

    def create_point_cloud(self, depth, left_frame, projection_matrix, de_noise=True, max_depth=10):
        xyz = depth2xyzmap(depth, projection_matrix)
        pcd = toOpen3dCloud(xyz.reshape(-1, 3), colors=left_frame.reshape(-1, 3))

        keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<= max_depth)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)

        if de_noise:
          cl, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
          inlier_cloud = pcd.select_by_index(ind)
          pcd = inlier_cloud

        return pcd

    def create_3d_model(self, left_image, right_image=None, depth_map=None, projection_matrix=None, baseline=None, focal_length=None):
        print("Creating model with images of dims: ", left_image.shape, " and ", right_image.shape)
        if right_image is None and depth_map is None:
            raise ValueError("Either right image or depth map must be provided")
        depth = depth_map
        projection_matrix = np.array(projection_matrix) if projection_matrix is not None else self.P1
        print("Projection matrix shape:", projection_matrix)
        print("Default projection matrix shape:", self.P1)
        left_image_ori = self.crop_to_dim_factor(left_image.copy())

        if right_image is not None:
            depth = self.predict_depth(left_image, right_image, baseline=baseline, focal_length=focal_length)

        pcd = self.create_point_cloud(depth, left_image_ori, projection_matrix=projection_matrix, de_noise=True, max_depth=10)
        mesh, densities = self.create_mesh(pcd)

        o3d.visualization.draw_geometries([mesh])

    def summary(self):
        print(self.foundation_stereo_model)

if __name__ == "__main__":
    foundation_pretrained_model_path = "server/foundation_stereo/pretrained_models/11-33-40"
    config_path = os.path.join(foundation_pretrained_model_path, "cfg.yaml")
    model_path = os.path.join(foundation_pretrained_model_path, "model_best_bp2.pth")

    #check if model files exist
    if not os.path.exists(config_path):
        print(f"Model config files not found at {foundation_pretrained_model_path}. Please check the path.")
        sys.exit(1)

    model_manager = ModelManager(config_path, model_path)
    #model_manager.create_3d_model(
    #    left_image=cv.imread("client/DepthImages/left1.jpg"),
    #    right_image=cv.imread("client/DepthImages/right1.jpg")
    #)

    model_manager.summary()