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
from .foundation_stereo.Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from .foundation_stereo.core.foundation_stereo import FoundationStereo
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../foundation_stereo')
from omegaconf import OmegaConf
#from core.utils.utils import InputPadder



class ModelManager:
    DIM_FACTOR = 32
    calib = np.load('stereo_params_2.npz')
    K1, dist1 = calib['K1'], calib['dist1']
    K2, dist2 = calib['K2'], calib['dist2']
    R1, R2 = calib['R1'], calib['R2']
    P1, P2 = calib['P1'], calib['P2']
    Q = calib['Q']
    ROI1, ROI2 = calib['ROI1'], calib['ROI2']

    BASELINE = 0.109

    x = max(ROI1[0], ROI2[0])
    y = max(ROI1[1], ROI2[1])
    w = min(ROI1[0] + ROI1[2], ROI2[0] + ROI2[2]) - x
    h = min(ROI1[1] + ROI1[3], ROI2[1] + ROI2[3]) - y

    w, h = 1200, 720

    map1L, map2L = cv.initUndistortRectifyMap(K1, dist1, R1, P1, (w, h), cv.CV_16SC2)
    map1R, map2R = cv.initUndistortRectifyMap(K2, dist2, R2, P2, (w, h), cv.CV_16SC2)


    dim_divis_factor = 32

    scale = float(0.5)

    P1[:2] *= scale

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
    def predict(self, left_image, right_image):
        # Crop images to be divisible by DIM_FACTOR
        left_image = self.scale_image(left_image)
        right_image = self.scale_image(right_image)
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

    def scale_image(self, image):
        image = cv.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv.INTER_LINEAR)
        return image

    def crop_to_dim_factor(self, image):
        h, w = image.shape[:2]
        new_h = h - (h % self.DIM_FACTOR)
        new_w = w - (w % self.DIM_FACTOR)
        return image[:new_h, :new_w]
    
    def rectify_images(self, left_frame, right_frame):
        rectifiedL = cv.remap(left_frame, self.map1L, self.map2L, cv.INTER_LINEAR)
        rectifiedR = cv.remap(right_frame, self.map1R, self.map2R, cv.INTER_LINEAR)
        # Crop the images to the valid ROI
        rectifiedL = rectifiedL[self.y:self.y+self.h, self.x:self.x+self.w]
        rectifiedR = rectifiedR[self.y:self.y+self.h, self.x:self.x+self.w]

        return rectifiedL, rectifiedR
    

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
    
    def create_point_cloud(self, depth, left_frame, de_noise=True, max_depth=10):
        xyz = depth2xyzmap(depth, self.P1)
        pcd = toOpen3dCloud(xyz.reshape(-1, 3), colors=left_frame.reshape(-1, 3))

        keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<= max_depth)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)

        if de_noise:
          cl, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
          inlier_cloud = pcd.select_by_index(ind)
          pcd = inlier_cloud

        return pcd

    def create_3d_model(self, left_image, right_image):
        print("Creating model with images of dims: ", left_image.shape, " and ", right_image.shape)
        left_frame_ori = left_image.copy()
        left_image_ori = self.scale_image(left_frame_ori)
        left_image_ori = self.crop_to_dim_factor(left_image_ori)
        depth = self.predict_depth(left_image, right_image)
        pcd = self.create_point_cloud(depth, left_image_ori, de_noise=False, max_depth=10)
        mesh, densities = self.create_mesh(pcd)
        #pc_tree = o3d.geometry.KDTreeFlann(pcd)
        print("Has colors:", pcd.has_colors())
        print("Number of colors:", len(pcd.colors))
        print("Sample colors:", np.asarray(pcd.colors)[:5])
        #colors = []
        #for vertex in mesh.vertices:
        #    vertex_np = np.asarray(vertex)
        #    k, idx, _ = pc_tree.search_knn_vector_3d(vertex_np, 1)
        #    if k == 1:
        #        colors.append(pcd.colors[idx[0]])
        #    else:
        #        colors.append([0, 0, 0])  # fallback black
#
        #mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        print("Has vertex colors:", mesh.has_vertex_colors())
        print("Number of vertex colors:", len(mesh.vertex_colors))
        print("Sample vertex color:", mesh.vertex_colors[0] if len(mesh.vertex_colors) > 0 else "None")
        #mesh = self.reverse_normals(mesh)
        o3d.visualization.draw_geometries([mesh])
        o3d.io.write_triangle_mesh("output_mesh.ply", mesh, write_vertex_colors=True)

    def predict_depth(self, left_image, right_image):
        P1 = self.P1.copy()
        disp = self.predict(left_image, right_image)
        depth_map = (P1[0, 0] * self.BASELINE) / (disp + 1e-6)
        depth_map = self.clean(depth_map, 0, 20)
        depth_average = np.mean(depth_map)
        depth_std = np.std(depth_map)

        depth_map = self.clean(depth_map, depth_average - 2 * depth_std, depth_average + 2 * depth_std)
        return depth_map

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