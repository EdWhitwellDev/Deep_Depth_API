import time

from shapely import points

import warnings
warnings.filterwarnings("ignore", message="xFormers is not available*")
import logging
#import imageio
import cv2 as cv
import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np
try:
    from .foundation_stereo.Utils import depth2xyzmap, toOpen3dCloud
except ImportError:
    from foundation_stereo.Utils import depth2xyzmap, toOpen3dCloud
#from core.utils.utils import InputPadder

class Reconstructor:
    DIM_FACTOR = 32

    def crop_to_dim_factor(image):
        h, w = image.shape[:2]
        new_h = h - (h % Reconstructor.DIM_FACTOR)
        new_w = w - (w % Reconstructor.DIM_FACTOR)
        return image[:new_h, :new_w]
    def reverse_normals(self, mesh):
        normals = np.asarray(mesh.vertex_normals)
        mesh.vertex_normals = o3d.utility.Vector3dVector(-normals)

        # Optionally reverse triangle winding too
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])
        return mesh

    def create_mesh(pcd):
    # Create a mesh from the point cloud
        pcd.estimate_normals()
        #pcd.orient_normals_consistent_tangent_plane(k=30)

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

    def create_point_cloud(depth, left_frame, projection_matrix, orthographic=False, de_noise=True, max_depth=10):
        xyz = depth2xyzmap(depth, projection_matrix, orthographic=orthographic)
        pcd = toOpen3dCloud(xyz.reshape(-1, 3), colors=left_frame.reshape(-1, 3))

        keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<= max_depth)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)

        if de_noise:
          cl, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
          inlier_cloud = pcd.select_by_index(ind)
          pcd = inlier_cloud

        return pcd

    def create_3d_model(left_image,  depth_map=None, projection_matrix=None, orthographic=False):
        if depth_map is None:
            raise ValueError("Depth map must be provided")

        print(left_image.shape, depth_map.shape)
        if not depth_map.shape == left_image.shape[:2]:
            print("Depth map shape does not match left image shape")
            left_image = Reconstructor.crop_to_dim_factor(left_image)
        left_image_ori = left_image.copy()

        pcd = Reconstructor.create_point_cloud(depth_map, left_image_ori, projection_matrix=projection_matrix, orthographic=orthographic
                                               , de_noise=True, max_depth=10)
        pcd.estimate_normals()
        print("Point cloud created with", len(pcd.points), "points")
        #o3d.visualization.draw_geometries([pcd])
#
        #mesh, densities = Reconstructor.create_mesh(pcd)
#
        #o3d.visualization.draw_geometries([mesh])

