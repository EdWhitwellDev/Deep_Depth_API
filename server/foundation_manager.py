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
    from .foundation_stereo.core.foundation_stereo import FoundationStereo
    from omegaconf import OmegaConf

except ImportError:
    from foundation_stereo.core.foundation_stereo import FoundationStereo
    from omegaconf import OmegaConf
#from core.utils.utils import InputPadder



class FoundationModel:
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

    model_manager = FoundationModel(config_path, model_path)
    model_manager.summary()