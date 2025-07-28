import io
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from matplotlib import pyplot as plt
from pydantic import BaseModel
from .foundation_manager import FoundationModel
from .mono_manager import MonoModel
from .reconstruction_manager import Reconstructor
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
import sys
import base64

class ImageEncoded(BaseModel):
    image: str

    def to_numpy(self):
        # Decode the base64 string
        decoded_data = base64.b64decode(self.image)
        # Convert bytes to numpy array
        np_array = np.frombuffer(decoded_data, dtype=np.uint8)
        # Decode the image from the numpy array
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Decoded image is None")
        return img

class DepthImages(BaseModel):
    left_image: ImageEncoded
    right_image: Optional[ImageEncoded] = None
    base_line: Optional[float] = None
    focal_length: Optional[float] = None

    def to_numpy(self):
        left = self.left_image.to_numpy()
        if self.right_image is not None:
            right = self.right_image.to_numpy()
        else:
            right = None
        return left, right
    
class Reconstruction(BaseModel):
    left_image: ImageEncoded
    projection_matrix: list

    right_image: Optional[ImageEncoded] = None
    depth_map: Optional[ImageEncoded] = None
    base_line: Optional[float] = None
    focal_length: Optional[float] = None
    
    def to_numpy(self):
        left = self.left_image.to_numpy()
        projection_matrix = np.array(self.projection_matrix)
        if self.right_image is not None:
            right = self.right_image.to_numpy()
        else:
            right = None
        if self.depth_map is not None:
            depth = self.depth_map.to_numpy()
        else:
            depth = None
        return left, right, depth, projection_matrix

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

calib = np.load('stereo_params_2.npz')

foundation_pretrained_model_path = "foundation_stereo/pretrained_models/11-33-40"
config_path = os.path.join(foundation_pretrained_model_path, "cfg.yaml")
model_path = os.path.join(foundation_pretrained_model_path, "model_best_bp2.pth")
foundation_model = FoundationModel(config_path, model_path, calib=calib)
mono_model = MonoModel()

@app.get("/")
async def root():
    print(foundation_model.summary())
    return {"message": "Hello World"}

@app.post("/predict_depth/")
async def predict_depth(numpy_images: DepthImages):
    left_image_np, right_image_np = numpy_images.to_numpy()
    if left_image_np is None:
        return {"error": "Invalid image data"}
    
    if numpy_images.base_line is not None and numpy_images.focal_length is not None:
        if right_image_np is not None:
            depth_map = foundation_model.predict_depth(left_image_np, right_image_np, numpy_images.base_line, numpy_images.focal_length)
        else:
            depth_map = mono_model.predict_depth(left_image_np)

    else:
        if right_image_np is not None:
            depth_map = foundation_model.predict(left_image_np, right_image_np)
        else:
            depth_map = mono_model.predict_depth(left_image_np)

    buffer = io.BytesIO()
    np.save(buffer, depth_map)
    buffer.seek(0)  

    return StreamingResponse(buffer, media_type="application/octet-stream", headers={
        "Content-Disposition": "attachment; filename=array.npy"
    })


@app.post("/create_3d_model/")
async def create_3d_model(recon_data: Reconstruction):
    left_image_np, right_image_np, depth_map_np, projection_matrix = recon_data.to_numpy()
    original_image = left_image_np.copy()
    orthographic = False
    if left_image_np is None:   
        return {"error": "Invalid image data"}
    if right_image_np is not None:
        depth_map_np = foundation_model.predict_depth(left_image_np, right_image_np, recon_data.base_line, recon_data.focal_length)
    if depth_map_np is None:
        orthographic = True
        depth_map_np = mono_model.predict_depth(left_image_np)
        depth_map_np = ((depth_map_np - np.min(depth_map_np)) / (np.max(depth_map_np) - np.min(depth_map_np))) * 10
        projection_matrix = MonoModel.generate_projection_matrix(left_image_np)

    Reconstructor.create_3d_model(original_image, depth_map=depth_map_np, projection_matrix=projection_matrix, orthographic=orthographic)

    return {"message": "3D model created successfully"}