import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .model_manager import ModelManager
from .vo_manager import VOManager
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
import sys
import base64

class NumpyStereoImages(BaseModel):
    left_image: str
    right_image: str

    def to_numpy(self):

        # Decode the base64 string
        print("length of left image base64 string:", len(self.left_image))
        print("length of right image base64 string:", len(self.right_image))
        decoded_data = base64.b64decode(self.left_image)
        # Convert bytes to numpy array
        np_array_left = np.frombuffer(decoded_data, dtype=np.uint8)
        # Decode the image from the numpy array
        left = cv2.imdecode(np_array_left, cv2.IMREAD_COLOR)

        # Decode the base64 string
        decoded_data = base64.b64decode(self.right_image)
        # Convert bytes to numpy array
        np_array_right = np.frombuffer(decoded_data, dtype=np.uint8)
        # Decode the image from the numpy array
        right = cv2.imdecode(np_array_right, cv2.IMREAD_COLOR)
        print("Decoded images shapes:", left.shape, right.shape)
        return left, right
    
class TrajectoryImages(BaseModel):
    images: list[str]

    def to_numpy(self):
        images = []
        for img_b64 in self.images:
            decoded_data = base64.b64decode(img_b64)
            np_array = np.frombuffer(decoded_data, dtype=np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        return images

class StereoTrajectoryImages(BaseModel):
    left_images: TrajectoryImages
    right_images: TrajectoryImages

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
model_manager = ModelManager(config_path, model_path, calib=calib)

optometry_manager = VOManager(calib, stereo=model_manager)

@app.get("/")
async def root():
    print(model_manager.summary())
    return {"message": "Hello World"}

@app.post("/predict_foundation_stereo/")
async def predict_foundation_stereo(numpy_images: NumpyStereoImages):
    #print(numpy_images)
    left_image_np, right_image_np = numpy_images.to_numpy()
    #left_image_np, right_image_np = model_manager.rectify_images(left_image_np, right_image_np)
    if left_image_np is None or right_image_np is None:
        return {"error": "Invalid image data"}
    print("Received images with shapes:", left_image_np.shape, right_image_np.shape)
    depth_map = model_manager.predict_depth(left_image_np, right_image_np)
    print("Depth map shape:", depth_map.shape)

    # Convert depth map to bytes
    buffer = io.BytesIO()
    np.save(buffer, depth_map)
    buffer.seek(0)  # Reset buffer position to the beginning

    return StreamingResponse(buffer, media_type="application/octet-stream", headers={
        "Content-Disposition": "attachment; filename=array.npy"
    })

@app.post("/compute_trajectory/")
async def compute_trajectory(trajectory_images: StereoTrajectoryImages):
    left_images = trajectory_images.left_images.to_numpy()
    right_images = trajectory_images.right_images.to_numpy()
    
    #left_images, right_images = model_manager.rectify_images(left_images, right_images)

    if not left_images or not right_images:
        return {"error": "No valid images provided"}

    print(f"Processing {len(left_images)} left images and {len(right_images)} right images for trajectory computation.")
    trajectory = optometry_manager.compute_trajectory(left_images, right_images)

    # Convert trajectory to bytes
    buffer = io.BytesIO()
    np.save(buffer, trajectory)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/octet-stream", headers={
        "Content-Disposition": "attachment; filename=trajectory.npy"
    })

@app.post("/create_3d_model/")
async def create_3d_model(numpy_images: NumpyStereoImages):
    left_image_np, right_image_np = numpy_images.to_numpy()
    if left_image_np is None or right_image_np is None:
        print("bad")
        return {"error": "Invalid image data"}
    
    print("Received images with shapes:", left_image_np.shape, right_image_np.shape)
    model_manager.create_3d_model(left_image_np, right_image_np)
    
    return {"message": "3D model created successfully"}