# script to test the requests to the FastAPI server
import glob
import io
import os
from turtle import left
import requests
import numpy as np
import cv2 as cv
import base64
import matplotlib.pyplot as plt


calib = np.load('server/stereo_params_2.npz')
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

P1 *= 0.5


def load_image(image_path):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    print(f"Loaded image from {image_path} with shape: {image.shape}")
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    return image

def scale_image(image, scale=0.5):
    image = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
    return image

def load_images(left, right, rect=True):
    left_image = load_image(left)
    right_image = load_image(right)
    left_image, right_image = right_image, left_image
    left_image, right_image  = rectify_images(left_image, right_image)    
    left_image = scale_image(left_image)
    right_image = scale_image(right_image)
    print("Loaded images shapes:", left_image.shape, right_image.shape)
    return left_image, right_image

def encode_image(image):
    success, encoded_img = cv.imencode(".jpg", image)
    if not success:
        raise ValueError("Image encoding failed.")
    return base64.b64encode(encoded_img.tobytes()).decode('utf-8')

def plot_result(disp, ref_image_left=None, ref_image_right=None):
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].imshow(disp, cmap='plasma')
    ax[0].set_title("Disparity Map")
    ax[0].axis('off')
    if ref_image_left is not None:
        ax[1].imshow(ref_image_left)
        ax[1].set_title("Reference Image")
        ax[1].axis('off')

    if ref_image_right is not None:
        ax[2].imshow(ref_image_right)
        ax[2].set_title("Reference Image Right")
        ax[2].axis('off')
    plt.show()

def rectify_images(left_frame, right_frame):
    rectifiedL = cv.remap(left_frame, map1L, map2L, cv.INTER_LINEAR)
    rectifiedR = cv.remap(right_frame, map1R, map2R, cv.INTER_LINEAR)
    # Crop the images to the valid ROI
    rectifiedL = rectifiedL[y:y+h, x:x+w]
    rectifiedR = rectifiedR[y:y+h, x:x+w]
    return rectifiedL, rectifiedR
        

def send_depth_request(left_image, right_image):
    left_image_b64 = encode_image(left_image)
    right_image_b64 = encode_image(right_image)

    url = "http://localhost:8000/predict_depth/"
    response = requests.post(url, json={"left_image": {"image": left_image_b64}, "right_image": {"image": right_image_b64}, "base_line": BASELINE, "focal_length": P1[0, 0]})
    print("Response status code:", response.status_code)
    return response

def send_mono_depth_request(left_image):
    left_image_b64 = encode_image(left_image)

    url = "http://localhost:8000/predict_depth/"
    response = requests.post(url, json={"left_image": {"image": left_image_b64}, "base_line": BASELINE, "focal_length": P1[0, 0]})
    print("Response status code:", response.status_code)
    return response

def send_3d_reconstruction_request(left_image, right_image):
    left_image_b64 = encode_image(left_image)
    right_image_b64 = encode_image(right_image)

    url = "http://localhost:8000/create_3d_model/"
    print("Projection matrix shape:", P1)
    response = requests.post(url, json={"left_image": {"image": left_image_b64}, "right_image": {"image": right_image_b64},  "projection_matrix": P1.tolist(), "base_line": BASELINE, "focal_length": P1[0, 0]})
    print("Response status code:", response.status_code)
    return response

def test_depth(swap = False):
    left_image_path = "client/DepthImages/left2.jpg"  # Replace with image path
    right_image_path = "client/DepthImages/right2.jpg"  # Replace with image path

    try:
        left_image, right_image = load_images(left_image_path, right_image_path)
        #right_image, left_image = rectify_images(left_image, right_image)

        #result = send_depth_request(left_image, right_image)
#
        #if result.status_code == 200:
        #    disp = np.load(io.BytesIO(result.content))
        #    print("Received depth map with shape:", disp.shape)
        #    plot_result(disp, ref_image_left=left_image, ref_image_right=right_image)
        #else:
        #    print("Request failed with status code:", result.status_code)

        result = send_mono_depth_request(left_image)
        if result.status_code == 200:
            disp = np.load(io.BytesIO(result.content))
            print("Received mono depth map with shape:", disp.shape)
            plot_result(disp, ref_image_left=left_image)
        else:
            print("Mono request failed with status code:", result.status_code)
    except Exception as e:
        print("An error occurred:", e)

def test_3d_model():
    left_image_path = "client/DepthImages/left5.jpg"  # Replace with image path
    right_image_path = "client/DepthImages/right5.jpg"  # Replace with image path

    try:
        left_image, right_image = load_images(left_image_path, right_image_path)
        #right_image, left_image = rectify_images(left_image, right_image)
        result = send_3d_reconstruction_request(left_image, right_image)

        if result.status_code == 200:
            print("3D model created successfully.")
        else:
            print("Request failed with status code:", result.status_code)
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
   #test_depth(swap=True)  # Set to True to test with wrong order
   test_3d_model()
   #test_trajectory(swap=True)  # Set to True to test with wrong order

