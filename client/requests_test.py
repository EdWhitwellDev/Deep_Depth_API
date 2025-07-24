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


def load_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    return image

def load_images(left, right, rect=True):
    left_image = load_image(left)
    right_image = load_image(right)

    if rect:
        left_image, right_image  = rectify_images(left_image, right_image)
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

def plot_trajectory(trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Trajectory")
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

    url = "http://localhost:8000/predict_foundation_stereo/"
    response = requests.post(url, json={"left_image": left_image_b64, "right_image": right_image_b64})
    print("Response status code:", response.status_code)
    return response

def send_trajectory_request(images_left, images_right):
    encoded_images_left = [encode_image(img) for img in images_left]
    encoded_images_right = [encode_image(img) for img in images_right]
    url = "http://localhost:8000/compute_trajectory/"
    response = requests.post(url, json={"left_images": {"images": encoded_images_left}, "right_images": {"images": encoded_images_right}})
    print("Response status code:", response.status_code)
    return response

def send_3d_reconstruction_request(left_image, right_image):
    left_image_b64 = encode_image(left_image)
    right_image_b64 = encode_image(right_image)

    url = "http://localhost:8000/create_3d_model/"
    response = requests.post(url, json={"left_image": left_image_b64, "right_image": right_image_b64})
    print("Response status code:", response.status_code)
    return response

def test_depth(swap = False):
    left_image_path = "client/DepthImages/left1.jpg"  # Replace with image path
    right_image_path = "client/DepthImages/right1.jpg"  # Replace with image path

    try:
        left_image, right_image = load_images(left_image_path, right_image_path, False)
        #right_image, left_image = rectify_images(left_image, right_image)
        if swap:
            left_image, right_image = right_image, left_image  # Swap images to simulate wrong order
        result = send_depth_request(left_image, right_image)

        if result.status_code == 200:
            disp = np.load(io.BytesIO(result.content))
            print("Received depth map with shape:", disp.shape)
            plot_result(disp, ref_image_left=left_image, ref_image_right=right_image)
        else:
            print("Request failed with status code:", result.status_code)
    except Exception as e:
        print("An error occurred:", e)

def test_3d_model():
    left_image_path = "client/DepthImages/left3.jpg"  # Replace with image path
    right_image_path = "client/DepthImages/right3.jpg"  # Replace with image path

    try:
        left_image, right_image = load_images(left_image_path, right_image_path, False)
        #right_image, left_image = rectify_images(left_image, right_image)
        result = send_3d_reconstruction_request(left_image, right_image)

        if result.status_code == 200:
            print("3D model created successfully.")
        else:
            print("Request failed with status code:", result.status_code)
    except Exception as e:
        print("An error occurred:", e)


def test_trajectory(swap=False):
    image_dir_left = "client/DATA/StereoLeft/StereoLeft/15"  # Set your path
    image_dir_right = "client/DATA/StereoRight/StereoRight/15"  # Set your path
    
    if swap:
        image_dir_left, image_dir_right = image_dir_right, image_dir_left
        
    image_paths_left = sorted(glob.glob(os.path.join(image_dir_left, "*.jpg")))
    image_paths_right = sorted(glob.glob(os.path.join(image_dir_right, "*.jpg")))
    images_left = [load_image(path) for path in image_paths_left if os.path.isfile(path)]
    images_right = [load_image(path) for path in image_paths_right if os.path.isfile(path)]
    print("Found images:", len(images_left), "left images and", len(images_right), "right images.")

    if not images_left or not images_right:
        print("No valid images found for trajectory computation.")
        return

    response = send_trajectory_request(images_left, images_right)
    if response.status_code == 200:
        trajectory = np.load(io.BytesIO(response.content))
        plot_trajectory(trajectory)
        print("Received trajectory with length:", trajectory.shape)
    else:
        print("Trajectory request failed with status code:", response.status_code)

if __name__ == "__main__":
   #test_depth(swap=True)  # Set to True to test with wrong order
   test_3d_model()
   #test_trajectory(swap=True)  # Set to True to test with wrong order
#
   #trajectory = np.load("client/traj.npy")
   ##print(trajectory.shape)
   #plot_trajectory(trajectory=trajectory)
