import time
import cv2
import numpy as np
import os
import glob

# --- Camera intrinsics (modify based on your setup)
calib = np.load('server\stereo_params_2.npz')
K, dist2 = calib['K2'], calib['dist2']

# --- Load image sequence
image_dir = "client/DATA/OpIm"  # Set your path
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

print("Found images:", len(image_paths))

# --- ORB feature detector
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- Initialize pose
cur_pose = np.eye(4)
trajectory = [cur_pose[:3, 3].copy()]

def get_matches(img1, img2):
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return None, None, None

    # Double check types
    if des1.dtype != np.uint8 or des2.dtype != np.uint8:
        des1 = des1.astype(np.uint8)
        des2 = des2.astype(np.uint8)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2, matches

for i in range(1, len(image_paths)):
    print(f"Processing images {i-1} and {i}")
    img1 = cv2.imread(image_paths[i - 1], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"Error loading images {i-1} or {i}. Skipping.")
        continue

    pts1, pts2, matches = get_matches(img1, img2)
    if pts1 is None or pts2 is None or len(matches) < 8:
        print(f"Not enough matches between images {i-1} and {i}. Skipping.")
        continue

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    if E is None:
        print(f"Essential matrix could not be computed for images {i-1} and {i}. Skipping.")
        continue

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # Update current pose
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    cur_pose = cur_pose @ np.linalg.inv(T)  # Chain motion

    trajectory.append(cur_pose[:3, 3].copy())

    # Draw trajectory
    traj = np.zeros((600, 600, 3), dtype=np.uint8)
    scale = 50

    for idx, pt in enumerate(trajectory):
        x = int(pt[0] * scale) + 300
        z = int(pt[2] * scale) + 100
        if 0 <= x < traj.shape[1] and 0 <= z < traj.shape[0]:
            color = (0, 255, 0)
            if idx == len(trajectory) - 1:
                color = (0, 0, 255)  # current point in red
            cv2.circle(traj, (x, z), 2, color, -1)

    step_text = f"Step: {i}/{len(image_paths)-1}"
    cv2.putText(traj, step_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Trajectory", traj)
    #cv2.imshow("Frame", img2)
    cv2.waitKey(100)  # Wait for 100 ms
    if len(trajectory) > 1:
        prev_pos = trajectory[-2]
        cur_pos = trajectory[-1]
        displacement = np.linalg.norm(cur_pos - prev_pos)
        print(f"Displacement at step {i}: {displacement:.4f} units")

cv2.destroyAllWindows()