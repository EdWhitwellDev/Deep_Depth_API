import cv2
import numpy as np

class VOManager:
    scale = 0.5
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

    map1L, map2L = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, (w, h), cv2.CV_16SC2)
    map1R, map2R = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, (w, h), cv2.CV_16SC2)

    dim_divis_factor = 32

    def __init__(self, calib, stereo=None):
        self.calib = calib
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.stereo_manager = stereo
        print("Camera intrinsics loaded:", self.calib)

    def process_frame(self, left_frame, right_frame):
        # Undistort the image
        left_undistorted = cv2.remap(left_frame, self.map1L, self.map2L, cv2.INTER_LINEAR)
        right_undistorted = cv2.remap(right_frame, self.map1R, self.map2R, cv2.INTER_LINEAR)
        left_undistorted = self.scale_image(left_undistorted)
        right_undistorted = self.scale_image(right_undistorted)
        # Crop the images to the valid ROI
        left_undistorted = self.crop_to_dim_factor(left_undistorted)
        right_undistorted = self.crop_to_dim_factor(right_undistorted)
        return left_undistorted, right_undistorted
    
    def scale_image(self, image):
        return cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
    def crop_to_dim_factor(self, image):
        h, w = image.shape[:2]
        h_crop = h - (h % self.dim_divis_factor)
        w_crop = w - (w % self.dim_divis_factor)
        return image[:h_crop, :w_crop]

    def get_matches(self, img1, img2):
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            return None, None, None

    # Double check types
        if des1.dtype != np.uint8 or des2.dtype != np.uint8:
            des1 = des1.astype(np.uint8)
            des2 = des2.astype(np.uint8)

        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2, matches
    
    def depth_projection(self, left_image, right_image, points):
        # Compute the depth map
        depth = self.stereo_manager.predict_depth(left_image, right_image)
        if depth is None:
            return None
        
        fx, fy = self.calib['K1'][0, 0], self.calib['K1'][1, 1]
        cx, cy = self.calib['K1'][0, 2], self.calib['K1'][1, 2]
        points_3d = []
        valid_im_points = []
        for pt in points:
            z = depth[int(pt[1]), int(pt[0])]
            if z > 0:
                x = (pt[0] - cx) * z / fx
                y = (pt[1] - cy) * z / fy
                points_3d.append((x, y, z))
                valid_im_points.append(pt)
        return np.array(points_3d, dtype=np.float32), np.array(valid_im_points, dtype=np.float32)

    def compute_trajectory(self, left_images, right_images):
        trajectory = []
        previous_pose = np.eye(4)
        trajectory.append(previous_pose[:3, 3].copy())
        previous_img = None
        previous_img_right = None
        for index, img in enumerate(left_images):
            undistorted_img_left, undistorted_img_right = self.process_frame(img, right_images[index])
            pts1, pts2, matches = self.get_matches(previous_img, undistorted_img_left if previous_img is not None else undistorted_img_left)
            if pts1 is None or pts2 is None or len(matches) < 8:
                previous_img = undistorted_img_left
                previous_img_right = undistorted_img_right
                print(f"Not enough matches between images {index-1} and {index}. Skipping.")
                continue

            if self.stereo_manager:
                points_3d, valid_im_points = self.depth_projection(left_images[index-1], right_images[index-1], pts1)
                if points_3d is None or len(points_3d) < 8:
                    previous_img = undistorted_img_left
                    previous_img_right = undistorted_img_right
                    print(f"Depth projection failed between images {index-1} and {index}. Skipping.")
                    continue
                success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, valid_im_points, self.calib['K1'], None)
                if not success:
                    previous_img = undistorted_img_left
                    previous_img_right = undistorted_img_right
                    print(f"Pose estimation failed between images {index-1} and {index}. Skipping.")
                    continue
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec.flatten()
                previous_pose = previous_pose @ np.linalg.inv(T)
                trajectory.append(previous_pose[:3, 3].copy())

            else:
            # Estimate the motion between the previous and current frame
                E, mask = cv2.findEssentialMat(pts1, pts2, self.calib['K1'])
                if E is None:
                    previous_img = undistorted_img_left
                    previous_img_right = undistorted_img_right
                    print(f"Essential matrix computation failed between images {index-1} and {index}. Skipping.")
                    continue
                
                # Recover the pose from the essential matrix
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.calib['K1'])
                if R is None or t is None:
                    previous_img = undistorted_img_left
                    previous_img_right = undistorted_img_right
                    print(f"Pose recovery failed between images {index-1} and {index}. Skipping.")
                    continue

                # Update the trajectory
                previous_pose = previous_pose @ np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
                trajectory.append(previous_pose[:3, 3].copy())
                previous_img = undistorted_img_left
                previous_img_right = undistorted_img_right

        return trajectory