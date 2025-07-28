from matplotlib import pyplot as plt
import numpy as np
import torch

class MonoModel:
    def __init__(self):
        self.mono_model_type = "MiDaS_small"
        self.mono_model = torch.hub.load("intel-isl/MiDaS", self.mono_model_type, trust_repo=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.mono_model.to(self.device)
        self.mono_model.eval()
        mono_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.mono_model_type == "DPT_Large" or self.mono_model_type == "DPT_Hybrid":
            self.mono_transform = mono_transforms.dpt_transform
        else:
            self.mono_transform = mono_transforms.small_transform

    def clean(self, data, min, max):
        data[data < min] = 0
        data[data > max] = 0  # Remove outliers
        return data
    
    def generate_projection_matrix(image):
        h, w = image.shape[:2]
        focal_length = 0.5 * w / np.tan(0.5 * np.deg2rad(60))
        projection_matrix = np.array([[focal_length, 0, w / 2],
                                       [0, focal_length, h / 2],
                                       [0, 0, 1]])
        return projection_matrix
    
    def predict(self, image):
        with torch.no_grad():
            mono_input = self.mono_transform(image).to(self.device)
            mono_depth = self.mono_model(mono_input)
            mono_depth = torch.nn.functional.interpolate(
                mono_depth.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
            return mono_depth
    
    def predict_depth(self, image):
        depth_map = self.predict(image)

        average_depth = np.mean(depth_map)
        std_depth = np.std(depth_map)

        depth_map = self.clean(depth_map, average_depth - 2 * std_depth, average_depth + 2 * std_depth)

        max_depth = np.max(depth_map)
        depth_map = max_depth - depth_map  # Invert depth map

        return depth_map        
    

if __name__ == "__main__":
    mono_model = MonoModel()
    # Example usage
    import cv2
    image = cv2.imread("client/DepthImages/left4.jpg")
    depth_map = mono_model.predict_depth(image)
    np.save("client/DepthImages/mono_depth_map.npy", depth_map)
    plt.imshow(depth_map, cmap='plasma')
    plt.show()
