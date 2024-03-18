import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import pypose as pp

def parse_timestamp(file_name):
    # Assumes the file name is the timestamp
    return int(os.path.basename(file_name).split('.')[0])

class TUMVisualInertialDataset(Dataset):
    def __init__(self, path, transforms, sequence_length=5, skip_frames=10):
        self.path = path
        self.sequence_length = sequence_length
        self.transforms = transforms

        # Load image paths
        self.cam0_images = sorted([os.path.join(path, 'cam0/images', f) for f in os.listdir(os.path.join(path, 'cam0/images')) if f.endswith('.png')])[::skip_frames]
        self.cam1_images = sorted([os.path.join(path, 'cam1/images', f) for f in os.listdir(os.path.join(path, 'cam1/images')) if f.endswith('.png')])[::skip_frames]

        # Load IMU and ground truth data
        self.imu_data = np.loadtxt(os.path.join(path, 'imu.txt'), comments='#', delimiter=' ')
        self.ground_truth = np.loadtxt(os.path.join(path, 'gt_imu.csv'), comments='#', delimiter=',')
        self.ground_truth[:, [-4, -1]] = self.ground_truth[:, [-1, -4]] # Quaternion wxyz to xyzw
        
        self.K = np.array([
            [190.97847715128717, 0, 254.93170605935475],
            [0, 190.9733070521226, 256.8974428996504],
            [0, 0, 1]
        ])
        
        self.T_cam_imu = np.array([
            [-0.9995250378696743, 0.029615343885863205, -0.008522328211654736, 0.04727988224914392],
            [0.0075019185074052044, -0.03439736061393144, -0.9993800792498829, -0.047443232143367084],
            [-0.02989013031643309, -0.998969345370175, 0.03415885127385616, -0.0681999605066297],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.distortion_coef = np.array([0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202, 0.00020293673591811182])
        
        self.image_shape = np.array([512, 512])

        # Adjust K due to resizing of image.
        self.image_resized_shape = self.transforms.transforms[0].get_size(*self.image_shape)
        scale_x = self.image_shape[0] / self.image_resized_shape[0]
        scale_y = self.image_shape[1] / self.image_resized_shape[1]
        self.K[0,0] *= scale_x
        self.K[0,-1] *= scale_x
        self.K[1,1] *= scale_y
        self.K[1,-1] *= scale_y
        
        # No undistortion (due to fisheye), rather, we just use the camera model to project points. 
        #self.K, self.roi = cv2.getOptimalNewCameraMatrix(self.K_distorted, self.distortion_coef, self.image_shape, 0, self.image_shape)

    def __len__(self):
        return len(self.cam0_images) - self.sequence_length + 1

    def __getitem__(self, idx):
        cam0_frames = []
        cam1_frames = []
        imu_readings = []
        gt_poses = []

        # Get timestamps of the first and last images in the sequence
        start_timestamp = parse_timestamp(self.cam0_images[idx])
        end_timestamp = parse_timestamp(self.cam0_images[idx + self.sequence_length - 1])

        # Gather all IMU readings between these two timestamps
        imu_start = np.searchsorted(self.imu_data[:, 0], start_timestamp)
        imu_end = np.searchsorted(self.imu_data[:, 0], end_timestamp)
        imu_readings = self.imu_data[imu_start:imu_end]

        # Handle ground truth poses for each image
        for i in range(self.sequence_length):
            cam0_img_path = self.cam0_images[idx + i]
            cam1_img_path = self.cam1_images[idx + i]

            # Load the images and undistort
            img0 = cv2.imread(cam0_img_path)
            img1 = cv2.imread(cam1_img_path)

            # Transform for DepthAnything and adjust K.
            img0 = self.transforms({'image': img0})['image']
            img1 = self.transforms({'image': img1})['image']

            img0= cv2.cvtColor(img0, cv2.COLOR_BGR2RGB) / 255.0
            img1= cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0

            cam0_frames.append(img0)
            cam1_frames.append(img1)

            # Find the nearest ground truth pose
            gt_ind = np.searchsorted(self.ground_truth[:, 0], parse_timestamp(cam0_img_path))
            gt_poses.append(self.ground_truth[gt_ind, 1:])

        # Convert everything to PyTorch tensors
        cam0_frames = torch.stack([torch.tensor(frame) for frame in cam0_frames]).float().permute(0,3,1,2)
        cam1_frames = torch.stack([torch.tensor(frame) for frame in cam1_frames]).float().permute(0,3,1,2)
        t = torch.tensor(imu_readings[:, 0]) / 1e9
        dt = torch.diff(t, append=t[-2:-1]).unsqueeze(1).float()
        gyro = torch.tensor(imu_readings[:, [1,2,3]]).float()
        acc = torch.tensor(imu_readings[:, [4,5,6]]).float()
        gt_poses = pp.SE3(np.vstack(gt_poses)) # Might need @ self.T_cam0_imu

        all_poses = pp.SE3(self.ground_truth[:,1:])
        #velocities = self.calculate_velocities(all_poses, t)

        return {
            'cam0': cam0_frames,
            'cam1': cam1_frames,
            'dt': dt,
            'gyro': gyro,
            'acc': acc,
            'gt_pose': gt_poses,
        }

    def calculate_velocities(self, poses, timestamps):
        # Ensure timestamps are in seconds and in float format
        time_diffs = (timestamps[1:] - timestamps[:-1]).float()
        # Calculate pose differences in Lie Algebra
        pose_diffs = (poses[1:] @ poses[:-1].Inv()).Log()


        # Calculate velocities by dividing pose differences by time intervals
        # Assuming uniform time intervals for simplicity; adjust as needed for variable intervals
        print(pose_diffs.shape, time_diffs.shape)
        velocities = pose_diffs[:, :3] / time_diffs  # Add an extra dimension for broadcasting

        # Handle edge cases for the first and last velocities
        # Here, just duplicating the second and second-to-last velocities for simplicity
        first_velocity = velocities[0].unsqueeze(0)  # Add dimension to match the velocities shape
        last_velocity = velocities[-1].unsqueeze(0)
        
        # Concatenate to get the full velocities tensor
        velocities = torch.cat([first_velocity, velocities, last_velocity], dim=0)
        print(velocities.shape)
        return velocities