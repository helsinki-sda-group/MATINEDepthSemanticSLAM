import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import pypose as pp
import pandas as pd
from pytorch3d.renderer.fisheyecameras import FishEyeCameras


class UZH_FPV_Dataset(Dataset):
    def __init__(self, path, transforms, sequence_length=5, skip_frames=10):
        self.path = path
        self.sequence_length = sequence_length
        self.transforms = transforms

        # Associate images to times.
        df = pd.read_csv('/home/ilari/Downloads/indoor_forward_6_snapdragon_with_gt/left_images.txt', delimiter=' ', skiprows=1, header=None)
        self.image_times = list(df[1])[::skip_frames]
        self.image_paths = list(df[2])[::skip_frames]

        # Load image paths
        self.cam0_images = [os.path.join(self.path, imgpath) for imgpath in self.image_paths]

        # Load IMU and ground truth data
        self.imu_data = np.loadtxt(os.path.join(path, 'imu.txt'), comments='#', delimiter=' ')[:, 1:]
        self.ground_truth = np.loadtxt(os.path.join(path, 'groundtruth.txt'), comments='#', delimiter=' ')

        # Messes up the integration big time.
        # self.T_cam_imu = pp.mat2SE3(torch.tensor([
        #     [-0.011823057800830705, -0.9998701444077991, -0.010950325390841398, -0.057904961033265645],
        #     [0.011552991631909482, 0.01081376681432078, -0.9998747875767439, 0.00043766687615362694],
        #     [0.9998633625093938, -0.011948086424720228, 0.011423639621249038, -0.00039944945687402214],
        #     [0.0, 0.0, 0.0, 1.0]
        # ]))
        # self.ground_truth[:, 1:] = (self.T_cam_imu.Inv() @ pp.SE3(self.ground_truth[:,1:])).tensor().numpy()

        self.velocities = self.get_velocities(self.ground_truth[:,0], pp.SE3(self.ground_truth[:,1:]))
        
        self.K = torch.tensor([
            [278.66723066149086, 0, 319.75221200593535],
            [0,  278.48991409740296, 241.96858910358173],
            [0, 0, 1]
        ])
        
        self.distortion_coef = torch.tensor([-0.013721808247486035, 0.020727425669427896, -0.012786476702685545, 0.0025242267320687625, 0, 0])
        
        self.image_shape = np.array([640, 480])

        # Adjust K due to resizing of image.
        self.image_resized_shape = self.transforms.transforms[0].get_size(*self.image_shape)
        scale_x = self.image_shape[0] / self.image_resized_shape[0]
        scale_y = self.image_shape[1] / self.image_resized_shape[1]
        self.K[0,0] *= scale_x
        self.K[0,-1] *= scale_x
        self.K[1,1] *= scale_y
        self.K[1,-1] *= scale_y

        print(self.K, self.distortion_coef, self.image_resized_shape)

        # Initialize the camera model.
        self.initialize_camera_model()
        
        # No undistortion (due to fisheye), rather, we just use the camera model to project points. 
        #self.K, self.roi = cv2.getOptimalNewCameraMatrix(self.K_distorted, self.distortion_coef, self.image_shape, 0, self.image_shape)

    def initialize_camera_model(self):
        self.camera_model = FishEyeCameras(
        focal_length=torch.tensor([278.66723066149086]).repeat(2, 1),
        principal_point=torch.tensor([[319.75221200593535, 241.96858910358173]]),
        radial_params=torch.tensor(
            [
                [
                    0.373004838186,
                    0.372994740336,
                    0.498890050897,
                    0.502729380663,
                    0.00348238940225,
                    0.000715034845216,
                ]
            ]
        ),
        R=torch.tensor([np.eye(3)]),
        T=torch.tensor([[0.0, 0.0, 0.0]]),
        world_coordinates=True,
        use_radial=True,
        use_tangential=False,
        use_thin_prism=False,
        device='cpu',
        image_size=self.image_resized_shape,
        )


    def __len__(self):
        return len(self.cam0_images) - self.sequence_length + 1

    def __getitem__(self, idx):
        cam0_frames = []
        imu_readings = []
        gt_poses = []

        # Get timestamps of the first and last images in the sequence.
        start_timestamp = self.image_times[idx]
        end_timestamp = self.image_times[idx + self.sequence_length - 1]


        # Gather all IMU readings between these two timestamps.
        imu_start = np.searchsorted(self.imu_data[:, 0], start_timestamp) - 1
        imu_end = np.searchsorted(self.imu_data[:, 0], end_timestamp) - 1
        imu_readings = self.imu_data[imu_start:imu_end]

        # Get velocity at the start of the sequence.
        velocity_index = np.searchsorted(self.velocities[:,0], start_timestamp) - 1
        velocity = self.velocities[velocity_index, 1:]

        # Handle ground truth poses for each image
        for i in range(self.sequence_length):
            img_idx = idx + i
            cam0_img_path = self.cam0_images[img_idx]

            # Load the images and undistort
            img0 = cv2.imread(cam0_img_path)

            # Transform for DepthAnything and adjust K.
            img0 = self.transforms({'image': img0})['image']

            img0= cv2.cvtColor(img0, cv2.COLOR_BGR2RGB) / 255.0

            cam0_frames.append(img0)

            # Find the nearest ground truth pose
            gt_ind = np.searchsorted(self.ground_truth[:, 0], self.image_times[img_idx]) - 1
            gt_poses.append(self.ground_truth[gt_ind, 1:])

        # Convert everything to PyTorch tensors
        cam0_frames = torch.stack([torch.tensor(frame) for frame in cam0_frames]).float().permute(0,3,1,2)
        t = torch.tensor(imu_readings[:, 0]) # Should be in seconds
        dt = t[1:] - t[:-1] 
        dt = torch.cat([dt, dt[-2:-1]]).unsqueeze(1).float()
        gyro = torch.tensor(imu_readings[:, [1,2,3]]).float()
        acc = torch.tensor(imu_readings[:, [4,5,6]]).float()
        velocity = torch.tensor(velocity).float()
        gt_poses = pp.SE3(torch.tensor(np.vstack(gt_poses)).float())

        return {
            'cam0': cam0_frames,
            'dt': dt,
            'gyro': gyro,
            'acc': acc,
            'gt_pose': gt_poses,
            'velocity': velocity,
        }

    def get_velocities(self, t, pose_sequence):
        """
        Calculate velocity vectors for a sequence of poses in meters per second.

        Parameters:
        t (torch.Tensor): Timestamps for each pose in milliseconds.
        pose_sequence (pp.SE3): Sequence of poses in SE3 format with translation components in meters.

        Returns:
        torch.Tensor: [timestamps, velocity3d] for each pose in meters per second.
        """
        # Convert timestamps from nanoseconds to seconds for proper velocity calculation
        t_seconds = t # Should be in seconds

        # Extract translation components (assumed to be in meters)
        translations = pose_sequence.translation().numpy()

        # Compute displacement vectors for consecutive poses (in meters)
        displacements = translations[1:] - translations[:-1]

        # Calculate time differences between consecutive timestamps (in seconds)
        time_differences = (t_seconds[1:] - t_seconds[:-1]).reshape(-1,1)

        # Ensure time_differences is not zero to avoid division by zero errors
        #time_differences = np.clamp(time_differences, min=1e-9)

        # Divide displacements by time differences to get velocity vectors (in meters per second)
        velocity_vectors = displacements / time_differences
        velocity_vectors = np.concatenate((np.zeros((1,3)), velocity_vectors))


        return np.hstack([t.reshape(-1,1), velocity_vectors])