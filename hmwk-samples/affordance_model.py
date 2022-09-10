from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

import torchvision.transforms.functional as TF

from common import draw_grasp

"""
Excerpt from robotics homework assignment teaching a robot
to grab objects (more like computer vision learning).

Code written by me:
- class AffordanceDataset: getitem
- class AffordanceModel: init(), forward(), predict(), get_criterion(), visualize(), predict_grasp()
"""


def get_gaussian_scoremap(
        shape: Tuple[int, int],
        keypoint: np.ndarray,
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return:
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # TODO: complete this method
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================

        #get data out
        rgbdata = np.array(data['rgb'], dtype=np.uint8) #(H, W, 3) uint8 [0, 255]
        center = data['center_point'] #(2, ) float32 [0, 127]
        angle = data['angle'].item() #float32 [0, 180]

        #find closest 22.5 increment
        closest_angle = 22.5 * round(angle/22.5)

        #define augmentation sequence: rotate by -closest_angle
        seq = iaa.Sequential([iaa.Affine(rotate = -closest_angle)])

        #define keypoints: center point
        kps = KeypointsOnImage([Keypoint(x=center[0], y=center[1]),], shape=rgbdata.shape)

        #apply augmentation
        image_rot, kps_rot = seq(image=rgbdata, keypoints=kps)
        #print(image_rot.dtype)

        #recover centerpoint
        center_rot = np.array([kps_rot[0].x, kps_rot[0].y], dtype=np.float32)

        #get scoremap from rotated centerpoint
        scoremap = get_gaussian_scoremap((128, 128), center_rot).reshape(1, 128, 128)

        #normalize image to (3, 128, 128) (3, H, W) float32 [0, 1]
        image_rot = image_rot.astype(np.float32) / 255

        image_rot = np.moveaxis(image_rot, 2, 0)

        #assert(image_rot.shape == (3, 128, 128))
        #print("input dtype", image_rot.dtype, "target dtype", scoremap.dtype, "input max min", np.max(image_rot), np.min(image_rot), "target max min", np.max(scoremap), np.min(scoremap))

        dataOut = {
            'input': torch.from_numpy(image_rot),
            'target': torch.from_numpy(scoremap)
        }
        return dataOut
        # ===============================================================================


class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray,
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(self, rgb_obs: np.ndarray
            ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given a RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: complete this method (prediction)
        # Hint: why do we provide the model's device here?


        #create rotated images
        #convert to (3, H, W) float32 and normalize to 255
        #stack rotated images
        #feed into model
        #find max affordance pixel: [0] is which rotated image gets picked, [2],[3] is rotated grasp coordinate


        #create rotated images:
        rot_imgs_numpy = [None]*8 #list of np arrays (H, W, 3) uint8 [0-255]
        rot_imgs_tensor = [None]*8 #list of tensors (3, H, W) float32 [0-1]
        for i in range(8):
            rotation_angle = i*22.5
            seq = iaa.Sequential([iaa.Affine(rotate=rotation_angle)])
            rotated_img = seq(image=rgb_obs)
            #print(type(rotated_img), rotated_img.dtype, rotated_img.shape, np.max(rotated_img), np.min(rotated_img))
            #add to numpy list
            rot_imgs_numpy[i] = rotated_img

            #convert to proper format
            tensor_img = rotated_img.astype(np.float32) #convert to float
            tensor_img = np.moveaxis(tensor_img, 2, 0) #move axis
            tensor_img = tensor_img / 255 #normalize
            #print(type(tensor_img), tensor_img.dtype, tensor_img.shape, np.max(tensor_img), np.min(tensor_img))
            #add to tensor list as tensor
            rot_imgs_tensor[i] = torch.from_numpy(tensor_img)



        #stack rotated images
        stackedData = torch.stack(rot_imgs_tensor, dim=0)
        assert(stackedData.shape == (8, 3, 128, 128))

        #feed into model
        with torch.no_grad():
            data = stackedData.to(device)
            prediction = self.predict(data)

        #find max affordance pixel
        max_aff = np.unravel_index(torch.argmax(prediction), prediction.shape)

        rot_ind = max_aff[0]
        rot_coords = [max_aff[3], max_aff[2]] #grasp coordinate in rotated image, flipped for convention

        #picked image, as numpy array (H, W, 3) uint 8 [0-255]
        picked_img = rot_imgs_numpy[rot_ind]

        #get unrotated grasp coordinate:
        image_rot_angle = rot_ind * 22.5 #amount that picked_img is rotated by
        seq = iaa.Sequential([iaa.Affine(rotate=-image_rot_angle)])#want to UNROTATE it, so rotate by negative that amount
        kps = KeypointsOnImage([Keypoint(x = rot_coords[0], y = rot_coords[1]),], shape=picked_img.shape)

        image_unrot, kps_unrot = seq(image=picked_img, keypoints=kps)


        # ===============================================================================
        coord, angle = None, None

        #recover unrotated coordinate from keypoint
        coord = (round(kps_unrot[0].x), round(kps_unrot[0].y))
        #grasp angle is -image rotation angle
        grasp_angle = -image_rot_angle

        # TODO: complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================

        #draw grasp onto picked_img

        draw_grasp(picked_img, rot_coords, 0) #uint8 image

        #convert graspdrawn image to (3, 128, 128) float32 [0-1]
        img_with_grasp = picked_img.astype(np.float32) #convert to float32
        img_with_grasp = img_with_grasp / 255 #normalize
        img_with_grasp = np.moveaxis(img_with_grasp, 2, 0) #reshape


        #pair images with predictions; visualize() requires float32

        paired_imgs = [None]*8
        for i in range(8):
            if (i == rot_ind):
                float_img = img_with_grasp
            else:
                float_img = np.array(rot_imgs_tensor[i], dtype=np.float32)
            pred_img = np.array(prediction[i], dtype=np.float32)

            paired_img = self.visualize(float_img, pred_img)
            #draw gray line
            paired_img = np.vstack((paired_img, np.ones((1, 256, 3))*127)).astype(np.uint8)
            paired_imgs[i] = paired_img


        #stack images with lines inbetween
        row1 = np.hstack((paired_imgs[0], paired_imgs[1]))
        row2 = np.hstack((paired_imgs[2], paired_imgs[3]))
        row3 = np.hstack((paired_imgs[4], paired_imgs[5]))
        row4 = np.hstack((paired_imgs[6], paired_imgs[7]))
        vis_img = np.vstack((row1, row2, row3, row4))





        #print(coord, grasp_angle)
        # ===============================================================================
        return coord, grasp_angle, vis_img
