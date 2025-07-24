import sys
sys.path.append('core')
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from monster import Monster
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2
import torch.nn.functional as F
import sys
import time

DEVICE = 'cuda'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class NormalizeTensor(object):
    """Normalize a tensor by given mean and std."""
    
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            
        Returns:
            Tensor: Normalized Tensor image.
        """
        # Ensure mean and std have the same number of channels as the input tensor
        Device = tensor.device
        self.mean = self.mean.to(Device)
        self.std = self.std.to(Device)

        # Normalize the tensor
        if self.mean.ndimension() == 1:
            self.mean = self.mean[:, None, None]
        if self.std.ndimension() == 1:
            self.std = self.std[:, None, None]

        return (tensor - self.mean) / self.std

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    # Load calibration maps
    calib_data = np.load("ov_calibration_from_pipeline_clean.npz")
    map1_left = calib_data['map1_left']
    map2_left = calib_data['map2_left']
    map1_right = calib_data['map1_right']
    map2_right = calib_data['map2_right']

    model = torch.nn.DataParallel(Monster(args), device_ids=[0])

    assert os.path.exists(args.restore_ckpt)
    checkpoint = torch.load(args.restore_ckpt)
    ckpt = dict()
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    for key in checkpoint:
        if key.startswith("module."):
            ckpt[key] = checkpoint[key]
        else:
            ckpt["module." + key] = checkpoint[key]

    model.load_state_dict(ckpt, strict=True)
    model = model.module
    model.to(DEVICE)
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = NormalizeTensor(mean, std)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    # Set device resolution to 4416x1242
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4416)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1242)

    # Set your stereo image width/height here
    stereo_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    stereo_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    single_width = stereo_width // 2

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Split stereo image
            left_img = frame[:, :single_width, :]
            right_img = frame[:, single_width:, :]

            # Rectify
            left_rect = cv2.remap(left_img, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_img, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)

            # Scale down rectified images by a factor of 2
            left_rect = cv2.resize(left_rect, (left_rect.shape[1] // 5, left_rect.shape[0] // 5), interpolation=cv2.INTER_LINEAR)
            right_rect = cv2.resize(right_rect, (right_rect.shape[1] // 5, right_rect.shape[0] // 5), interpolation=cv2.INTER_LINEAR)

            # Convert to tensor and normalize
            image1 = torch.from_numpy(left_rect).permute(2, 0, 1).float() / 255.0
            image2 = torch.from_numpy(right_rect).permute(2, 0, 1).float() / 255.0
            image1 = normalize(image1)[None].to(DEVICE)
            image2 = normalize(image2)[None].to(DEVICE)

            print("Image shapes (pre-pad):", image1.shape, image2.shape)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            print("Image shapes (post-pad):", image1.shape, image2.shape)

            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)
            disp = disp.cpu().numpy().squeeze()
            disp = np.clip(disp, 0, None)
            disp_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(disp, alpha=255.0 / (disp.max() if disp.max() > 0 else 1)),
                cv2.COLORMAP_JET
            )
            result = np.hstack((left_rect, disp_vis))
            result = cv2.resize(result, (result.shape[1] * 2, result.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Disparity', result)
            # cv2.imshow('Left Rectified', left_rect)
            # cv2.imshow('Right Rectified', right_rect)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    '''
    calib_data = np.load("ov_calibration_from_pipeline_clean.npz")
    map1_left = calib_data['map1_left']
    map2_left = calib_data['map2_left']
    map1_right = calib_data['map1_right']
    map2_right = calib_data['map2_right']

    model = torch.nn.DataParallel(Monster(args), device_ids=[0])

    assert os.path.exists(args.restore_ckpt)
    checkpoint = torch.load(args.restore_ckpt)
    ckpt = dict()
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    for key in checkpoint:
        # ckpt['module.' + key] = checkpoint[key]
        if key.startswith("module."):
            ckpt[key] = checkpoint[key]  # 保持原样
        else:
            ckpt["module." + key] = checkpoint[key]  # 添加 "module."

    model.load_state_dict(ckpt, strict=True)

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = NormalizeTensor(mean, std)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            start_time = time.time()
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)

            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference time: {inference_time:.4f} seconds")
            disp = padder.unpad(disp)
            file_stem = os.path.join(output_directory, os.path.basename(imfile1))
            disp = disp.cpu().numpy().squeeze()
            disp = np.round(disp * 256).astype(np.uint16)
            # skimage.io.imsave(file_stem, disp)
            disp_color = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=255.0/disp.max()), cv2.COLORMAP_JET)
            skimage.io.imsave(file_stem, disp_color)
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disp.squeeze())
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="pretrained\\kitti.pth")

    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data2/cjd/StereoDatasets/kitti//2015/testing/image_2/*_10.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data2/cjd/StereoDatasets/kitti/2015/testing/image_3/*_10.png")
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="dataset\\KITTI15\\training\\image_2\\*_10.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="dataset\\KITTI15\\training\\image_3\\*_10.png")

    parser.add_argument('--output_directory', help="directory to save output", default="kitti_2012")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()

    demo(args)
