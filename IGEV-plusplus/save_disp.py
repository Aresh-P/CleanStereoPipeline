import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2
import time
import threading

DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()
    '''
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)
            file_stem = os.path.join(output_directory, imfile1.split('/')[-1])
            disp = disp.cpu().numpy().squeeze()
            if args.save_png:
                disp_16 = np.round(disp * 256).astype(np.uint16)
                skimage.io.imsave(file_stem, disp_16)
            # plt.imsave(file_stem, disp, cmap='jet')

            if args.save_numpy:
                np.save(file_stem.replace('.png', '.npy'), disp)
    '''
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

    # Shared frame and lock
    latest_frame = {'frame': None, 'ret': False}
    frame_lock = threading.Lock()
    stop_flag = {'stop': False}

    def frame_grabber():
        while not stop_flag['stop']:
            ret, frame = cap.read()
            with frame_lock:
                latest_frame['frame'] = frame
                latest_frame['ret'] = ret
            # Sleep a little to avoid busy-waiting
            time.sleep(0.001)

    grabber_thread = threading.Thread(target=frame_grabber, daemon=True)
    grabber_thread.start()

    with torch.no_grad():
        while True:
            # Get the latest frame
            with frame_lock:
                ret = latest_frame['ret']
                frame = latest_frame['frame']
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Split stereo image
            left_img = frame[:, :single_width, :]
            right_img = frame[:, single_width:, :]

            # Rectify
            left_rect = cv2.remap(left_img, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_img, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)

            # Scale rectified images
            left_rect = cv2.resize(left_rect, (left_rect.shape[1] // 5, left_rect.shape[0] // 5), interpolation=cv2.INTER_LINEAR)
            right_rect = cv2.resize(right_rect, (right_rect.shape[1] // 5, right_rect.shape[0] // 5), interpolation=cv2.INTER_LINEAR)

            # Convert to tensor and normalize
            image1 = torch.from_numpy(left_rect).permute(2, 0, 1).float() / 255.0
            image2 = torch.from_numpy(right_rect).permute(2, 0, 1).float() / 255.0
            image1 = image1.to(DEVICE)
            image2 = image2.to(DEVICE)
            image1 = torch.unsqueeze(image1, 0)  # Add batch dimension  
            image2 = torch.unsqueeze(image2, 0)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)

            disp = disp.cpu().numpy().squeeze()
            disp = np.clip(disp, 0, None)
            disp_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(disp, alpha=255.0 / (disp.max() if disp.max() > 0 else 1)),
                cv2.COLORMAP_JET
            )
            result = np.hstack((left_rect, disp_vis))
            result = cv2.resize(result, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Disparity', result)
            # cv2.imshow('Left Rectified', left_rect)
            # cv2.imshow('Right Rectified', right_rect)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    stop_flag['stop'] = True
    grabber_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/igev_plusplus/kitti2015.pth')
    parser.add_argument('--save_png', action='store_true', default=True, help='save output as gray images')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/StereoDatasets/kitti/2015/testing/image_2/*_10.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/StereoDatasets/kitti/2015/testing/image_3/*_10.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/StereoDatasets/kitti/2012/testing/colored_0/*_10.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/StereoDatasets/kitti/2012/testing/colored_1/*_10.png")
    parser.add_argument('--output_directory', help="directory to save output", default="output/kitti2015/disp_0")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    parser.add_argument('--s_disp_range', type=int, default=48, help="max disp of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_range', type=int, default=96, help="max disp of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_range', type=int, default=192, help="max disp of large disparity-range geometry encoding volume")
    parser.add_argument('--s_disp_interval', type=int, default=1, help="disp interval of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_interval', type=int, default=2, help="disp interval of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_interval', type=int, default=4, help="disp interval of large disparity-range geometry encoding volume")
    
    args = parser.parse_args()

    demo(args)
