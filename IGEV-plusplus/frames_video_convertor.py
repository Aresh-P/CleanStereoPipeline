import cv2
import os
import glob

# Folder containing frames
frames_folder = 'ov_output'
# Output video file
output_video = 'output_video_l.mp4'

# Get list of image files (sorted)
image_files = sorted(glob.glob(os.path.join(frames_folder, '*.png')))  # Change extension if needed

# Read first image to get frame size
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

# Define video writer (using mp4v codec, you can change if needed)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 15  # Set desired frames per second
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for img_file in image_files:
    img = cv2.imread(img_file)
    video_writer.write(img)

video_writer.release()
print(f'Video saved as {output_video}')