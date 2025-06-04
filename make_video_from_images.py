import cv2
import os


# === CONFIGURATION ===
image_folder = 'output_vis'
video_name = 'visualization_result.mp4'
fps = 10  



images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

if not images:
    raise RuntimeError("No .png files found in output_vis. Did you run visualize_prediction.py first?")



first_img_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_img_path)
height, width, layers = frame.shape



video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for image in images:
    img_path = os.path.join(image_folder, image)
    video.write(cv2.imread(img_path))

video.release()
print(f"✅ Video saved to '{video_name}'")
