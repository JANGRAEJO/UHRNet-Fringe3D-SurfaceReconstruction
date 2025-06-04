import os
from PIL import Image

# === CONFIGURATION ===
image_folder = 'output_vis'
gif_output = 'combined_visualization.gif'
fps = 2  # 1 frame every 0.5 seconds

# === LOAD IMAGES ===
images_2d = sorted([img for img in os.listdir(image_folder) if img.startswith("vis_") and img.endswith(".png")])
images_3d = sorted([img for img in os.listdir(image_folder) if img.startswith("surface3D_") and img.endswith(".png")])

if not images_2d or not images_3d:
    raise RuntimeError("Check that vis_*.png and surface3D_*.png files exist in output_vis/")

# === COMBINE IMAGES VERTICALLY (2D on top, 3D on bottom) ===
combined_frames = []
for img_3d, img_2d in zip(images_3d, images_2d):
    path_3d = os.path.join(image_folder, img_3d)
    path_2d = os.path.join(image_folder, img_2d)

    im_3d = Image.open(path_3d)
    im_2d = Image.open(path_2d)

    # Resize to match widths
    target_width = max(im_2d.width, im_3d.width)
    if im_2d.width != target_width:
        im_2d = im_2d.resize((target_width, im_2d.height))
    if im_3d.width != target_width:
        im_3d = im_3d.resize((target_width, im_3d.height))

    total_height = im_2d.height + im_3d.height
    combined = Image.new('RGB', (target_width, total_height))
    combined.paste(im_2d, (0, 0))                # 2D on top
    combined.paste(im_3d, (0, im_2d.height))      # 3D on bottom

    combined_frames.append(combined)

# === SAVE AS GIF ===
combined_frames[0].save(
    gif_output,
    save_all=True,
    append_images=combined_frames[1:],
    duration=int(1000 / fps),
    loop=0
)

print(f"✅ GIF saved to: {gif_output}")
