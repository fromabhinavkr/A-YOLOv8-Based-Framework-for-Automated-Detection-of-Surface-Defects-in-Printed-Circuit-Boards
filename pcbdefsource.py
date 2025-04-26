#  PCB Defect Detection using YOLOv8

# Step 1: Install YOLOv8 and Dependencies


# Step 1b: Import libraries
import os
import glob
import shutil
import zipfile
import xmltodict
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from PIL import Image as PILImage

#  working directory
os.chdir(os.path.dirname(__file__))

# Step 2: Extract Dataset 
zip_path = 'pcbarchive.zip'  #data set file
extract_path = 'pcb_data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Step 3: Conversion Pascal VOC XML to YOLO Format
#  class mapping
classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
class_to_id = {cls: i for i, cls in enumerate(classes)}

# Paths
annotation_root = os.path.join(extract_path, 'PCB_DATASET/Annotations')
image_root = os.path.join(extract_path, 'PCB_DATASET/Images')
yolo_base = 'dataset'
yolo_image_dir = os.path.join(yolo_base, 'images/train')
yolo_label_dir = os.path.join(yolo_base, 'labels/train')
os.makedirs(yolo_image_dir, exist_ok=True)
os.makedirs(yolo_label_dir, exist_ok=True)

#  Recursively gather all images
image_path_map = {}
for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
    for img_path in glob.glob(f'{image_root}/**/{ext}', recursive=True):
        filename = os.path.splitext(os.path.basename(img_path))[0].strip().lower()
        image_path_map[filename] = img_path

print("✅ Sample of collected image keys:", list(image_path_map.keys())[:5])

#  Gather all XML annotations
xml_files = glob.glob(f'{annotation_root}/**/*.xml', recursive=True)

missing_images = []
converted_count = 0

# Convert annotations
for xml_file in xml_files:
    with open(xml_file) as f:
        data = xmltodict.parse(f.read())

    raw_img_filename = data['annotation']['filename'].strip()
    img_name_no_ext = os.path.splitext(raw_img_filename)[0].strip().lower()

    img_path = image_path_map.get(img_name_no_ext)

    if not img_path or not os.path.exists(img_path):
        missing_images.append((raw_img_filename, img_name_no_ext))
        continue

    final_img_filename = os.path.basename(img_path)
    shutil.copy(img_path, os.path.join(yolo_image_dir, final_img_filename))

    label_path = os.path.join(yolo_label_dir, os.path.splitext(final_img_filename)[0] + '.txt')
    with open(label_path, 'w') as f:
        objs = data['annotation'].get('object', [])
        if not isinstance(objs, list):
            objs = [objs]
        for obj in objs:
            cls_name = obj['name']
            if cls_name not in class_to_id:
                continue
            cls_id = class_to_id[cls_name]

            bbox = obj['bndbox']
            xmin = int(bbox['xmin'])
            ymin = int(bbox['ymin'])
            xmax = int(bbox['xmax'])
            ymax = int(bbox['ymax'])

            img_width = int(data['annotation']['size']['width'])
            img_height = int(data['annotation']['size']['height'])
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            w = (xmax - xmin) / img_width
            h = (ymax - ymin) / img_height

            f.write(f"{cls_id} {x_center} {y_center} {w} {h}\n")

    converted_count += 1

img_count = len(glob.glob(os.path.join(yolo_image_dir, '*')))
print(f"✅ Total images copied to YOLO train folder: {img_count}")
print(f"✅ Total annotations converted: {converted_count}")
if missing_images:
    print("⚠️ Sample of missing image matches:")
    for miss in missing_images[:5]:
        print(f"XML filename: {miss[0]} | Processed key: {miss[1]}")
    print("⚠️ Total missing image matches:", len(missing_images))

# 🔀 Split into Train and Validation Sets
image_filenames = [os.path.basename(p) for p in glob.glob(os.path.join(yolo_image_dir, '*'))]
print("Sample images:", image_filenames[:5])

if len(image_filenames) == 0:
    raise ValueError("No images found in the training folder. Please check your dataset path and extraction.")

train_imgs, val_imgs = train_test_split(image_filenames, test_size=0.2, random_state=42)

# Create validation folders
yolo_image_val_dir = os.path.join(yolo_base, 'images/val')
yolo_label_val_dir = os.path.join(yolo_base, 'labels/val')
os.makedirs(yolo_image_val_dir, exist_ok=True)
os.makedirs(yolo_label_val_dir, exist_ok=True)

# Move validation images and labels
for img in val_imgs:
    shutil.move(os.path.join(yolo_image_dir, img), os.path.join(yolo_image_val_dir, img))
    label = os.path.splitext(img)[0] + '.txt'
    src_label = os.path.join(yolo_label_dir, label)
    dst_label = os.path.join(yolo_label_val_dir, label)
    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)

# Step 4: Create YOLOv8 Config YAML with absolute path
yolo_abs_path = os.path.abspath(yolo_base).replace('\\', '/')  # For Windows compatibility
with open('pcb.yaml', 'w') as f:
    f.write(f"""
path: {yolo_abs_path}
train: images/train
val: images/val
nc: 6
names: ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
""")

# Step 5: Train YOLOv8 Model or Load if Already Trained
save_dir = Path("runs/detect/train5")
weights_path = save_dir / "weights/best.pt"

if weights_path.exists():
    print(f"✅ Model already trained. Loading from {weights_path}")
    model = YOLO(weights_path)
else:
    print("🚀 Training model from scratch...")
    model = YOLO('yolov8n.pt')  # Use YOLOv8n for speed
    model.train(data=os.path.abspath('pcb.yaml'), epochs=20, imgsz=640)

# Step 6: Predict on a Local Test Image
test_img_path = 'test2.jpg'  
results = model(test_img_path)
results[0].save(filename='result.jpg')

# Display the result image 
img = PILImage.open('result.jpg')
img.show()
