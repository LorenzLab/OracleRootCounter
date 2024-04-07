from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train7/weights/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['datasets/OracleDS/images/00000d32fae14ba688bfd99080c6e0b7.png', 'datasets/OracleDS/images/000076231ba345e8a14060b72808de6c.png'])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk
