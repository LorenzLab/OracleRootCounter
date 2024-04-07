import os
import cv2


img_dir = "LOBI_Words"
out_img_dir = "LOBI_Words_inv"
imgs = os.listdir(img_dir)

for img_path in imgs:
    img = cv2.imread(os.path.join(img_dir, img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = 255 - img
    cv2.imwrite(os.path.join(out_img_dir, img_path), img)

