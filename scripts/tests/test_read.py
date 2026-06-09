import torchvision.io as io
import cv2

path = '/home/adam/workspace/RDNU/GameIR/mini_dataset/train/GameIR-SR/GameIR-SR/static_town08/19/720p/00000050.depth.png'
img_tensor = io.read_image(path)
print("torchvision shape:", img_tensor.shape, img_tensor.dtype)

img_cv2 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
print("cv2 shape:", img_cv2.shape, img_cv2.dtype)
