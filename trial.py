import numpy as np
import matplotlib.pyplot as plt

def pil_to_np(img):
  img = img.convert("RGB")
  img = np.asarray(img, dtype=np.float32) / 255
  img = img[:, :, :3]
  return img