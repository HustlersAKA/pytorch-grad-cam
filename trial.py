import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os
import glob
import shutil
import math 
from PIL import Image
from IPython.display import Image as im
from IPython.display import display as dis


def pil_to_np(img):
  img = img.convert("RGB")
  img = np.asarray(img, dtype=np.float32) / 255
  img = img[:, :, :3]
  return img


def integerize(num):
  return int(math.floor(num))  


def pre_detection_preprocessing(img , max_dim = 1000 ): 
  
  #INPUT : original image of type numpy.ndarray
  #OUTPUT : preprocessed img for logo detection , vertical boundary in pixels , horizontal boundary in pixels , rescaled dimentions 

  dims = img.shape[:2]
  print("original image shape : {}".format(dims))
  new_dims = [0] * 2
  new_dims[dims.index(max(dims))] = max_dim
  new_dims[dims.index(min(dims))] = math.ceil(( min(dims) / max(dims) ) *max_dim)
  if dims[0] == dims[1]:
    new_dims = (max_dim, max_dim)
  print(f"errror tuple {new_dims}")
  resized_img = cv2.resize(img , tuple(new_dims[::-1]))
  new_img = np.zeros([max_dim , max_dim , 3])
  (rows , cols) = resized_img.shape[:2]
  ar , ac = (max_dim - rows) // 2 , (max_dim - cols) // 2
  new_img[ ar : ( rows + ar ) , ac : ( cols + ac ) , :] = resized_img
  #new_img= new_img[:, :, ::-1] 
  return new_img , ar , ac , new_dims   


def push_box(c1,c2):
  (xmin,ymin),(xmax,ymax) = c1,c2    
  if ymax <0:
    diff = 0 - ymax
    ymin += diff
  if xmin <0:
    diff = 0 - xmin
    xmax += diff 
  return (xmin,ymin), (xmax, ymax)     

def draw_boxes(original_img , processed_img, preds, new_dims ,output , borders):
  
  colors = {
      0 : [(0,0,0), (256,256,256)],       
      1 : [(256,256,256), (0,0,0)]   
    }

  class_names = {
      0 : "head",
      1 : "hard hat"
  }  

  ar , ac = borders  
  
  orig_shape = original_img.shape[:2]
  rescaled_shape = new_dims   #scaled down shape
  row_scaling = orig_shape[0] / rescaled_shape[0]
  col_scaling = orig_shape[1] / rescaled_shape[1]
  for i in range(len(preds)):
    x1 , y1 , x2 , y2 , conf , class_no = preds.iloc[i,:6]
    x1 = round((x1 - ac) * row_scaling) 
    x2 = round((x2 - ac) * row_scaling) 
    y1 = round((y1 - ar) * col_scaling) 
    y2 = round((y2 - ar)* col_scaling)
    tl = 3
    c1, c2 = (x1,y1) , (x2,y2)
    cv2.rectangle(output , (x1,y1) , (x2,y2) , color = colors[class_no][0] , thickness = 3)
    label = str(round(conf*100)/100)+ f" {class_names[class_no]}"
    tf = max(tl - 1, 1) 
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 7, thickness=tf)[0]  
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    c1,c2 = push_box(c1,c2)
    cv2.rectangle(output, c1, c2, colors[class_no][0], -1, cv2.LINE_AA)  # filled
    cv2.putText(output, label, (c1[0], c1[1] - 2), 0, tl / 8,color = colors[class_no][1], thickness= 2, lineType=cv2.LINE_AA)
  return output
