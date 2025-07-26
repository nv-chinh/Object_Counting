import os
import cv2
images_folder = 'Shrimp_Counting_Dataset/test/images'
labels_folder = 'Shrimp_Counting_Dataset/test/labels'

min_width = 1e6
min_height = 1e6
max_width = 0
max_height = 0
for image_name in os.listdir(images_folder):
    image = cv2.imread(os.path.join(images_folder, image_name))
    height, width, _ = image.shape
    with open(os.path.join(labels_folder, image_name[:-4]+'.txt'),'r') as f:
        labels = f.read().splitlines()
    for label in labels:
        _,x,y,w,h = list(map(float, label.split(' ')))
        xmin = int((x-w/2)*width)
        xmax = int((x+w/2)*width)
        ymin = int((y-h/2)*height)
        ymax = int((y+h/2)*height)
        obj_width = xmax - xmin
        obj_height = ymax - ymin
        if obj_width < min_width:
            min_width = obj_width
        if obj_height < min_height:
            min_height = obj_height
        if obj_width > max_width:
            max_width = obj_width
        if obj_height > max_height:
            max_height = obj_height
print(min_width)
print(max_width)
print(min_height)
print(max_height)