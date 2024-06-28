import glob
import os
from shutil import move
from os import rmdir

target_folder = 'tiny-imagenet-200/val/'

val_dict = {}
with open('tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]
        
paths = glob.glob('tiny-imagenet-200/val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        os.mkdir(target_folder + str(folder) + '/images')
       
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    dest = target_folder + str(folder) + '/images/' + str(file)
    move(path, dest)
    
rmdir('tiny-imagenet-200/val/images')

# for cls in os.listdir(target_folder):
#     clspath = os.path.join(target_folder, cls)
#     rm = os.path.join(clspath, cls+"_boxes.txt")
#     if os.path.exists(rm):
#         os.remove(rm)
#     for img in os.listdir(os.path.join(clspath, "images")):
#         move(os.path.join(clspath, "images", img), clspath)
#     rmdir(os.path.join(clspath, "images"))