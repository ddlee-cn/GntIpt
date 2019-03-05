# - model_logs
# - neuralgym_logs
# - training_data
#   -- training
#     --- <folder1>
#     --- <folder2>
#     --- .....
#   -- validation
#     --- <val_folder1>
#     --- <val_folder2>
#     --- .....
# - <this_file.py>
# https://github.com/JiahuiYu/generative_inpainting/issues/15


# Places2
# /userhome/data/places2/original/Small-Images/data_256/a/arena/*.jpg
#                                               /val_256/*.jpg
# train_sub = a
# train_subub = arena
# non-definite sub folders

import argparse
import os
import random
from pathlib import Path

train_origin = '/userhome/data/places2/original/Small-images/data_256/'
val_origin = '/userhome/data/places2/original/Small-images/val_256/'

train_list = '/userhome/data/places2/train.lst'
val_list = '/userhome/data/places2/val.lst'

seed = 2019

if __name__ == "__main__":
    random.seed(seed)

    # # get the list of directories and separate them into 2 types: training and validation
    # val_dir = Path(val_origin)

    # # append all files into 2 lists
    # validation_file_names = [str(name) for name in val_dir.rglob('*.jpg')]


    # # print all file paths
    # print("Val: {}".format(len(validation_file_names)))

    # # shuffle file names if set
    # random.shuffle(validation_file_names)
    # if not os.path.exists(val_list):
    #     os.mknod(val_list)

    # fo = open(val_list, "w")
    # fo.write("\n".join(validation_file_names))
    # fo.close()

    # # print process
    # print("Written file is: ", val_list)

    # Train
    train_dir = Path(train_origin)
    training_file_names = []
    cnt = 0
    for name in train_dir.rglob('*.jpg'):
        training_file_names.append(str(name))
        cnt += 1
        if cnt%1000==0:
            print("{} founded.".format(cnt))

    print("Train: {}".format(len(training_file_names)))
    random.shuffle(training_file_names)
    # make output file if not existed

    if not os.path.exists(train_list):
        os.mknod(train_list)

    # write to file
    fo = open(train_list, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    # print process
    print("Written file is: ", train_list)