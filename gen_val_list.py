"""
Generate image and mask pairs for ICME data

# valid_img/error_concealment_valid/61/61_16.jpg
# valid_img/object_removal_valid/61/61_16.jpg
# valid_img/valid_img/63.jpg

"""
from pathlib import Path
import random
import cv2
import pdb
import sys


# Def
data_root_path = "/userhome/data/ICME2019_Image_Inpainting/"

ecv_path = "valid_img/error_concealment_valid/"
orv_path = "valid_img/object_removal_valid/"
gt_path = "valid_img/valid_img/"

ec_pair_file = "./list/ec_pair.lst"
ec_gt_file = "./list/ec_gt.lst"
ec_mask_file = "./list/ec_mask.lst"
or_pair_file = "./list/or_pair.lst"
or_gt_file = "./list/or_gt.lst"
or_mask_file = "./list/or_mask.lst"

toy_gt_file = "./list/toy_gt.lst"
toy_mask_file = "./list/toy_mask.lst"

# Error Concealment Mask
ecv = Path(data_root_path).joinpath(ecv_path).resolve()
ecv_list = [i for i in ecv.rglob("*.jpg")] # 4900
# print("ecv", len(ecv_list))

# Object Removal Mask
orv = Path(data_root_path).joinpath(orv_path).resolve()
orv_list = [i for i in orv.rglob("*.jpg")] # 5600

# Ground Truth
gt = Path(data_root_path).joinpath(gt_path).resolve()
gt_list = [i for i in gt.rglob("*.jpg")] # 70
print("gt", len(gt_list))

# stat resolution
stat = dict()
# pdb.set_trace()
for gt in gt_list:
    img = cv2.imread(str(gt))
    res = '-'.join([str(i) for i in img.shape])
    if res not in stat.keys():
        stat[res] = 1
    else:
        stat[res] += 1

for res, cnt in stat.items():
    print(res, cnt)

sys.exit()


# gen EC track pair
ec_pair = []
for gt in gt_list:
    # print(gt)
    img_id = gt.name.replace(".jpg", "")
    ec_mask_path = ecv.joinpath(img_id)
    ec_pair += [(gt, i) for i in ec_mask_path.glob("*.jpg")]
    # print(len(ec_pair))

print(len(ec_pair), ec_pair[0]) # 4900

# gen EC track pair
or_pair = []
for gt in gt_list:
    # print(gt)
    img_id = gt.name.replace(".jpg", "")
    or_mask_path = orv.joinpath(img_id)
    or_pair += [(gt, i) for i in or_mask_path.glob("*.jpg")]
    # print(len(ec_pair))

print(len(or_pair), or_pair[0]) # 5600

# write toy list
n = 20
toy_ec_pair = random.choices(ec_pair, k=n)
toy_or_pair = random.choices(or_pair, k=n)

with open(toy_gt_file, 'w') as f:
    for i, j in zip(toy_ec_pair, toy_or_pair):
        f.write(str(i[0]) + '\n')
        f.write(str(j[0]) + '\n')

with open(toy_mask_file, 'w') as f:
    for i, j in zip(toy_ec_pair, toy_or_pair):
        f.write(str(i[1]) + '\n')
        f.write(str(j[1]) + '\n')
# write pairs
with open(ec_pair_file, 'w') as f:
    for pair in ec_pair:
        f.write(','.join([str(i) for i in pair]) + '\n')

with open(or_pair_file, 'w') as f:
    for pair in or_pair:
        f.write(','.join([str(i) for i in pair]) + '\n')
        
# write seperate list for gt and mask
with open(ec_gt_file, 'w') as f:
    for pair in ec_pair:
        f.write(str(pair[0]) + '\n')
        
with open(ec_mask_file, 'w') as f:
    for pair in ec_pair:
        f.write(str(pair[1]) + '\n')
        
with open(or_gt_file, 'w') as f:
    for pair in or_pair:
        f.write(str(pair[0]) + '\n')
        
with open(or_mask_file, 'w') as f:
    for pair in or_pair:
        f.write(str(pair[1]) + '\n')

