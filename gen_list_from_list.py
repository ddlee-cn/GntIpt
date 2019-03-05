"""
generate train file list from officially provided list(with label)

# head standard/categories_places365.txt
/a/airfield 0
/a/airplane_cabin 1
/a/airport_terminal 2

# head toy/places365_train_standard.txt
/d/discotheque/00003105.jpg 122
/a/art_school/00002916.jpg 20
/b/bus_station/indoor/00004835.jpg 71

# head toy/places365_val.txt
Places365_val_00020696.jpg 156
Places365_val_00029894.jpg 188
Places365_val_00006556.jpg 175

/standard# head places365_test.txt
Places365_test_00000001.jpg
Places365_test_00000002.jpg
Places365_test_00000003.jpg

"""

import argparse
from pathlib import Path
import numpy as np
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--list_file', default='', type=str,
                    help='list file')
parser.add_argument('--dataset_root', type=str, help='dataset root directory')
parser.add_argument('--output', type=str, default='', help='list file save path')


if __name__ == "__main__":
    args = parser.parse_args()
    
    # ignore label
    img_filename_lst = list(np.genfromtxt(args.list_file, dtype=np.str, encoding='utf-8')[:, 0])
    
    output_path = Path(args.output)
    if not output_path.parent.exists():
        output_path.parent.mkdir()

    if not output_path.exists():
        output_path.touch()

    # pdb.set_trace()

    root = Path(args.dataset_root)

    with output_path.open(mode='w') as f:
        for img_file in img_filename_lst:
            # Places2 have /m/mansion/00002450.jpg style, make it as child dir
            # BUG: don't work for val and test images!
            f.write(str(root.joinpath('.' + img_file)) + '\n')
        
    print("list file saved at {}".format(output_path))
