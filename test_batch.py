import argparse

import cv2
import os
import numpy as np
import tensorflow as tf
import neuralgym as ng
from pathlib import Path
from inpaint_model import InpaintCAModel
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--image_list', default='', type=str,
                    help='The filenames of image to be completed.')
parser.add_argument('--mask_list', default='', type=str,
                    help='The filenames of mask, value 255 indicates mask.')
parser.add_argument('--output_dir', default='./output', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--demo', default=False, type=bool,
                    help='demo for comparison')

if __name__ == "__main__":
    ng.get_gpus(1, dedicated=False)
    args = parser.parse_args()

    image_list = np.genfromtxt(args.image_list, dtype=np.str, encoding='utf-8')
    mask_list = np.genfromtxt(args.mask_list, dtype=np.str, encoding='utf-8')

    model = InpaintCAModel()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    
    def build_graph(height, width):
        x = tf.placeholder(tf.float32, shape=(1, height, width, 3))

        output = model.build_server_graph(x, reuse=tf.AUTO_REUSE)            
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        return x, output

    # read the first image to init graph shape
    image = cv2.imread(image_list[-2])
    mask = cv2.imread(mask_list[-2])
    h_old, w_old, _ = image.shape
    grid = 8
    image = image[:h_old//grid*grid, :w_old//grid*grid, :]
    mask = mask[:h_old//grid*grid, :w_old//grid*grid, :]
    # grided image shape
    h_old_grid, w_old_grid, _ = image.shape
    x, output = build_graph(h_old_grid, w_old_grid*2)

    with tf.Session(config=sess_config) as sess:
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print("Model loaded.")

        for image_file, mask_file in zip(image_list, mask_list):
            image = cv2.imread(image_file)
            mask = cv2.imread(mask_file)
            mask = 255-mask

            assert image.shape == mask.shape

            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            h_grid, w_grid, _ = image.shape
            # print(Path(mask_file).name)                
            print('Shape of image: {}'.format(image.shape))

            # rebuild graph when resolution changed
            if h != h_old or w != w_old:
                x, output = build_graph(h_grid, w_grid*2)
                h_old, w_old = h, w

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)
            # print(input_image.shape)

            result = sess.run(output, {x: input_image})
            # pdb.set_trace()

            # write output
            save_path = str(Path(args.output_dir).joinpath(Path(mask_file).name))

            # print(image.shape, mask.shape, result[0][:, :, ::-1].shape)
            # pdb.set_trace()
            if args.demo:
                mask_mult = (255-mask[0])
                ori_image = image.copy()
                image[0][mask_mult==0] = 255
                # choose concat dimension
                if image.shape[1] < image.shape[2]:
                    final = np.concatenate([image[0], result[0][:, :, ::-1], ori_image[0]], axis=0)
                else:
                    final = np.concatenate([image[0], result[0][:, :, ::-1], ori_image[0]], axis=1)
            else:
                final = result[0][:, :, ::-1]
            cv2.imwrite(save_path, final)
            print("tested: {}".format(save_path))
