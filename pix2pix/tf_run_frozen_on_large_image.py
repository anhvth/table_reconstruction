from time import time
import tensorflow as tf
import cv2
import numpy as np
import os
from glob import glob
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--export', default=False, type=bool)
parser.add_argument('--checkpoint', default='pix2pix/output/128x1024_v3')
parser.add_argument('--strides', default=[64, 256], type=int, help='the strides for exported model')
parser.add_argument('--input_dir', default='data/cell_data',
                    help='directory to input images')
parser.add_argument(
    '--output_dir', default='output/run_method2', help='output image')
args = parser.parse_args()

def get_tensor_by_name(name):
    name_on_device = '{}:0'.format(name)
    return tf.get_default_graph().get_tensor_by_name(name_on_device)

def load_image(path, verbal=False):
    # output 3-d image RGB
    print('---------------------------\nprocess:', path)
    name = path.split('/')[-1].split('.')[0]
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def write(path, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(path, image)
if __name__ == '__main__':
    frozen_checkpoint = os.path.join(args.checkpoint, 'frozen') 
    if args.export == True or os.path.exists(frozen_checkpoint)==False:
        print('\n\n{} DOES NOT EXIST\n\n'.format(frozen_checkpoint))
        assert os.path.exists(args.checkpoint), args.checkpoint+' does not exist'
        cmd_line = '''python pix2pix/pix2pix_deep_unet.py  \
                        --mode export \
                        --checkpoint {}\
                        --output_dir {}'''\
                        .format(args.checkpoint,  frozen_checkpoint)
        print(cmd_line)
        os.system(cmd_line)


    meta_path = os.path.join(frozen_checkpoint, 'export.meta')
    print('meta path:', meta_path)

    assert os.path.exists(meta_path), meta_path+' does not exist'

    tf.train.import_meta_graph(meta_path)
    sess = tf.Session()
    saver = tf.train.Saver()

    saver.restore(sess, tf.train.latest_checkpoint(
        frozen_checkpoint))

    os.makedirs(args.output_dir, exist_ok=True)
    start = time()
    paths = glob('{}/*.png'.format(args.input_dir))
    paths = [path for path in paths]
    assert len(paths) > 0
    print('Num of sample:', len(paths), args.input_dir)
    inputs = get_tensor_by_name('inputs')
    outputs = get_tensor_by_name('outputs')
    for path in paths:
        name = path.split('/')[-1].split('.')[0]
        image = load_image(path, verbal=True)
        output_image = sess.run(outputs, {inputs: image})
        merge_image = 0.5*image+0.5*output_image
        # os.makedirs('output/{}_{}'.format(args.stride, name), exist_ok=True)
        print('{}/{}_input.png'.format(args.output_dir, name))
        write('{}/{}_input.png'.format(args.output_dir, name), image)
        write('{}/{}_output.png'.format(args.output_dir, name), output_image)
        write('{}/{}_merge.png'.format(args.output_dir, name), merge_image)
    print('Running time:', time()-start)
