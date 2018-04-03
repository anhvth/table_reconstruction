from time import time
import tensorflow as tf
import cv2
import numpy as np
import os
from glob import glob
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--strides', default=[128, 256], type=int,help='')
parser.add_argument('--k_size', default=[256,512], type=int, help='')
parser.add_argument('--checkpoint', default='pix2pix/output/frozen/128x1024')
parser.add_argument('--input_dir', default='data/cell_data', help='directory to input images')
parser.add_argument('--output_dir', default='output/run_method2', help='output image')
args = parser.parse_args()

meta_path = '{}/export.meta'.format(args.checkpoint)
print('meta path:', meta_path)
assert os.path.exists(meta_path)

tf.train.import_meta_graph(meta_path)
sess = tf.Session()
saver = tf.train.Saver()

saver.restore(sess, tf.train.latest_checkpoint(
    args.checkpoint))





def get_tensor_by_name(name):
    name_on_device = '{}:0'.format(name)
    return tf.get_default_graph().get_tensor_by_name(name_on_device)


def resize(image):
    h, w = image.shape[:2]
    new_h = math.ceil(h/args.strides[0])*args.strides[0]
    new_w = math.ceil(w/args.strides[1])*args.strides[1]
    return cv2.resize(image, (w, h))

def extract_patches(image, k_size, strides):
    images = tf.extract_image_patches(tf.expand_dims(
        image, 0), k_size, strides, rates=[1, 1, 1, 1], padding='SAME')[0]
    images_shape = tf.shape(images)
    images_reshape = tf.reshape(
        images, [images_shape[0]*images_shape[1], *k_size[1:3], 3])
    images, n1, n2 = tf.cast(images_reshape, tf.uint8) , images_shape[0], images_shape[1]
    return images, n1, n2

def join_patches(images, n1, n2, k_size, strides):

    s1 = k_size[1]//2-strides[1]//2
    s2 = k_size[2]//2-strides[2]//2
    roi = images[:, 
                 s1:s1+strides[1],\
                 s2:s2+strides[2],
                 :]
    new_shape = [n1, n2, *roi.get_shape().as_list()[1:]]
    reshaped_roi = tf.reshape(roi, new_shape)
    reshaped_roi = tf.transpose(reshaped_roi, perm=[0,2,1,3,4])
    rs = tf.shape(reshaped_roi)
    rv = tf.cast(tf.reshape(reshaped_roi, [rs[0]*rs[1], rs[2]*rs[3], -1]), tf.uint8)
    return rv

def run_image(image):
    h, w = image.shape[:2]
    resized_image = resize(image)
    splited_images = extract_patches(image, [1,*args.k_size,1], [1,*args.strides,1]):
    output_images = sess.run(outputs, feed_dict={inputs: splited_images})

    output_image = join_patches(output_images)
    output_image = cv2.resize(output_image, (w, h))

    return output_image


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    start = time()
    paths = glob('{}/*.png'.format(args.input_dir))
    paths = [path for path in paths]
    assert len(paths) > 0
    print('Num of sample:', len(paths), args.input_dir)
    for path in paths:
        print(path)
        name = path.split('/')[-1].split('.')[0]
        image = cv2.imread(path, 0)
        image = np.stack([image]*3, axis=2)
        inputs = get_tensor_by_name('inputs')
        outputs = get_tensor_by_name('outputs')
        output_image = run_image(image)

            
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        merge_image = 0.5*image+0.5*output_image

        # os.makedirs('output/{}_{}'.format(args.stride, name), exist_ok=True)
        cv2.imwrite('{}/{}_{}_input.png'.format(args.output_dir,args.stride, name), image)
        cv2.imwrite('{}/{}_{}_output.png'.format(args.output_dir, args.stride, name), output_image)
        cv2.imwrite('{}/{}_{}_merge.png'.format(args.output_dir, args.stride, name), merge_image)
    print('Running time:', time()-start)