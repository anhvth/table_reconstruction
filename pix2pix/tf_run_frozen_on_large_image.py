from time import time
import tensorflow as tf
import cv2
import numpy as np
import os
from glob import glob
import argparse
from tool.split_join import split_patches, join_patches


parser = argparse.ArgumentParser()
parser.add_argument('--strides', default=[128, 256], type=int,help='')
parser.add_argument('--k_size', default=[256,512], type=int, help='')
parser.add_argument('--checkpoint', default='pix2pix/output/frozen/128x1024')
parser.add_argument('--input_dir', default='data/cell_data', help='directory to input images')
parser.add_argument('--output_dir', default='output/run_method2', help='output image')
parser.add_argument('--method', type=str, default='method2', help='method')
args = parser.parse_args()

meta_path = '{}/export.meta'.format(args.checkpoint)
print('meta path:', meta_path)
assert os.path.exists(meta_path)

tf.train.import_meta_graph(meta_path)
sess = tf.Session()
saver = tf.train.Saver()

saver.restore(sess, tf.train.latest_checkpoint(
    args.checkpoint))


def split(image, ksizes, strides):
    images = tf.extract_image_patches(image, ksizes=ksizes, strides=strides, padding='SAME', rates=[1,1,1,1])
    new_shape = [-1, *images.get_shape().as_list()[1:3], *ksizes[1:3], image.get_shape().as_list()[-1]]
    images = tf.reshape(images, new_shape)
    return images


def get_tensor_by_name(name):
    name_on_device = '{}:0'.format(name)
    return tf.get_default_graph().get_tensor_by_name(name_on_device)


def resize(image):
    h, w = image.shape[:2]
    new_h = math.ceil(h/args.strides[0])*agrs.strides[0]
    new_w = math.ceil(w/args.strides[1])*agrs.strides[1]
    return cv2.resize(image, (w, h))



def run_image(image):
    h, w = image.shape[:2]
    resized_image = resize(image)
    splited_images = split_patches(resized_image, args.k_size, agrs.strides)

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
        assert args.method is not None
        if args.method=='method1':
            output_image = run_image(image)
        elif args.method=='method2':
            output_image = run_large_image(image)
        else:
            assert False
            
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        merge_image = 0.5*image+0.5*output_image

        # os.makedirs('output/{}_{}'.format(args.stride, name), exist_ok=True)
        cv2.imwrite('{}/{}_{}_input.png'.format(args.output_dir,args.stride, name), image)
        cv2.imwrite('{}/{}_{}_output.png'.format(args.output_dir, args.stride, name), output_image)
        cv2.imwrite('{}/{}_{}_merge.png'.format(args.output_dir, args.stride, name), merge_image)
    print('Running time:', time()-start)