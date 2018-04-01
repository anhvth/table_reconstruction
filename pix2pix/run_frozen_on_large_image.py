import tensorflow as tf
import cv2
import numpy as np
import os
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stride', default=32, type=int,
                    help='stride when spliting the whole page image')
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


def get_tensor_by_name(name):
    name_on_device = '{}:0'.format(name)
    return tf.get_default_graph().get_tensor_by_name(name_on_device)


def resize(image):
    h, w = image.shape[:2]
    px = 1024 / w
    py = 128 * (h//128+1) / h
    return cv2.resize(image, (0, 0), fx=px, fy=py)

def pad(image):
    h, w = image.shape[:2]
    new_h = path.ceil(h/128)*128
    new_w = path.ceil(w/1024)*1024
    pad = np.zeros([new_h, new_w, 3])
    start_h = (new_h-h) // 2
    start_w = (new_w-w) // 2
    pad[:,:,:] = 255
    pad[start_h:start_h+h, start_w:start_w+w] = image
    return pad

def depad(image):
    pass

def split_image(image):
    image_resized = resize(image)
    splited_images = np.split(image_resized, image_resized.shape[0]//128)
    return np.array(splited_images)


def split_image_overlap(image, height=128, width=1024):
    rv = []
    for i in range(0, image.shape[0], args.stride):
        
        pad = image[i:i+height]
        if pad.shape[0] == height:
            rv.append(pad)
    return np.array(rv)

def join_image_overlap(images):
    to_concat = [images[0][:args.stride+args.stride//2]]
    for i, image in enumerate(images[1:-1]):
        roi = image[args.stride//2:args.stride//2+args.stride]
        if False:  # i % 2 == 0:
            roi = 255 - roi
        to_concat.append(roi)
    to_concat.append(images[-1][args.stride//2:])
    return np.concatenate(to_concat)

def run_image(image):
    h, w = image.shape[:2]
    resized_image = resize(image)
    # splited_images = split_image(resized_image)
    splited_images = split_image_overlap(resized_image)

    output_images = sess.run(outputs, feed_dict={inputs: splited_images})
    # np.concatenate(output_images)
    output_image = join_image_overlap(output_images)
    output_image = cv2.resize(output_image, (w, h))
    return output_image

def run_large_image(large_image, min_w = 800, max_w = 1600):
    h, w,c = large_image.shape
    print(w)
    if w < min_w:
        pad = np.zeros([h, 1024, c])
        pad[:h,:w] = large_image
        large_image = pad
        output = run_image(large_image)
        return output[:h,:w]    
    if w > max_w:
        im1, im2 = large_image[:,:w//2], large_image[:,w//2:]
        rt_im1, rt_im2 = run_image(im1), run_image(im2)
        output = np.concatenate([rt_im1, rt_im2], axis=1)
        return output
    return run_image(large_image)
if __name__ == '__main__':
    from time import time
    start = time()
    os.makedirs(args.output_dir, exist_ok=True)
    
    paths = glob('{}/*.png'.format(args.input_dir))[:20]
    paths = [path for path in paths]
    assert len(paths) > 0
    print('num of sample:', len(paths), args.input_dir)
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
        # cv2.imwrite('output/{}_{}_input.png'.format(args.stride, name), image)
        # cv2.imwrite('output/{}_{}_output.png'.format(args.stride, name), output_image)
        cv2.imwrite('{}/{}_{}_merge.png'.format(args.output_dir, args.stride, name), merge_image)
    print('Running time:', time()-start)