from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True,
                    choices=["train", "test", "export"])
parser.add_argument("--output_dir",
                    required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int,
                    help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=200,
                    help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50,
                    help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0,
                    help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000,
                    help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", default=True, type=bool,
                    help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0,
                    help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true",
                    help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1,
                    help="number of images in batch")
parser.add_argument("--which_direction", type=str,
                    default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64,
                    help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64,
                    help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=[
                    256, 512], help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true",
                    help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip",
                    action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002,
                    help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5,
                    help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0,
                    help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0,
                    help="weight on GAN term for generator gradient")
parser.add_argument("--strides", default=[32, 128], type=int,
                    help="export strides to run on large image")

# export options
parser.add_argument("--output_filetype", default="png",
                    choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = [256, 512]

Examples = collections.namedtuple(
    "Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple(
    "Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        print('image:', image)
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [
                          1, 1], [0, 0]], mode="CONSTANT")
    # num_filter = batch_input.get_shape().as_list()[-1]
    # filters = tf.Variable(tf.random_normal([4, 4, num_filter, out_channels]))
    # batch_input2 = tf.nn.atrous_conv2d(padded_input, filters, 3, padding='SAME', name='dilated_conv_generator')
    # concat_input = tf.concat([padded_input, batch_input2], axis=-1)
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels, stride):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    num_filter = batch_input.get_shape().as_list()[-1]
    if False:  # a.separable_conv:
        filters = tf.Variable(tf.random_normal([4, 4, num_filter, num_filter]))
        batch_input2 = tf.nn.atrous_conv2d(
            batch_input, filters, 3, padding='SAME', name='dilated_conv_generator')
        g1 = tf.layers.separable_conv2d(batch_input, out_channels//2, kernel_size=4, strides=(stride, 2),
                                        padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
        g2 = tf.layers.separable_conv2d(batch_input2, out_channels//2, kernel_size=4, strides=(stride, 2),
                                        padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)

        return tf.concat([g1, g2], axis=-1, name='fuse')
    else:
        filters = tf.Variable(tf.random_normal(
            [4, 4, num_filter, num_filter], mean=0, stddev=0.02))
        batch_input2 = tf.nn.atrous_conv2d(
            batch_input, filters, 3, padding='SAME', name='dilated_conv_generator')
        g1 = tf.layers.conv2d(batch_input, out_channels//2, kernel_size=4,
                              strides=(stride, 2), padding="same", kernel_initializer=initializer)
        g2 = tf.layers.conv2d(batch_input2, out_channels-out_channels//2, kernel_size=4,
                              strides=(stride, 2), padding="same", kernel_initializer=initializer)
        return tf.concat([g1, g2], axis=-1, name='fuse')


def gen_deconv(batch_input, out_channels, stride):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    num_filter = batch_input.get_shape().as_list()[-1]
    if False:  # a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(
            batch_input, [h * 2, w * stride], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        filters = tf.Variable(tf.random_normal(
            [4, 4, num_filter, out_channels//2]))
        g1 = tf.layers.separable_conv2d(resized_input, out_channels//2, kernel_size=4, strides=(
            1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
        _, h, w, _ = g1.get_shape().as_list()
        # g2 = tf.nn.atrous_conv2d_transpose(resized_input, filters, rate=3, padding='SAME', output_shape=[-1, h, w, -1])
        return g1  # tf.concat([g1], axis=-1, name='fuse')
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(stride, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a=.2):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2
        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x \
            + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True if a.mode == 'train' else False, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    assertion = tf.assert_equal(
        tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(
            input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(
            tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        width = tf.shape(raw_input)[1]  # [height, width, channels]
        a_images = preprocess(raw_input[:, :width//2, :])
        b_images = preprocess(raw_input[:, width//2:, :])

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)

    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        # new_h = tf.cast(a.scale_size[0]*tf.Variable(tf.random_uniform([], minval=0.75, maxval=1.5)), tf.int32)
        # new_w = tf.cast(a.scale_size[1]*tf.Variable(tf.random_uniform([], minval=0.75, maxval=1.5)), tf.int32)

        r = tf.image.resize_images(
            r, a.scale_size, method=tf.image.ResizeMethod.AREA)
        # offset1 = tf.cast(tf.floor(tf.random_uniform([1], 0, a.scale_size[0] - CROP_SIZE[0] + 1, seed=seed)), dtype=tf.int32)
        offset1 = tf.cast(tf.floor(tf.random_uniform(
            [], 0, a.scale_size[0] - CROP_SIZE[0] + 1, seed=seed)), dtype=tf.int32)
        offset2 = tf.cast(tf.floor(tf.random_uniform(
            [], 0, a.scale_size[1] - CROP_SIZE[1] + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size[0] > CROP_SIZE[0]:
            r = tf.image.crop_to_bounding_box(
                r, offset1, offset2, CROP_SIZE[0], CROP_SIZE[1])
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch(
        [paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


class downblock:
    def __init__(self, x, filters1=a.ngf, filters2=a.ngf//2, k_size=[3, 3], use_batch_norm=True, fisrt_block=False, name=None):
        with tf.variable_scope(name):
            self.x = x
            if fisrt_block:
                self.conv1 = tf.layers.conv2d(
                    x, filters1, k_size, padding='same', name='conv1')
            else:

                self.conv1 = tf.layers.conv2d(
                    lrelu(x), filters1, k_size, padding='same', name='conv1')
                self.conv1 = batchnorm(
                    self.conv1) if use_batch_norm else self.conv1

            self.conv2 = tf.layers.conv2d(
                lrelu(self.conv1), filters2, k_size, padding='same', name='conv2')
            self.conv2 = batchnorm(
                self.conv2) if use_batch_norm else self.conv2

            # pad if input shape and conv2 shape is not the samconv1e
            n_layer_diff = self.conv2.get_shape().as_list(
            )[-1] - self.x.get_shape().as_list()[-1]
            if n_layer_diff != 0:
                pooled_input = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                                              strides=[1, 1, 1, 1], padding='SAME')

                padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [n_layer_diff // 2,
                                                                              n_layer_diff - n_layer_diff // 2]])
            else:
                padded_input = x
            self.plus = padded_input + self.conv2
            self.y = self.plus
            out_filter = self.plus.get_shape().as_list()[-1]
            self.downsampling_output = tf.layers.conv2d(
                self.plus, out_filter, [3, 3], strides=2, padding='same')
            print('{}--> {}'.format(self.x.shape, self.downsampling_output.shape))


class upblock:
    def __init__(self, x1, x2, filters1=a.ngf, filters2=a.ngf//2, k_size=[3, 3], use_batch_norm=True, first_layer=False, name=None):
        with tf.variable_scope(name):
            self.x1 = x1
            self.x2 = x2
            self.x = tf.concat([x1, x2], axis=-1) if not first_layer else x1

            self.conv1 = tf.layers.conv2d(tf.nn.relu(
                self.x), filters1, k_size, padding='same', name='conv1')
            self.conv1 = batchnorm(
                self.conv1) if use_batch_norm else self.conv1

            self.conv2 = tf.layers.conv2d(tf.nn.relu(
                self.conv1),  filters2, k_size, padding='same', name='conv2')
            self.conv2 = batchnorm(
                self.conv2) if use_batch_norm else self.conv2

            # pad if input shape and conv2 shape is not the samconv1e
            n_layer_diff = self.conv2.get_shape().as_list(
            )[-1] - self.x.get_shape().as_list()[-1]

            self.plus = self.x1 + self.conv2

            self.y = self.plus
            out_filter = self.plus.get_shape().as_list()[-1]
            self.upsampling_output = tf.layers.conv2d_transpose(
                self.plus, out_filter, [3, 3], strides=2, padding='same')
            if not first_layer:
                print('{}+{}--> {}'.format(self.x1.shape,
                                           self.x2.shape, self.upsampling_output.shape))
            else:
                print('{}+[**] ---> {}'.format(self.x1.shape,
                                               self.upsampling_output.shape))


def create_generator(generator_inputs, generator_outputs_channels):
    db1 = downblock(generator_inputs, fisrt_block=True,
                    name='db1')  # 128x512->64x256
    db2 = downblock(db1.downsampling_output, name='db2')  # 64x256->32x128
    db3 = downblock(db2.downsampling_output, name='db3')  # 32x128->16x64
    db4 = downblock(db3.downsampling_output, name='db4')  # 16x64->8x32
    db5 = downblock(db4.downsampling_output, name='db5')  # 8x32->4x16

    up5 = upblock(db5.downsampling_output, None,
                  first_layer=True, name='up5')  # 4x16->8x32
    up4 = upblock(up5.upsampling_output, db5.y, name='up4')  # 8x32
    up3 = upblock(up4.upsampling_output, db4.y, name='up3')  # 16x64
    up2 = upblock(up3.upsampling_output, db3.y, name='up2')  # 32x128
    up1 = upblock(up2.upsampling_output, db2.y,
                  name='up1')  # 64x256->128x512x32

    input = up1.upsampling_output  # 128x512 ->128x512
    rectified = tf.nn.relu(input)
    logits = tf.layers.conv2d_transpose(rectified, 3, 1, padding='same')
    outputs = tf.tanh(logits)
    print('generator output: ', outputs.shape)
    return outputs


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(
                    layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) +
                                        tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables(
        ) if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(
            discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables(
            ) if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(
                gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )

def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        assert a.strides is not None

        def extract_patches(image, k_size, strides):
            images = tf.extract_image_patches(tf.expand_dims(
                image, 0), k_size, strides, rates=[1, 1, 1, 1], padding='SAME')[0]
            images_shape = tf.shape(images)
            images_reshape = tf.reshape(
                images, [images_shape[0]*images_shape[1], *k_size[1:3], 3])
            images, n1, n2 = tf.cast(
                images_reshape, tf.uint8), images_shape[0], images_shape[1]
            return images, n1, n2

        def join_patches(images, n1, n2, k_size, strides):

            s1 = k_size[1]//2-strides[1]//2
            s2 = k_size[2]//2-strides[2]//2
            roi = images[:,
                         s1:s1+strides[1],
                         s2:s2+strides[2],
                         :]
            new_shape = [n1, n2, *roi.shape[1:]]
            reshaped_roi = tf.reshape(roi, new_shape)
            reshaped_roi = tf.transpose(reshaped_roi, perm=[0, 2, 1, 3, 4])
            rs = tf.shape(reshaped_roi)
            rv = tf.reshape(reshaped_roi, [rs[0]*rs[1], rs[2]*rs[3], -1])
            return rv

        def resize(image, new_size=None):
            shape = tf.shape(image)
            h, w = shape[0], shape[1]
            if new_size is None:
                new_h = tf.cast(tf.ceil(h/CROP_SIZE[0])*CROP_SIZE[0], tf.int32)
                new_w = tf.cast(tf.ceil(w/CROP_SIZE[1])*CROP_SIZE[1], tf.int32)
            else:
                new_h, new_w = new_size
            return tf.image.resize_bilinear(tf.expand_dims(image, 0), (new_h, new_w))[0]
        # inputs = tf.placeholder(tf.float32, [None, *CROP_SIZE, 3], 'inputs')
        inputs = tf.placeholder(tf.float32, [None, None, 3], 'inputs')
        inputs_shape = tf.shape(inputs)
        input_resized = resize(inputs)
        # strides = tf.placeholder_with_default([32, 256], shape=[2], name='strides')
        strides = a.strides
        images, n1, n2 = extract_patches(
            input_resized, [1, *CROP_SIZE, 1], [1, *strides, 1])

        batch_input = images / 255
        print('Batch input:', batch_input)
        with tf.variable_scope("generator"):
            batch_output = deprocess(
                create_generator(preprocess(batch_input), 3))
        batch_output = join_patches(batch_output, n1, n2, [
                                    1, *CROP_SIZE, 1], [1, *strides, 1])
        batch_output = resize(batch_output, [inputs_shape[0], inputs_shape[1]])
        outputs = tf.identity(
            tf.cast(batch_output*255, tf.uint8), name='outputs')

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model:", checkpoint)
            export_saver.export_meta_graph(
                filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(
                a.output_dir, "export"), write_meta_graph=False)

        return

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output

    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(
                image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(
            model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(
            model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum(
            [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for f in (filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options,
                                   run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(
                        results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(
                        results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(
                        run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(
                        results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] -
                                  1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" %
                          (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(
                        a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
