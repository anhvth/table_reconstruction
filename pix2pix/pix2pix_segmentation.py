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
parser.add_argument("--input_shape", default=[128, 1024], help="shape of input")
parser.add_argument("--num_of_class", default=2, help="shape of input")
parser.add_argument("--mode", required=True,
                    choices=["train", "test", "export"])
parser.add_argument("--output_dir", default='pix2pix/frozen/128x1024',
                    required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int,
                    help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100,
                    help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50,
                    help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0,
                    help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=1000,
                    help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true",
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
                    128, 1024], help="scale images to this size before cropping to 256x256")
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

parser.add_argument("--steps_per_epoch", type=int, default=None,
                    help="")

# export options
parser.add_argument("--output_filetype", default="png",
                    choices=["png", "jpeg"])
a = parser.parse_args()
EPS = 1e-12
CROP_SIZE = [128, 1024]

Examples = collections.namedtuple(
    "Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple(
    "Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train, global_step")


if a.mode == 'train':
    a.steps_per_epoch = 500*len(glob.glob('{}/*'.format(a.input_dir)))

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
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
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels, stride):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(stride, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(stride, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels, stride):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(
            batch_input, [h * 2, w * stride], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(stride, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return    (0.5 * (1 + a)) * x \
                + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


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


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []
    print('Create GENERATOR--------------------------------')
    # encoder_1: [batch, 128, 1024, in_channels] => [batch, 64, 512, ngf]
    with tf.variable_scope("encoder_1"):
        print('generator_inputs: ',generator_inputs.shape)
        
        output = gen_conv(generator_inputs, a.ngf, 2)
        layers.append(output)
        print(output.shape)
    layer_specs = [
        # encoder_2: [batch, 64, 512, ngf] => [batch,  32, 256, ngf * 2]
        a.ngf * 2,
        # encoder_3: [batch, 32, 256, ngf * 2] => [batch, 16, 128, ngf * 4]
        a.ngf * 4,
        # encoder_4: [batch, 16, 128, ngf * 4] => [batch, 8, 64, ngf * 8]
        a.ngf * 8,
        # encoder_5: [batch, 8, 64, ngf * 8] => [batch, 4, 32, ngf * 8]
        a.ngf * 8,
        # encoder_6: [batch, 4, 32, ngf * 8] => [batch, 2, 16, ngf * 8]
        a.ngf * 8,
        # encoder_7: [batch, 2, 16, ngf * 8] => [batch, 1, 8, ngf * 8]
        a.ngf * 8,
        # encoder_8: [batch, 1, 8, ngf * 8] => [batch, 1, 4, ngf * 8]
        a.ngf * 8,
    ]

    strides = [2, 2, 2, 2, 2, 2, 1]

    for out_channels, stride in zip(layer_specs, strides):
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, stride)
            output = batchnorm(convolved)
            layers.append(output)
            print(output.shape)

    layer_specs = [
        # decoder_8: [batch, 1, 4, ngf * 8] =>          [batch, 1, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.5, 1),
        # decoder_7: [batch, 1, 8, ngf * 8 * 2] =>      [batch, 2, 16, ngf * 8 * 2]
        (a.ngf * 8, 0.5, 2),
        # decoder_6: [batch, 2, 16, ngf * 8 * 2] =>     [batch, 4, 32, ngf * 8 * 2]
        (a.ngf * 8, 0.5, 2),
        # decoder_5: [batch, 4, 32, ngf * 8 * 2] =>     [batch, 8, 64, ngf * 8 * 2]
        (a.ngf * 8, 0.0, 2),
        # decoder_4: [batch, 8, 64, ngf * 8 * 2] =>     [batch, 16, 128, ngf * 4 * 2]
        (a.ngf * 4, 0.0, 2),
        # decoder_3: [batch, 16, 128, ngf * 4 * 2] =>   [batch, 32,256, ngf * 2 * 2]
        (a.ngf * 2, 0.0, 2),
        # decoder_2: [batch, 32, 256, ngf * 2 * 2] =>   [batch, 64, 512, ngf * 2]
        (a.ngf, 0.0, 2),
    ]
    print('DECONV')
    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout, stride) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels, stride)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)
            print(output.shape)
    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        # logits_text_area = gen_deconv(rectified, a.num_of_class, 2)
        # logits_text_line = gen_deconv(rectified, a.num_of_class, 2)
        
        logits = gen_deconv(rectified, a.num_of_class, 2) #tf.concat([logits_text_area[:,:,:,:1], logits_text_line[:,:,:,:1]], axis=-1)

        output = tf.sigmoid(logits)

        print('output: ', output.shape)
    return logits, output


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
        logits, outputs = create_generator(inputs, out_channels)

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        print('Outputs:', outputs.shape, '\t target:', targets.shape)
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):

        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) +
                                        tf.log(1 - predict_fake + EPS)))

    def distance_loss(targets, logits_text_area, logits_text_line):
        targets = tf.to_int32(targets)
        targets_text_area = tf.one_hot(targets[:,:,:,0], 2)
        targets_text_line = tf.one_hot(targets[:,:,:,1], 2)
        l1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_text_area, labels=targets_text_area))
        l2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_text_line, labels=targets_text_line))
        return (l1+l2) / 2

    with tf.name_scope("generator_loss"):
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 =  tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = tf.trainable_variables(
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
        global_step=global_step
    )

def get_data_generator(input_dir):
    paths = glob.glob('{}/*'.format(input_dir))
    while True:
        path = paths[np.random.choice(len(paths))]
        with np.load(path) as data:
            inputs = data['inputs']
            labels = data['labels']
            idxs = np.arange(len(inputs))
            np.random.shuffle(idxs)
            inputs = inputs[idxs]
            labels = labels[idxs]
        x, y = [], []
        for i in range(0, len(inputs), a.batch_size):
            k = min(len(inputs), i+a.batch_size)
            # print('yield:', i, k)
            yield inputs[i:k], labels[i:k]
            



def export():
    inputs = tf.placeholder(tf.float32, [None, 128, 1024, 3], 'inputs')

    batch_input = inputs / 255
    with tf.variable_scope("generator"):
        batch_output = deprocess(
            create_generator(preprocess(batch_input), 3))
    outputs = tf.identity(batch_output*255, name='outputs')

    init_op = tf.global_variables_initializer()
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        print("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        restore_saver.restore(sess, checkpoint)
        print("exporting model")
        export_saver.export_meta_graph(
            filename=os.path.join(a.output_dir, "export.meta"))
        export_saver.save(sess, os.path.join(
            a.output_dir, "export"), write_meta_graph=False)


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

    inputs = tf.placeholder(tf.float32, [None, *a.input_shape, 3], name='inputs')#deprocess(examples.inputs)
    targets = tf.placeholder(tf.float32, [None, *a.input_shape, a.num_of_class], name='targets')
    model = create_model(inputs, targets)
    outputs = model.outputs

    def labelmap_to_image(images):

        def f(labelmap):
            labelmap = tf.to_float(labelmap>0.8)
            im1 = labelmap[:,:,:1] * np.array([255,0,0]).reshape([1,1,3])
            im2 = labelmap[:,:,1:2] * np.array([255,255,0]).reshape([1,1,3])
            im = im1+im2
            return tf.image.convert_image_dtype(im, dtype=tf.uint8, saturate=True)
        return tf.map_fn(f, images, dtype=tf.uint8)
    # summaries
    with tf.name_scope("inputs_summary"):
        converted_inputs = tf.image.convert_image_dtype((inputs+1)/2, tf.uint8)
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        converted_targets = labelmap_to_image(targets)
        tf.summary.image("targets", converted_targets)
        
    with tf.name_scope("outputs_summary"):
        converted_outputs = labelmap_to_image(outputs)
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
    summary_merged = tf.summary.merge_all()


    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum(
            [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    # sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    data_generator = get_data_generator(a.input_dir)
    with tf.Session() as sess:
        # assert a.steps_per_epoch is not None
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = a.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps


        # training
        start = time.time()
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(a.output_dir, sess.graph)
        for step in range(max_steps):
            input_data, target_data = next(data_generator)

            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            fetches = {
                "train": model.train,
                "global_step": model.global_step,
            }

            if should(a.summary_freq):
                fetches["summary"] = summary_merged

            if should(a.progress_freq):
                fetches["discrim_loss"] = model.discrim_loss
                fetches["gen_loss_GAN"] = model.gen_loss_GAN
                fetches["gen_loss_L1"] = model.gen_loss_L1

        
            if should(a.display_freq):
                fetches["display"] = display_fetches

            feed_dict = {inputs: input_data, targets: target_data}
            results = sess.run(fetches, feed_dict=feed_dict)

            # Print out training process to terminal
            if should(a.progress_freq):
                train_epoch = math.ceil(
                    results["global_step"] / a.steps_per_epoch)
                train_step = (results["global_step"] -
                                1) % a.steps_per_epoch + 1
                rate = (step + 1) * a.batch_size / (time.time() - start)
                remaining = (max_steps - step) * a.batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" %
                        (train_epoch, train_step, rate, remaining / 60))
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])
            
            if should(a.summary_freq):
                print("recording summary")
                train_writer.add_summary(results["summary"], results["global_step"])
            
            # Save model    
            if should(a.save_freq):
                print("saving model")
                saver.save(sess, os.path.join(
                    a.output_dir, "model"), global_step=model.global_step)

if a.mode == "export":
    export()
else:
    main()
