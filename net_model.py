import os
import pathlib
import glob
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ProgressBar import ProgressBar
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''============= MODEL REGION ================'''
p_drop = 0.5


def sample_images(dataset, in_cubes, out_cubes, epoch, batch_i, show_title=False):
    assert len(in_cubes) == len(out_cubes)
    assert len(in_cubes) % 2 == 0
    os.makedirs('%s/generated' % dataset['cube_dir'], exist_ok=True)
    mid = len(in_cubes)//2
    r, c = 4, mid
    gen_imgs = np.concatenate([0.5*in_cubes[:mid]+0.5, 0.5*out_cubes[:mid]+0.5, 0.5*in_cubes[mid:]+0.5, 0.5*out_cubes[mid:]+0.5])
    titles = ['in_cube', 'out_cube', 'in_cube', 'out_cube']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(np.clip(gen_imgs[cnt], 0., 1.))
            if show_title:
                axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig('%s/generated/%d_%d.png' % (dataset['cube_dir'], epoch, batch_i))
    plt.close()


def dense(x, out_unit, xavier=False, scope=None, return_weights=False):
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix', [x.get_shape()[-1], out_unit], tf.float32,
                                 tf.contrib.layers.xavier_initializer() if xavier else tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', [out_unit], initializer=tf.constant_initializer(0.0))
        if return_weights:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias


def conv2d(x, out_channel, filter_size=3, stride=1, scope=None, return_filters=False):
    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    assert len(filter_size) == 2
    with tf.variable_scope(scope):
        in_channel = x.get_shape()[-1]
        w = tf.get_variable('w', [filter_size[0], filter_size[1], in_channel, out_channel], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        result = tf.nn.conv2d(x, w, [1, stride, stride, 1], 'SAME') + b
        if return_filters:
            return result, w, b
        return result


def conv_transpose(x, output_shape, filter_size=3, scope=None, return_filters=False):
    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    assert len(filter_size) == 2
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [filter_size[0], filter_size[1], output_shape[-1], x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.bias_add(tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, 2, 2, 1]), b)
        if return_filters:
            return convt, w, b
        return convt


# input (n, 10, 10, 3)
def Generator(input_data, cube_size, is_training, keep_prob, n_row, n_col, return_layers=False):
    def G_conv_conv_bn_relu(x, conv_channels, filter_sizes, strides, training=False, bn=True, scope=None):
        assert len(conv_channels) == 2
        assert len(filter_sizes) == 2
        assert len(strides) == 2
        with tf.variable_scope(scope):
            d = conv2d(x, conv_channels[0], filter_size=filter_sizes[0], stride=strides[0], scope='conv1')
            if conv_channels[1] is not None:
                d = conv2d(d, conv_channels[1], filter_size=filter_sizes[1], stride=strides[1], scope='conv2')
            if bn:
                d = tf.layers.batch_normalization(d, training=training)
            d = tf.nn.leaky_relu(d)
            return d

    def G_deconv_conv_bn_dr_relu(x, deconv_shape, conv_channel, filter_sizes, p_keep_drop, training=False, scope=None):
        assert len(filter_sizes) == 2
        with tf.variable_scope(scope):
            """Layers used during upsampling"""
            u = conv_transpose(x, deconv_shape, filter_size=filter_sizes[0], scope='deconv')
            if filter_sizes[1] is not None:
                u = conv2d(u, conv_channel, filter_size=filter_sizes[1], stride=1, scope='conv')
            u = tf.layers.batch_normalization(u, training=training)
            u = tf.nn.dropout(u, p_keep_drop)
            u = tf.nn.relu(u)
            return u

    def dense_net(flattened_input, hidden_units, hidden_active_func, p_keep_drop, out_unit, scope=None):
        assert len(hidden_units) == 2
        with tf.variable_scope(scope):
            h = flattened_input
            print('flatten_shape:', h)
            if hidden_units[0] is not None:
                h = hidden_active_func(dense(h, hidden_units[0], scope='hidden1'))
                h = tf.nn.dropout(h, p_keep_drop)
            if hidden_units[1] is not None:
                h = hidden_active_func(dense(h, hidden_units[1], scope='hidden2'))
                h = tf.nn.dropout(h, p_keep_drop)
            logits = dense(h, out_unit, scope='output')
            softmax = tf.nn.softmax(logits, name='out_softmax')
            return logits, softmax

    with tf.variable_scope('generator'):
        b_size = tf.shape(input_data)[0]
        h = cube_size[0]
        w = cube_size[1]
        d = cube_size[2]

        hs = [h, int(np.ceil(h/2)), int(np.ceil(h/4))]
        ws = [w, int(np.ceil(w/2)), int(np.ceil(w/4))]

        h0 = input_data

        '''encoder'''
        h1 = G_conv_conv_bn_relu(h0, [32, 64], [3, 3], [1, 1], training=is_training, bn=False, scope='gen_h1_enc')
        h2 = G_conv_conv_bn_relu(h1, [64, 128], [3, 3], [2, 1], training=is_training, bn=True, scope='gen_h2_enc')
        h3 = G_conv_conv_bn_relu(h2, [128, 256], [3, 3], [2, 1], training=is_training, bn=True, scope='gen_h3_enc')

        '''decoder'''
        h4 = G_conv_conv_bn_relu(h3, [128, None], [3, None], [1, None], training=is_training, bn=True, scope='gen_h4_dec')
        h5 = G_deconv_conv_bn_dr_relu(h4, [b_size, hs[1], ws[1], 128], 64, [3, 3], keep_prob, training=is_training, scope='gen_h5_dec')
        h6 = G_deconv_conv_bn_dr_relu(h5, [b_size, h, w, 64], 32, [3, 3], keep_prob, training=is_training, scope='gen_h6_dec')
        g_output = conv2d(h6, d, filter_size=3, stride=1, scope='gen_output')

        '''classification'''
        h1_conv_1x1 = conv2d(h1, 32, filter_size=1, stride=1, scope='gen_h1_conv_1x1')
        h2_conv_1x1 = conv2d(h2, 64, filter_size=1, stride=1, scope='gen_h2_conv_1x1')
        flattened_input = tf.concat([tf.layers.Flatten()(h1_conv_1x1), tf.layers.Flatten()(h2_conv_1x1), tf.layers.Flatten()(h3)], 1)
        out_row_logits, out_row_softmax = dense_net(flattened_input, [1024, 1024], tf.nn.relu, keep_prob, n_row, scope='gen_row_index')
        out_col_logits, out_col_softmax = dense_net(flattened_input, [1024, 1024], tf.nn.relu, keep_prob, n_col, scope='gen_col_index')

        if return_layers:
            return g_output, out_row_logits, out_row_softmax, out_col_logits, out_col_softmax, [h1, h2, h3, h4, h5, h6, h1_conv_1x1, h2_conv_1x1]
        return g_output, out_row_logits, out_row_softmax, out_col_logits, out_col_softmax


# (n, 10, 10, 3)
def Discriminator(input_data, cube_size, is_training, reuse=False, return_layers=False):
    def D_conv_bn_active(x, out_channel, filter_size, stride=2, training=False, bn=True, active=tf.nn.leaky_relu, scope=None):
        with tf.variable_scope(scope):
            d = conv2d(x, out_channel, filter_size=filter_size, stride=stride, scope='conv')
            if bn:
                d = tf.layers.batch_normalization(d, training=training)
            if active is not None:
                d = active(d)
            return d

    with tf.variable_scope('discriminator') as var_scope:
        if reuse:
            var_scope.reuse_variables()

        # h = cube_size[0]
        # w = cube_size[1]

        # hs = [h, int(np.ceil(h/2)), int(np.ceil(h/4))]
        # ws = [w, int(np.ceil(w/2)), int(np.ceil(w/4))]

        filters, filter_size = 32, 3

        h0 = input_data
        h1 = D_conv_bn_active(h0, filters, filter_size, stride=2, training=is_training, bn=False, scope='dis_h1')
        h2 = D_conv_bn_active(h1, filters*2, filter_size, stride=2, training=is_training, bn=True, scope='dis_h2')
        h3 = D_conv_bn_active(h2, filters*4, filter_size, stride=2, training=is_training, bn=True, scope='dis_h3')
        h4 = D_conv_bn_active(h3, 1, 1, stride=1, training=is_training, bn=True, active=None, scope='dis_h4')
        d_logits = dense(tf.layers.Flatten()(h4), 128, scope='dis_logit')
        return tf.nn.sigmoid(d_logits), d_logits


def train_model_naive_with_batch_norm(dataset, training_cubes, cube_row_indices, cube_col_indices,
                                      n_row, n_col, max_epoch, start_model_idx=0, batch_size=256):
    assert len(training_cubes) == len(cube_row_indices)
    assert len(cube_row_indices) == len(cube_col_indices)
    print('no. of training cubes = %d' % len(training_cubes))
    h, w, d = training_cubes.shape[1:4]
    #
    training_cubes /= 0.5
    training_cubes -= 1.
    #
    with tf.device('/device:GPU:0'):
        plh_cube_true = tf.placeholder(tf.float32, shape=[None, h, w, d])
        plh_row_idx = tf.placeholder(tf.uint8, shape=[None])
        plh_col_idx = tf.placeholder(tf.uint8, shape=[None])
        plh_is_training = tf.placeholder(tf.bool)

        # generator
        plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
        cube_recon, out_row_logits, out_row_softmax, out_col_logits, out_col_softmax = \
            Generator(plh_cube_true, [h, w, d], plh_is_training, plh_dropout_prob, n_row, n_col)

        # discriminator
        D_real, D_real_logits = Discriminator(plh_cube_true, [h, w, d], plh_is_training, reuse=False)
        D_fake, D_fake_logits = Discriminator(cube_recon, [h, w, d], plh_is_training, reuse=True)

        # appearance loss
        dy1, dx1 = tf.image.image_gradients(cube_recon)
        _, dt1 = tf.image.image_gradients(tf.transpose(cube_recon, perm=[0, 1, 3, 2]))
        dy0, dx0 = tf.image.image_gradients(plh_cube_true)
        _, dt0 = tf.image.image_gradients(tf.transpose(plh_cube_true, perm=[0, 1, 3, 2]))
        loss_inten = tf.reduce_mean((cube_recon - plh_cube_true)**2)
        loss_gradi = tf.reduce_mean(tf.reduce_mean(tf.abs(tf.abs(dy1)-tf.abs(dy0))) +
                                    tf.reduce_mean(tf.abs(tf.abs(dx1)-tf.abs(dx0))) +
                                    tf.reduce_mean(tf.abs(tf.abs(dt1)-tf.abs(dt0))))
        loss_appe = loss_inten + loss_gradi

        # classification loss
        loss_class_row = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_row_logits, labels=tf.one_hot(plh_row_idx, n_row)))
        loss_class_col = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_col_logits, labels=tf.one_hot(plh_col_idx, n_col)))
        loss_class = loss_class_row + loss_class_col

        # GAN loss
        D_loss = 0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real))) + \
                 0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))
        G_loss_total = 0.25*G_loss + loss_appe + loss_class

        # tensorboard
        tf.summary.histogram('D_loss', D_loss)
        tf.summary.histogram('G_loss', G_loss)
        tf.summary.histogram('appe_loss', loss_appe)
        tf.summary.histogram('class_row_loss', loss_class_row)
        tf.summary.histogram('class_col_loss', loss_class_col)

        # optimizers
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'gen_' in var.name]
        d_vars = [var for var in t_vars if 'dis_' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            D_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001, name='SGDD').minimize(D_loss, var_list=d_vars)
            G_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9, name='AdamG').minimize(G_loss_total, var_list=g_vars)
        init_op = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=20)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        # define log path for tensorboard
        tensorboard_path = '%s/logs/1/train' % dataset['cube_dir']
        model_dir = '%s/models' % dataset['cube_dir']
        if not os.path.exists(tensorboard_path):
            pathlib.Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(model_dir):
            pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        #
        train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
        print('Run: tensorboard --logdir logs/1')
        # executive training stage
        sess.run(init_op)
        losses = np.array([], dtype=np.float32).reshape((0, 5))

        if start_model_idx > 0:
            # delete old checkpoints
            model_to_delete = [v for v in np.arange(1, start_model_idx) if v % 5 != 0]
            for m in model_to_delete:
                model_files = glob.glob('%s/model_ckpt_%d.ckpt.*' % (model_dir, m))
                for model_file in model_files:
                    os.remove(model_file)
            # load saved model
            saver.restore(sess, '%s/model_ckpt_%d.ckpt' % (model_dir, start_model_idx))
            losses = np.loadtxt('%s/train_loss_%d.txt' % (model_dir, start_model_idx), delimiter=',')

        for i in range(start_model_idx, max_epoch):
            tf.set_random_seed(i)
            np.random.seed(i)
            #
            # N_EPOCH = np.min(1000, len(training_cubes)//batch_size)
            # selected_idx = np.random.choice(np.arange(len(training_cubes)), N_EPOCH * batch_size, replace=False)
            # training_cubes = training_cubes[selected_idx]
            # cube_row_indices = cube_row_indices[selected_idx]
            # cube_col_indices = cube_col_indices[selected_idx]
            #
            batch_idx = np.array_split(np.random.permutation(len(training_cubes)),
                                       np.ceil(len(training_cubes)/batch_size))

            if len(batch_idx) > 840:  # max 945 to fit memory
                batch_idx = batch_idx[:840]
            for j in range(len(batch_idx)):
                merge = tf.summary.merge_all()
                # discriminator
                _, curr_D_loss, summary = sess.run([D_optimizer, D_loss, merge],
                                                   feed_dict={plh_cube_true: training_cubes[batch_idx[j]],
                                                              plh_row_idx: cube_row_indices[batch_idx[j]].astype(np.uint8),
                                                              plh_col_idx: cube_col_indices[batch_idx[j]].astype(np.uint8),
                                                              plh_is_training: True,
                                                              plh_dropout_prob: p_drop})
                if j % 20 == 0:
                    _, curr_G_loss, curr_loss_appe, curr_loss_class_row, curr_loss_class_col, curr_gen_cubes, summary = \
                           sess.run([G_optimizer, G_loss, loss_appe, loss_class_row, loss_class_col, cube_recon[:8], merge],
                                    feed_dict={plh_cube_true: training_cubes[batch_idx[j]],
                                               plh_row_idx: cube_row_indices[batch_idx[j]].astype(np.uint8),
                                               plh_col_idx: cube_col_indices[batch_idx[j]].astype(np.uint8),
                                               plh_is_training: True,
                                               plh_dropout_prob: p_drop})
                    sample_images(dataset, training_cubes[batch_idx[j]][:8], curr_gen_cubes, i, j)
                else:
                    _, curr_G_loss, curr_loss_appe, curr_loss_class_row, curr_loss_class_col, summary = \
                           sess.run([G_optimizer, G_loss, loss_appe, loss_class_row, loss_class_col, merge],
                                    feed_dict={plh_cube_true: training_cubes[batch_idx[j]],
                                               plh_row_idx: cube_row_indices[batch_idx[j]].astype(np.uint8),
                                               plh_col_idx: cube_col_indices[batch_idx[j]].astype(np.uint8),
                                               plh_is_training: True,
                                               plh_dropout_prob: p_drop})
                # write log for tensorboard
                train_writer.add_summary(summary, i*len(batch_idx)+j)

                print('epoch %2d/%d, iter %3d/%d: D_loss = %.4f, G_loss = %.4f, loss_appe = %.4f, loss_class = (%.4f %.4f)'
                      % (i+1, max_epoch, j+1, len(batch_idx), curr_D_loss, curr_G_loss, curr_loss_appe, curr_loss_class_row, curr_loss_class_col))
                if np.isnan(curr_D_loss) or np.isnan(curr_G_loss) or np.isnan(curr_loss_appe) or \
                   np.isnan(curr_loss_class_row) or np.isnan(curr_loss_class_col):
                    return
                losses = np.concatenate((losses, [[curr_D_loss, curr_G_loss, curr_loss_appe, curr_loss_class_row, curr_loss_class_col]]), axis=0)
        saver.save(sess, '%s/model_ckpt_%d.ckpt' % (model_dir, max_epoch))
        np.savetxt('%s/train_loss_%d.txt' % (model_dir, max_epoch), losses, delimiter=',')


def test_model_naive_with_batch_norm(dataset, test_cubes, cube_row_indices, cube_col_indices, n_row, n_col,
                                     sequence_n_frame, clip_idx, model_idx=20, batch_size=256, using_test_data=True):
    assert len(test_cubes) == len(cube_row_indices)
    assert len(cube_row_indices) == len(cube_col_indices)
    print('shape of test cubes = %s' % str(test_cubes.shape))
    print(np.sum(sequence_n_frame))
    h, w, d = test_cubes.shape[1:4]
    #
    score_dir = '%s/scores' % dataset['cube_dir']
    if not os.path.exists(score_dir):
        pathlib.Path(score_dir).mkdir(parents=True, exist_ok=True)
    saved_data_path = '%s/output_%s/%d_epoch' % (score_dir, 'test' if using_test_data else 'train', model_idx)
    if not os.path.exists(saved_data_path):
        pathlib.Path(saved_data_path).mkdir(parents=True, exist_ok=True)
    saved_data_file = '%s/output_%d.npz' % (saved_data_path, clip_idx)
    if os.path.isfile(saved_data_file):
        print('File existed! Return!')
        return
    #
    test_cubes /= 0.5
    test_cubes -= 1.
    #
    plh_cube_true = tf.placeholder(tf.float32, shape=[None, h, w, d])
    plh_row_idx = tf.placeholder(tf.uint8, shape=[None])
    plh_col_idx = tf.placeholder(tf.uint8, shape=[None])
    plh_is_training = tf.placeholder(tf.bool)

    # generator
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    cube_recon, _, out_row_softmax, _, out_col_softmax = Generator(plh_cube_true, [h, w, d], plh_is_training, plh_dropout_prob, n_row, n_col)
    #
    saver = tf.train.Saver(max_to_keep=20)
    #
    saved_out_cubes = np.zeros(test_cubes.shape)
    saved_out_row_softmax = np.zeros([len(test_cubes), n_row])
    saved_out_col_softmax = np.zeros([len(test_cubes), n_col])

    with tf.Session() as sess:
        saved_model_file = '%s/models/model_ckpt_%d.ckpt' % (dataset['cube_dir'], model_idx)
        saver.restore(sess, saved_model_file)
        #

        batch_idx = np.array_split(np.arange(len(test_cubes)), np.ceil(len(test_cubes)/batch_size))
        progress = ProgressBar(len(batch_idx), fmt=ProgressBar.FULL)
        start_pt = time.time()
        for j in range(len(batch_idx)):
            progress.current += 1
            progress()
            saved_out_cubes[batch_idx[j]], saved_out_row_softmax[batch_idx[j]], saved_out_col_softmax[batch_idx[j]] = \
                sess.run([cube_recon, out_row_softmax, out_col_softmax],
                         feed_dict={plh_cube_true: test_cubes[batch_idx[j]],
                         plh_row_idx: cube_row_indices[batch_idx[j]],
                         plh_col_idx: cube_col_indices[batch_idx[j]],
                         plh_is_training: False,
                         plh_dropout_prob: 1.0})
            saved_out_cubes[batch_idx[j]] = 0.5*(saved_out_cubes[batch_idx[j]] + 1.)
        print('batch size: %d => average time per batch: %f' % (batch_size, (time.time()-start_pt)/len(batch_idx)))
        progress.done()

    np.savez_compressed(saved_data_file, cube=saved_out_cubes, row=saved_out_row_softmax, col=saved_out_col_softmax)


def visualize_filters(dataset, cube_size, n_row, n_col, model_idx=20):
    def combine_filters(data, dim=None, new_size=71, gap=4):
        c = 8
        r = data.shape[-1]//c
        if dim is None:
            full_img = np.ones((r*new_size+(r-1)*gap, c*new_size+(c-1)*gap, 3))
        else:
            full_img = np.ones((r*new_size+(r-1)*gap, c*new_size+(c-1)*gap))
        idx = 0
        for i in range(r):
            for j in range(c):
                img = data[..., dim, idx] if dim is not None else data[..., idx]
                img = (img-np.min(img))/(np.max(img)-np.min(img))
                img = cv2.resize(img, (new_size, new_size), cv2.INTER_LINEAR)
                full_img[i*(new_size+gap):i*(new_size+gap)+new_size, j*(new_size+gap):j*(new_size+gap)+new_size] = img
                idx += 1
        plt.figure()
        plt.imshow(full_img, cmap='gray')
        return (full_img*255).astype(int)

    def visualize_single(data, dim=None):
        c = 8
        r = data.shape[-1]//c
        plt.figure()
        idx = 0
        for i in range(r):
            for j in range(c):
                idx += 1
                plt.subplot(r, c, idx)
                if dim is not None:
                    img = data[:, :, dim, idx-1]
                else:
                    img = data[:, :, :, idx-1]
                img = (img-np.min(img))/(np.max(img)-np.min(img))
                plt.imshow(cv2.resize(img, (71, 71), cv2.INTER_LINEAR), cmap='gray')
    #
    out_dir = '%s/filters/model_%d' % (dataset['cube_dir'], model_idx)
    if not os.path.exists(out_dir):
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    #
    plh_cube_true = tf.placeholder(tf.float32, shape=[None, cube_size[0], cube_size[1], cube_size[2]])
    plh_is_training = tf.placeholder(tf.bool)

    # generator
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    cube_recon, _, out_row_softmax, _, out_col_softmax = Generator(plh_cube_true, cube_size, plh_is_training, plh_dropout_prob, n_row, n_col)
    #
    saver = tf.train.Saver(max_to_keep=20)
    #
    with tf.Session() as sess:
        saved_model_file = '%s/models/model_ckpt_%d.ckpt' % (dataset['cube_dir'], model_idx)
        saver.restore(sess, saved_model_file)
        #
        print('=========== scopes ==========')
        for v in tf.global_variables():
            print(v.name, v.shape)
            data = v.eval()
            if len(data.shape) == 4 and np.sum(data.shape[:3]) == 9:
                filter_all = combine_filters(data, dim=None)
                cv2.imwrite('%s/filters.png' % out_dir, filter_all)
                filter_0 = combine_filters(data, dim=0)
                cv2.imwrite('%s/filter_0.png' % out_dir, filter_0)
                filter_1 = combine_filters(data, dim=1)
                cv2.imwrite('%s/filter_1.png' % out_dir, filter_1)
                filter_2 = combine_filters(data, dim=2)
                cv2.imwrite('%s/filter_2.png' % out_dir, filter_2)
                np.savez_compressed('%s/filters.npz' % out_dir, filter_all=filter_all, filters=np.array([filter_0, filter_1, filter_2]))
                plt.show()


def convert_model(dataset, cube_size, n_row, n_col, model_idx=20):
    plh_cube_true = tf.placeholder(tf.float32, shape=[None, cube_size[0], cube_size[1], cube_size[2]])
    plh_is_training = tf.placeholder(tf.bool)

    # generator
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    cube_recon, _, out_row_softmax, _, out_col_softmax = Generator(plh_cube_true, cube_size, plh_is_training, plh_dropout_prob, n_row, n_col)
    #
    saver = tf.train.Saver(max_to_keep=20)
    #
    with tf.Session() as sess:
        saved_model_file = '%s/models/model_ckpt_%d.ckpt' % (dataset['cube_dir'], model_idx)
        saver.restore(sess, saved_model_file)
        out_file = '%s.pb' % dataset['name']
        tf.train.write_graph(sess.graph_def, dataset['cube_dir'], out_file, as_text=False)
        #
        graph_def = tf.GraphDef()
        with open('%s/%s' % (dataset['cube_dir'], out_file), 'rb') as f:
            graph_def.ParseFromString(f.read())
        for node in graph_def.node:
            print(node.name)
