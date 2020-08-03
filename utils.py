import os
import glob
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.utils.fixes import signature
from skimage.measure import compare_ssim as ssim

from scipy.misc import imread
from scipy.io import loadmat, savemat

from ROC import assessment
from ProgressBar import ProgressBar

import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def check_path(dataset, cube_size):
    cube_str = '%d_%d_%d' % tuple(cube_size)
    assert cube_str in dataset['cube_dir']


# [SECTION] IMAGE PROCESSING
# important: load as gray image (i.e. 1 channel)

def resize(datum, size):
    assert len(datum.shape) == 2
    ret = cv2.resize(datum.astype(float), tuple(size))
    return ret


def load_images_and_resize(dataset, new_size=[120, 160], train=True, force_recalc=False, return_entire_data=False):
    img_dir = dataset['path_train' if train else 'path_test']
    n_images = np.sum(count_sequence_n_frame(dataset, test=not train))
    print('number of images: ', n_images)
    n_clip = dataset['n_clip_train' if train else 'n_clip_test']
    #
    if return_entire_data:
        resized_image_data = np.empty((n_images, new_size[0], new_size[1], 1), dtype=np.float32)
        idx = 0
    #
    for i in range(n_clip):
        clip_path = '%s/%s%s/' % (img_dir, 'Train' if train else 'Test', str(i+1).zfill(3))
        print(clip_path)
        # image
        img_files = sorted(glob.glob(clip_path + '*.tif'))
        saved_image_file = '%s/%s_image_clip_%d.npz' % (dataset['cube_dir'], 'training' if train else 'test', i+1)
        if os.path.isfile(saved_image_file) and not force_recalc:
            image_data = np.load(saved_image_file)['image']
        else:
            image_data = np.array([resize(imread(img_file, 'L')/255., (new_size[1], new_size[0])) for img_file in img_files]).astype(np.float32)
            np.savez_compressed(saved_image_file, image=image_data)
        print('clip', i+1, image_data.shape)

        if return_entire_data:
            resized_image_data[idx:idx+len(image_data)] = image_data
            idx += len(image_data)
    #
    if return_entire_data:
        return resized_image_data


def load_images_single_clip(dataset, clip_idx, indices, train=True):
    assert clip_idx in np.arange(dataset['n_clip_train' if train else 'n_clip_test'])
    img_dir = dataset['path_train' if train else 'path_test']
    n_images = count_sequence_n_frame(dataset, test=not train)[clip_idx]
    print('number of images: ', n_images)
    #
    clip_path = '%s/%s%s/' % (img_dir, 'Train' if train else 'Test', str(clip_idx+1).zfill(3))
    print(clip_path)
    # image
    img_files = sorted(glob.glob(clip_path + '*.tif'))
    image_data = np.array([imread(img_files[idx])/255. for idx in indices]).astype(np.float32)
    print('clip', clip_idx+1, image_data.shape)
    return image_data


# [SECTION] CUBE PROCESSING
def split_cubes(dataset, clip_idx, cube_size, training_set=True, force_recalc=False, dist_thresh=None):
    check_path(dataset, cube_size)
    n_clip = dataset['n_clip_train' if training_set else 'n_clip_test']
    assert clip_idx in range(n_clip)
    print('clip %2d/%2d' % (clip_idx + 1, n_clip))
    # load from file if existed
    saved_cube_file = '%s/%s_cubes_clip_%d_size_%d_%d_%d.npz' % \
                      (dataset['cube_dir'], 'training' if training_set else 'test', clip_idx + 1, cube_size[0], cube_size[1], cube_size[2])
    if os.path.isfile(saved_cube_file) and not force_recalc:
        loader = np.load(saved_cube_file)
        cubes = loader['data']
        mapping = loader['mapping']
        return cubes, mapping

    # first load image data from file
    saved_image_file = '%s/%s_image_clip_%d.npz' % (dataset['cube_dir'], 'training' if training_set else 'test', clip_idx + 1)
    if not os.path.isfile(saved_image_file):
        print('image file not found! (%s)' % saved_image_file)
        return None, None
    image_data = np.load(saved_image_file)['image']
    h, w = image_data.shape[1:3]
    assert h % cube_size[0] == 0
    assert w % cube_size[1] == 0
    h_grid, w_grid = np.array([h, w])//cube_size[:2]

    # split images to cubes
    d_grid = len(image_data) + 1 - cube_size[2]
    cubes = np.zeros(np.concatenate(([h_grid * w_grid * d_grid], cube_size), axis=0), dtype=np.float32)
    mapping = np.zeros((h_grid * w_grid * d_grid, 4), dtype=int)
    print(cubes.shape, image_data.shape)
    for j in range(d_grid):
        for k in range(h_grid):
            for l in range(w_grid):
                cubes[j*h_grid*w_grid+k*w_grid+l] = np.moveaxis(image_data[j:j+cube_size[2],
                                                                           k*cube_size[0]:(k+1)*cube_size[0],
                                                                           l*cube_size[1]:(l+1)*cube_size[1]], 0, -1)
                mapping[j*h_grid*w_grid+k*w_grid+l] = [clip_idx, j, k, l]
    if dist_thresh is not None and training_set:
        successive_dist = np.array([np.mean(abs(cubes[i]-cubes[i+1])) for i in range(len(cubes)-1)])
        idx = np.where(successive_dist >= dist_thresh)[0]
        cubes, mapping = cubes[idx], mapping[idx]
        print('new shape:', cubes.shape, image_data.shape)
    np.savez_compressed(saved_cube_file, data=cubes, mapping=mapping)
    return cubes, mapping


def calc_n_cube_in_set(dataset, h, w, cube_size, training_set=True):
    check_path(dataset, cube_size)
    assert h % cube_size[0] == 0
    assert w % cube_size[1] == 0
    h_grid, w_grid = np.array([h, w])//cube_size[:2]
    sequence_n_frame = count_sequence_n_frame(dataset, test=not training_set)
    n_cube = np.sum([((n_frame + 1 - cube_size[2]) * h_grid * w_grid) for n_frame in sequence_n_frame])
    return n_cube


def load_all_cubes_in_set(dataset, h, w, cube_size, training_set=True):
    check_path(dataset, cube_size)
    n_cube_in_set = calc_n_cube_in_set(dataset, h, w, cube_size, training_set=training_set)
    n_clip = dataset['n_clip_train' if training_set else 'n_clip_test']
    #
    cubes = np.zeros(np.concatenate(([n_cube_in_set], cube_size), axis=0), dtype=np.float32)
    mapping = np.zeros((n_cube_in_set, 4), dtype=int)
    idx = 0
    for clip_idx in range(n_clip):
        tmp_cubes, tmp_mapping = split_cubes(dataset, clip_idx, cube_size, training_set=training_set)
        assert len(tmp_cubes) == len(tmp_mapping)
        cubes[idx:idx+len(tmp_cubes)] = tmp_cubes
        mapping[idx:idx+len(tmp_mapping)] = tmp_mapping
        idx += len(tmp_mapping)
    # to work with thresholding motion in training samples
    item_sum = np.array([np.sum(item) for item in cubes])
    idx = np.where(item_sum == 0.0)[0]
    cubes = np.delete(cubes, idx, axis=0)
    mapping = np.delete(mapping, idx, axis=0)
    print(cubes.shape, mapping.shape)
    #
    return cubes, mapping


# get sequence of number of clip's frames
def count_sequence_n_frame(dataset, test=True):
    sequence_n_frame = np.zeros(dataset['n_clip_test' if test else 'n_clip_train'], dtype=int)
    for i in range(len(sequence_n_frame)):
        clip_path = '%s/%s%s/' % (dataset['path_test' if test else 'path_train'], 'Test' if test else 'Train', str(i+1).zfill(3))
        sequence_n_frame[i] = len(sorted(glob.glob(clip_path + '*.tif')))
    return sequence_n_frame


# 1: abnormal, 0: normal
def get_test_frame_labels(dataset, sequence_n_frame, cube_size, is_subway=False):
    ground_truth = dataset['ground_truth']
    assert len(ground_truth) == len(sequence_n_frame)
    labels_select_last = np.zeros(0, dtype=int)
    labels_select_first = np.zeros(0, dtype=int)
    labels_select_mid = np.zeros(0, dtype=int)
    labels_full = np.zeros(0, dtype=int)
    for i in range(len(sequence_n_frame)):
        if not is_subway:
            seg = ground_truth[i]
            # label of full frames
            tmp_labels = np.zeros(sequence_n_frame[i])
            for j in range(0, len(seg), 2):
                tmp_labels[(seg[j]-1):seg[j+1]] = 1
        else:
            tmp_labels = ground_truth[i]
        # label of selected frames
        labels_full = np.append(labels_full, tmp_labels)
        n_removed_frame = cube_size[2] - 1
        labels_select_last = np.append(labels_select_last, tmp_labels[n_removed_frame:])
        labels_select_first = np.append(labels_select_first, tmp_labels[:-n_removed_frame])

        seq_length = sequence_n_frame[i] + 1 - cube_size[2]
        start_idx = cube_size[2]//2
        stop_idx = start_idx + seq_length
        labels_select_mid = np.append(labels_select_mid, tmp_labels[start_idx:stop_idx])
    assert len(np.unique([len(labels_select_last), len(labels_select_first), len(labels_select_mid)])) == 1
    # h_grid, w_grid = np.array(image_size)//cube_size[:2]
    assert len(labels_select_mid) == np.sum([(n_frame + 1 - cube_size[2]) for n_frame in sequence_n_frame])
    write_sequence_to_bin('%s/labels_full.bin' % dataset['path_test'], labels_full)
    return labels_select_last, labels_select_first, labels_select_mid


# POST-PROCESSING
def plot_error_map(dataset, image_size, cube_size, clip_idx, frame_idx, model_idx, score_type_idx=3, using_test_data=True):
    def scale_range(img):
        for i in range(img.shape[-1]):
            img[..., i] = (img[..., i] - np.min(img[..., i]))/(np.max(img[..., i]) - np.min(img[..., i]))
        return img
    # load score maps
    score_appe_maps, score_row_maps, score_col_maps = calc_score_one_clip(dataset, image_size, cube_size,
                                                                          model_idx, clip_idx,
                                                                          train=not using_test_data,
                                                                          force_calc=True)
    if isinstance(frame_idx, int):
        frame_idx = [frame_idx]
    if len(frame_idx) > 16:
        frame_idx = frame_idx[:16]
    score_appe_maps, score_row_maps, score_col_maps = score_appe_maps[frame_idx], score_row_maps[frame_idx], score_col_maps[frame_idx]
    print(score_appe_maps.shape, score_row_maps.shape, score_col_maps.shape)

    # plot
    color_map = 'copper'
    r, c = 6, 8
    fig, axs = plt.subplots(r, c)
    for j in range(c):
        if j in np.arange(len(score_appe_maps)):
            axs[0, j].imshow(scale_range(score_appe_maps[j, :, :, score_type_idx]), cmap=color_map)
            axs[1, j].imshow(scale_range(score_row_maps[j, :, :]), cmap=color_map)
            axs[2, j].imshow(scale_range(score_col_maps[j, :, :]), cmap=color_map)
        if j+c in np.arange(len(score_appe_maps)):
            axs[3, j].imshow(scale_range(score_appe_maps[j+c, :, :, score_type_idx]), cmap=color_map)
            axs[4, j].imshow(scale_range(score_row_maps[j+c, :, :]), cmap=color_map)
            axs[5, j].imshow(scale_range(score_col_maps[j+c, :, :]), cmap=color_map)
        for i in range(r):
            axs[i, j].axis('off')
    plt.show()


# SCORE PROCESSING
def calc_anomaly_score_cube_pair(in_cube, out_cube):
    assert in_cube.shape == out_cube.shape
    loss = (in_cube-out_cube)**2
    # loss = np.sum(loss, axis=-1)  # added
    PSNR = -10*np.log10(np.max(out_cube)**2/np.mean(loss))
    SSIM = ssim(in_cube, out_cube, data_range=np.max([in_cube, out_cube])-np.min([in_cube, out_cube]),
                multichannel=len(in_cube.shape) == 3 and in_cube.shape[-1] > 1)
    return np.array([np.mean(loss), np.max(loss), np.median(loss), np.std(loss), PSNR, SSIM])


def find_cube_idx(mapping, d_idx, h_idx, w_idx):
    tmp = np.absolute(mapping[:, 1:] - np.array([d_idx, h_idx, w_idx]))
    tmp = np.sum(tmp, axis=1)
    idx = np.where(tmp == 0)[0]
    assert len(idx) == 1
    return idx[0]


def calc_anomaly_score_maps_one_clip(in_cubes, in_mapping, out_cubes, out_row_softmax, out_col_softmax, image_size):
    h_grid, w_grid = np.array(image_size)//in_cubes.shape[1:3]
    # calc score for each cube pair
    assert in_cubes.shape == out_cubes.shape
    scores_appe = np.array([calc_anomaly_score_cube_pair(in_cubes[i], out_cubes[i]) for i in range(len(in_cubes))])
    scores_row = np.mean((seq_to_one_hot(in_mapping[:, 2], h_grid) - out_row_softmax)**2, axis=1)
    scores_col = np.mean((seq_to_one_hot(in_mapping[:, 3], w_grid) - out_col_softmax)**2, axis=1)
    assert len(np.unique([len(scores_appe), len(scores_row), len(scores_col)])) == 1
    # arrange scores according to frames
    assert len(np.unique(in_mapping[:, 0])) == 1
    d_values = sorted(np.unique(in_mapping[:, 1]))
    # process each frame
    score_appe_maps = np.zeros((len(d_values), h_grid, w_grid, scores_appe.shape[-1]), dtype=np.float32)
    score_row_maps = np.zeros((len(d_values), h_grid, w_grid), dtype=np.float32)
    score_col_maps = np.zeros((len(d_values), h_grid, w_grid), dtype=np.float32)
    progress = ProgressBar(len(d_values), fmt=ProgressBar.FULL)
    for i in range(len(d_values)):
        progress.current += 1
        progress()
        for j in range(h_grid):
            for k in range(w_grid):
                cube_idx = find_cube_idx(in_mapping, d_values[i], j, k)
                score_appe_maps[i, j, k] = scores_appe[cube_idx]
                score_row_maps[i, j, k] = scores_row[cube_idx]
                score_col_maps[i, j, k] = scores_col[cube_idx]
    progress.done()
    return score_appe_maps, score_row_maps, score_col_maps


def seq_to_one_hot(seq, n_class):
    ret = np.zeros((len(seq), n_class), dtype=np.float32)
    ret[np.arange(len(seq)), seq] = 1.0
    return ret


# suitable for Avenue
def calc_score_one_clip(dataset, image_size, cube_size, epoch, clip_idx, train=False, force_calc=False):
    dataset['cube_dir'] = './training_saver/%s/cube_%d_%d_%d_%d_%d' % \
                          (dataset['name'], image_size[0], image_size[1], cube_size[0], cube_size[1], cube_size[2])

    score_dir = '%s/scores' % dataset['cube_dir']
    saved_data_path = '%s/output_%s/%d_epoch' % (score_dir, 'train' if train else 'test', epoch)
    saved_score_file = '%s/score_epoch_%d_clip_%d.npz' % (saved_data_path, epoch, clip_idx + 1)
    if not force_calc and os.path.isfile(saved_score_file):
        loader = np.load(saved_score_file)
        return loader['appe'], loader['row'], loader['col']

    # load true data
    in_cubes, in_mapping = split_cubes(dataset, clip_idx, cube_size, training_set=train)
    assert len(np.unique(in_mapping[:, 0])) == 1
    print(in_cubes.shape, in_mapping.shape)

    # load outputted data
    score_dir = '%s/scores' % dataset['cube_dir']
    saved_data_path = '%s/output_%s/%d_epoch' % (score_dir, 'train' if train else 'test', epoch)
    saved_data_file = '%s/output_%d.npz' % (saved_data_path, clip_idx)
    out_loader = np.load(saved_data_file)
    out_cubes = out_loader['cube'].astype(np.float32)
    out_row_softmax = out_loader['row'].astype(np.float32)
    out_col_softmax = out_loader['col'].astype(np.float32)
    print(out_cubes.shape, out_row_softmax.shape, out_col_softmax.shape)

    # calc score and save to file
    score_appe_maps, score_row_maps, score_col_maps = \
        calc_anomaly_score_maps_one_clip(in_cubes, in_mapping, out_cubes, out_row_softmax, out_col_softmax, image_size)
    np.savez_compressed(saved_score_file, appe=score_appe_maps, row=score_row_maps, col=score_col_maps)
    return score_appe_maps, score_row_maps, score_col_maps


def calc_score_full_clips(dataset, image_size, cube_size, epoch, train=False, force_calc=False):
    dataset['cube_dir'] = './training_saver/%s/cube_%d_%d_%d_%d_%d' % \
                          (dataset['name'], image_size[0], image_size[1], cube_size[0], cube_size[1], cube_size[2])
    score_dir = '%s/scores' % dataset['cube_dir']
    saved_data_path = '%s/output_%s/%d_epoch' % (score_dir, 'train' if train else 'test', epoch)
    saved_score_file = '%s/score_epoch_%d_full.npz' % (saved_data_path, epoch)

    if os.path.isfile(saved_score_file) and not force_calc:
        loader = np.load(saved_score_file)
        return loader['appe'], loader['row'], loader['col']

    # calc scores for all clips and save to file
    n_clip = dataset['n_clip_train' if train else 'n_clip_test']
    print('training set' if train else 'test set')
    for i in range(n_clip):
        if i == 0:
            score_appe_maps, score_row_maps, score_col_maps = \
                calc_score_one_clip(dataset, image_size, cube_size, epoch, i, train=train, force_calc=force_calc)
        else:
            tmp_score_appe, tmp_score_row, tmp_score_col = \
                calc_score_one_clip(dataset, image_size, cube_size, epoch, i, train=train, force_calc=force_calc)
            score_appe_maps = np.concatenate((score_appe_maps, tmp_score_appe), axis=0)
            score_row_maps = np.concatenate((score_row_maps, tmp_score_row), axis=0)
            score_col_maps = np.concatenate((score_col_maps, tmp_score_col), axis=0)

    np.savez_compressed(saved_score_file, appe=score_appe_maps, row=score_row_maps, col=score_col_maps)
    return score_appe_maps, score_row_maps, score_col_maps


def score_maps_to_score_seq(score_maps, operation, max_avg_patch_size=None):
    assert operation in [np.mean, np.min, np.max, np.median, np.std]
    if not max_avg_patch_size:
        return np.array([operation(score_map, axis=(0, 1)) for score_map in score_maps])
    if len(score_maps.shape) == 4:
        return np.array([np.max(cv2.blur(score_map[..., 1], (max_avg_patch_size, max_avg_patch_size))[1:-1, 1:-1]) for score_map in score_maps])
    return np.array([np.max(cv2.blur(score_map, (max_avg_patch_size, max_avg_patch_size))[1:-1, 1:-1]) for score_map in score_maps])


def get_weights(dataset, image_size, cube_size, epoch, operation, save_as_image=False):
    score_appe_maps, score_row_maps, score_col_maps = \
        calc_score_full_clips(dataset, image_size, cube_size, epoch, train=True, force_calc=False)
    # score_appe_seq = score_maps_to_score_seq(score_appe_maps, operation)
    # score_row_seq = score_maps_to_score_seq(score_row_maps, operation)
    # score_col_seq = score_maps_to_score_seq(score_col_maps, operation)
    appe_weight, row_weight, col_weight = np.mean(score_appe_maps, axis=0)[..., 1], np.mean(score_row_maps, axis=0), np.mean(score_col_maps, axis=0)

    if save_as_image:
        from custom_cmap import parula_map
        print('shape:', appe_weight.shape, row_weight.shape, col_weight.shape)
        print('min:', np.min(appe_weight), np.min(row_weight), np.min(col_weight))
        print('max:', np.max(appe_weight), np.max(row_weight), np.max(col_weight))
        cm = plt.get_cmap(parula_map())
        norm_appe_weight = cm((appe_weight-np.min(appe_weight))/(np.max(appe_weight)-np.min(appe_weight)))
        norm_row_weight = cm((row_weight-np.min(row_weight))/(np.max(row_weight)-np.min(row_weight)))
        norm_col_weight = cm((col_weight-np.min(col_weight))/(np.max(col_weight)-np.min(col_weight)))
        #
        norm_appe_weight_save = cv2.resize(norm_appe_weight*255, (0, 0), fx=80, fy=80, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('plot_sequence/weight_%s_appe.png' % dataset['name'],
                    np.dstack((norm_appe_weight_save[..., 2], norm_appe_weight_save[..., 1], norm_appe_weight_save[..., 0])))
        norm_row_weight_save = cv2.resize(norm_row_weight*255, (0, 0), fx=80, fy=80, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('plot_sequence/weight_%s_row.png' % dataset['name'],
                    np.dstack((norm_row_weight_save[..., 2], norm_row_weight_save[..., 1], norm_row_weight_save[..., 0])))
        norm_col_weight_save = cv2.resize(norm_col_weight*255, (0, 0), fx=80, fy=80, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('plot_sequence/weight_%s_col.png' % dataset['name'],
                    np.dstack((norm_col_weight_save[..., 2], norm_col_weight_save[..., 1], norm_col_weight_save[..., 0])))
        #
        plt.subplot(1, 3, 1)
        plt.imshow(norm_appe_weight)
        plt.subplot(1, 3, 2)
        plt.imshow(norm_row_weight)
        plt.subplot(1, 3, 3)
        plt.imshow(norm_col_weight)
        plt.show()

    return appe_weight, row_weight, col_weight


# ASSESSMENT
def basic_assess_AUC(scores, labels, plot_pr_idx=None):
    print(len(scores), len(labels))
    assert len(scores) == len(labels)
    auc, eer, eer_expected = assessment(scores, labels)[:3]
    if plot_pr_idx is not None:
        precision, recall, _ = precision_recall_curve(labels, scores[:, plot_pr_idx])
        print(len(np.where(labels == 0)[0]), len(np.where(labels == 1)[0]), len(np.unique(precision)), len(np.unique(recall)))
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.show()
    return auc, eer, eer_expected, average_precision_score(labels, scores)


def full_assess_AUC(dataset, image_size, cube_size, model_idx, frame_labels, sequence_n_frame=None, plot_pr_idx=None,
                    selected_score_estimation_way=3, operation=np.std, show=True, save_roc_pr=False):
    way_names = ['mean', 'max', 'median', 'std', 'PSNR', 'SSIM']
    # load scores and calc weights (all converted to sequence)
    score_appe_maps, score_row_maps, score_col_maps = \
        calc_score_full_clips(dataset, image_size, cube_size, model_idx, train=False, force_calc=False)
    w_appe, w_row, w_col = get_weights(dataset, image_size, cube_size, model_idx, operation)
    if show:
        print('====================== %s =====================' % way_names[selected_score_estimation_way])

    # only consider selected score types
    score_appe_maps = score_appe_maps[..., selected_score_estimation_way]
    if show:
        print('weights:', w_appe.shape, w_row.shape, w_col.shape)
        print(np.min(w_appe), np.max(w_appe), np.min(w_row), np.max(w_row), np.min(w_col), np.max(w_col))

    # calc combined score maps
    # w_appe, w_row, w_col = 1, 5, 0.5
    score_full_maps = np.array([(score_appe_maps[i]**0.5 * (1-w_appe) + score_row_maps[i]**1 * (1-w_row) + score_col_maps[i]**1 * (1-w_col))
                                for i in range(len(score_appe_maps))])
    score_clf_maps = np.array([(score_row_maps[i]**1 * (1-w_row) + score_col_maps[i]**1 * (1-w_col))
                               for i in range(len(score_appe_maps))])

    # calc frame-level scores
    score_appe_seq = score_maps_to_score_seq(score_appe_maps, operation)
    score_row_seq = score_maps_to_score_seq(score_row_maps, operation)
    score_col_seq = score_maps_to_score_seq(score_col_maps, operation)
    score_clf_seq = score_maps_to_score_seq(score_clf_maps, operation)
    score_full_seq = score_maps_to_score_seq(score_full_maps, operation)

    if not show:
        return score_appe_seq, score_row_seq, score_col_seq, score_full_seq

    # split to clips
    if sequence_n_frame is not None and dataset['name'] not in ['Exit', 'Entrance']:
        accumulated_n_frame = np.cumsum(sequence_n_frame - cube_size[2] + 1)[:-1]
        score_appe_seq = np.split(score_appe_seq, accumulated_n_frame, axis=0)
        score_row_seq = np.split(score_row_seq, accumulated_n_frame, axis=0)
        score_col_seq = np.split(score_col_seq, accumulated_n_frame, axis=0)
        score_clf_seq = np.split(score_clf_seq, accumulated_n_frame, axis=0)
        score_full_seq = np.split(score_full_seq, accumulated_n_frame, axis=0)
        # normalize score in each clip
        norm_method = 1
        if norm_method == 1:
            score_appe_seq = [item/np.max(item, axis=0) for item in score_appe_seq]
            score_row_seq = [item/np.max(item, axis=0) for item in score_row_seq]
            score_col_seq = [item/np.max(item, axis=0) for item in score_col_seq]
            score_clf_seq = [item/np.max(item, axis=0) for item in score_clf_seq]
            score_full_seq = [item/np.max(item, axis=0) for item in score_full_seq]
        else:
            print('WARNING: norm_method should be 1')
            score_appe_seq = [(item-np.min(item, axis=0))/(np.max(item, axis=0)-np.min(item, axis=0))
                              for item in score_appe_seq]
            score_row_seq = [(item-np.min(item, axis=0))/(np.max(item, axis=0)-np.min(item, axis=0))
                             for item in score_row_seq]
            score_col_seq = [(item-np.min(item, axis=0))/(np.max(item, axis=0)-np.min(item, axis=0))
                             for item in score_col_seq]
            score_clf_seq = [(item-np.min(item, axis=0))/(np.max(item, axis=0)-np.min(item, axis=0))
                             for item in score_clf_seq]
            score_full_seq = [(item-np.min(item, axis=0))/(np.max(item, axis=0)-np.min(item, axis=0))
                              for item in score_full_seq]

        # test if keyframe assessment is better
        perform_check = not True
        if perform_check:
            frame_labels = np.split(frame_labels, accumulated_n_frame, axis=0)
            score_appe_seq = np.array([score_appe_seq[i][np.arange(1, len(score_appe_seq[i]), cube_size[-1])]
                                      for i in range(len(score_appe_seq))])
            score_row_seq = np.array([score_row_seq[i][np.arange(1, len(score_row_seq[i]), cube_size[-1])]
                                     for i in range(len(score_row_seq))])
            score_col_seq = np.array([score_col_seq[i][np.arange(1, len(score_col_seq[i]), cube_size[-1])]
                                     for i in range(len(score_col_seq))])
            score_clf_seq = np.array([score_clf_seq[i][np.arange(1, len(score_clf_seq[i]), cube_size[-1])]
                                     for i in range(len(score_clf_seq))])
            score_full_seq = np.array([score_full_seq[i][np.arange(1, len(score_full_seq[i]), cube_size[-1])]
                                      for i in range(len(score_full_seq))])
            frame_labels = np.array([frame_labels[i][np.arange(1, len(frame_labels[i]), cube_size[-1])]
                                    for i in range(len(frame_labels))])
            frame_labels = np.concatenate(frame_labels, axis=0)
        # concatenate again
        score_appe_seq = np.concatenate(score_appe_seq, axis=0)
        score_row_seq = np.concatenate(score_row_seq, axis=0)
        score_col_seq = np.concatenate(score_col_seq, axis=0)
        score_clf_seq = np.concatenate(score_clf_seq, axis=0)
        score_full_seq = np.concatenate(score_full_seq, axis=0)

    if dataset['name'] == 'Exit':
        n_remove = np.sum(sequence_n_frame[:5] - cube_size[2] + 1)*0
        score_appe_seq = score_appe_seq[n_remove:]
        score_row_seq = score_row_seq[n_remove:]
        score_col_seq = score_col_seq[n_remove:]
        score_full_seq = score_full_seq[n_remove:]
        frame_labels = frame_labels[n_remove:]

    print(score_appe_seq.shape, score_row_seq.shape, score_col_seq.shape, score_clf_seq.shape, score_full_seq.shape)

    auc, eer, eer_expected, prc = basic_assess_AUC(score_appe_seq, frame_labels, plot_pr_idx=plot_pr_idx) \
        if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('appearance AUCs: %.3f (%.3f, %.3f), PRscore: %.3f' % (auc, eer, eer_expected, prc))

    auc, eer, eer_expected, prc = basic_assess_AUC(score_row_seq, frame_labels, plot_pr_idx=plot_pr_idx) \
        if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('row index  AUCs: %.3f (%.3f, %.3f), PRscore: %.3f' % (auc, eer, eer_expected, prc))

    auc, eer, eer_expected, prc = basic_assess_AUC(score_col_seq, frame_labels, plot_pr_idx=plot_pr_idx) \
        if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('col index  AUCs: %.3f (%.3f, %.3f), PRscore: %.3f' % (auc, eer, eer_expected, prc))

    auc, eer, eer_expected, prc = basic_assess_AUC(score_clf_seq, frame_labels, plot_pr_idx=plot_pr_idx) \
        if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('clf index  AUCs: %.3f (%.3f, %.3f), PRscore: %.3f' % (auc, eer, eer_expected, prc))

    auc, eer, eer_expected, prc = basic_assess_AUC(score_full_seq, frame_labels, plot_pr_idx=plot_pr_idx) \
        if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('combinatio AUCs: %.3f (%.3f, %.3f), PRscore: %.3f' % (auc, eer, eer_expected, prc))

    if save_roc_pr:
        print('Saving ROC-PR results...')
        # ROC
        fpr_1, tpr_1, _ = roc_curve(frame_labels, score_appe_seq)
        roc_appe = [fpr_1, tpr_1]
        fpr_2, tpr_2, _ = roc_curve(frame_labels, score_row_seq)
        roc_row = [fpr_2, tpr_2]
        fpr_3, tpr_3, _ = roc_curve(frame_labels, score_col_seq)
        roc_col = [fpr_3, tpr_3]
        fpr_4, tpr_4, _ = roc_curve(frame_labels, score_clf_seq)
        roc_clf = [fpr_4, tpr_4]
        fpr_5, tpr_5, _ = roc_curve(frame_labels, score_full_seq)
        roc_full = [fpr_5, tpr_5]
        # PR
        p_1, r_1, _ = precision_recall_curve(frame_labels, score_appe_seq)
        pr_appe = [p_1, r_1]
        p_2, r_2, _ = precision_recall_curve(frame_labels, score_row_seq)
        pr_row = [p_2, r_2]
        p_3, r_3, _ = precision_recall_curve(frame_labels, score_col_seq)
        pr_col = [p_3, r_3]
        p_4, r_4, _ = precision_recall_curve(frame_labels, score_clf_seq)
        pr_clf = [p_4, r_4]
        p_5, r_5, _ = precision_recall_curve(frame_labels, score_full_seq)
        pr_full = [p_5, r_5]
        #
        dataset['cube_dir'] = './training_saver/%s/cube_%d_%d_%d_%d_%d' % \
                              (dataset['name'], image_size[0], image_size[1], cube_size[0], cube_size[1], cube_size[2])
        score_dir = '%s/scores' % dataset['cube_dir']
        saved_data_path = '%s/output_test/%d_epoch' % (score_dir, model_idx)
        out_file = '%s/%s_curves.mat' % (saved_data_path, dataset['name'])

        out_data = {'roc_appe': roc_appe, 'roc_row': roc_row, 'roc_col': roc_col, 'roc_clf': roc_clf, 'roc_full': roc_full,
                    'pr_appe': pr_appe, 'pr_row': pr_row, 'pr_col': pr_col, 'pr_clf': pr_clf, 'pr_full': pr_full}
        savemat(out_file, out_data)
        print('Curves saved!')

    # pixel-level AUC
    if dataset['name'] in ['UCSDped1-XX', 'UCSDped2-XX']:
        gt_masks = get_pixel_gt(dataset, cube_size, count_sequence_n_frame(dataset, test=True), select_frame='mid')
        auc_appe = pixel_wise_assessment(score_appe_maps, gt_masks, image_size, frame_labels)
        print('pixel-level AUC (appe): %.3f' % auc_appe)
        auc_row = pixel_wise_assessment(score_row_maps, gt_masks, image_size, frame_labels)
        print('pixel-level AUC (row) : %.3f' % auc_row)
        auc_col = pixel_wise_assessment(score_col_maps, gt_masks, image_size, frame_labels)
        print('pixel-level AUC (col) : %.3f' % auc_col)
        auc_full = pixel_wise_assessment(score_full_maps, gt_masks, image_size, frame_labels)
        print('pixel-level AUC (full): %.3f' % auc_full)

    # patch-level PR
    if dataset['name'] in ['Boat']:
        gt_masks = get_patch_gt(dataset, cube_size, count_sequence_n_frame(dataset, test=True), select_frame='mid')
        pr_calc = not False
        patch_wise_assessment(score_appe_maps, gt_masks, pr_calc=pr_calc, title='Appearance')
        patch_wise_assessment(score_row_maps, gt_masks, pr_calc=pr_calc, title='Row')
        patch_wise_assessment(score_col_maps, gt_masks, pr_calc=pr_calc, title='Column')
        patch_wise_assessment(score_full_maps, gt_masks, pr_calc=pr_calc, title='Full')


def write_video_result(dataset, image_size, cube_size, clip_idx, model_idx, train=False, operation=np.std, frame_gt=None,
                       show_all_score=False, frame_range=None, scale_video=True, show_clf=True, save_plot_exam_only=False):
    from custom_cmap import parula_map
    from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

    def plot_score_sequence(frames, seq, frame_gt, start_idx, end_idx, out_file, frame_to_vis=None, pos=None, frame_zoom=None):
        if frame_to_vis is not None:
            assert len(frames) == len(frame_to_vis)
        if pos is not None:
            assert len(pos) == len(frame_to_vis)
        x = np.arange(start_idx, end_idx)
        y = seq[start_idx:end_idx]
        fig, ax = plt.subplots()
        ax.plot(x, y, 'b')
        ax.set_ylim([-0.03, 1.05])
        ax.set_xlabel('frame index')
        ax.set_ylabel('frame-level score')
        if frame_gt is not None:
            for idx in range(0, len(frame_gt), 2):
                ax.axvspan(frame_gt[idx], np.amin([frame_gt[idx+1], end_idx]), facecolor='r', alpha=0.35)
        ax.set_xlim([start_idx, end_idx])
        #
        pos_idx = 0
        if frame_to_vis is not None:
            for idx in range(len(frame_to_vis)):
                img = frames[idx]
                if len(img.shape) == 2:
                    img = np.dstack((img, img, img))
                print(img.shape)

                if frame_zoom is not None:
                    ab = AnnotationBbox(OffsetImage(img, zoom=frame_zoom), (frame_to_vis[idx], seq[frame_to_vis[idx]]),
                                        xybox=(frame_to_vis[idx]+10, seq[frame_to_vis[idx]]-0.2*(1 if pos is None else pos[pos_idx])),
                                        xycoords='data', pad=-1, arrowprops=dict(arrowstyle='fancy'))
                    ax.add_artist(ab)
                else:
                    # -120 for Avenue, +2 for remaining
                    ax.annotate(str(frame_to_vis[idx]), xy=(frame_to_vis[idx], seq[frame_to_vis[idx]]),
                                xytext=(frame_to_vis[idx]+2, seq[frame_to_vis[idx]]-0.2*(1 if pos is None else pos[pos_idx])),
                                arrowprops=dict(arrowstyle='fancy', facecolor='g'))
                    cv2.imwrite('%s_%d.png' % (out_file[:-4], frame_to_vis[idx]),
                                (np.dstack([img[..., 2], img[..., 1], img[..., 0]])*255).astype(int))
                pos_idx += 1
        #
        if out_file is not None:
            plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
        plt.show()

    def scale_range(img, new_size=(160, 120), get_only_top=None, val_to_show=None, skip=False):
        cm = plt.get_cmap(parula_map())  # default:viridis
        # cm = plt.get_cmap('GnBu_r')
        if get_only_top:
            min_allowed = np.unique(img)[-get_only_top]
            img[img < min_allowed] = 0
        if val_to_show and get_only_top == 1:
            img[img == np.max(img)] = val_to_show
        if len(img.shape) == 3:
            if not skip:
                for i in range(img.shape[-1]):
                    img[..., i] = (img[..., i] - np.min(img[..., i]))/(np.max(img[..., i]) - np.min(img[..., i]))
        else:
            if not skip:
                img = (img - np.min(img))/(np.max(img) - np.min(img))
            img = cm(img)[..., :3]
        if new_size:
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return img

    saved_video_path = './video_results/%s/results_%s/%d_epoch' % (dataset['name'], 'train' if train else 'test', model_idx)
    if not os.path.exists(saved_video_path):
        pathlib.Path(saved_video_path).mkdir(parents=True, exist_ok=True)

    # load score data
    score_appe_maps, score_row_maps, score_col_maps = \
        calc_score_one_clip(dataset, image_size, cube_size, model_idx, clip_idx, train=train, force_calc=False)
    print('maps:', score_appe_maps.shape, score_row_maps.shape, score_col_maps.shape)
    w_appe, w_row, w_col = get_weights(dataset, image_size, cube_size, model_idx, operation)
    print('weights:', w_appe.shape, w_row.shape, w_col.shape)
    way = 1
    score_appe_maps = score_appe_maps[..., way]

    score_clf_maps = np.array([(score_row_maps[i]**1 * (1-w_row) + score_col_maps[i]**1 * (1-w_col)) for i in range(len(score_row_maps))])
    score_clf_seq = score_maps_to_score_seq(score_clf_maps, operation)
    score_clf_seq /= np.max(score_clf_seq)
    print(score_clf_seq.shape)

    score_full_maps = np.array([(score_appe_maps[i]**0.5 * (1-w_appe) + score_row_maps[i]**1 * (1-w_row) + score_col_maps[i]**1 * (1-w_col))
                               for i in range(len(score_appe_maps))])
    score_full_seq = score_maps_to_score_seq(score_full_maps, operation)
    score_full_seq /= np.max(score_full_seq)
    print(score_full_seq.shape)

    score_appe_seq = score_maps_to_score_seq(score_appe_maps, operation)
    score_appe_seq /= np.max(score_appe_seq)
    score_row_seq = score_maps_to_score_seq(score_row_maps, operation)
    score_row_seq /= np.max(score_row_seq)
    score_col_seq = score_maps_to_score_seq(score_col_maps, operation)
    score_col_seq /= np.max(score_col_seq)

    # video writer
    saved_video_file = '%s/clip_%d%s%s.avi' % (saved_video_path, clip_idx, '' if show_all_score else '_realtime', '_scaled' if scale_video else '')
    from skvideo.io import FFmpegWriter as VideoWriter
    out = VideoWriter(saved_video_file, outputdict={'-vcodec': 'libx264', '-b': '300000000'})

    # load image data
    saved_image_file = '%s/%s_image_clip_%d.npz' % (dataset['cube_dir'], 'training' if train else 'test', clip_idx + 1)
    if not os.path.isfile(saved_image_file):
        print('image file not found! (%s)' % saved_image_file)

    image_data = np.load(saved_image_file)['image']
    print('loaded image data:', image_data.shape)
    image_data = image_data[:len(score_full_seq)]
    image_data = np.array([np.dstack((img, img, img)) for img in image_data])
    print('image data:', image_data.shape)
    if frame_gt is not None:
        print('ground truth:', frame_gt)

    if scale_video:
        score_appe_maps /= np.max(score_appe_maps)
        score_row_maps /= np.max(score_row_maps)
        score_col_maps /= np.max(score_col_maps)
        score_clf_maps /= np.max(score_clf_maps)
        score_full_maps /= np.max(score_full_maps)

    #
    if save_plot_exam_only:
        out_file = 'plot_sequence/%s_clip_%d_%s.pdf' % (dataset['name'], clip_idx, 'Rxy' if dataset['name'] in ['UCSDped2', 'Avenue'] else 'xy')
        if dataset['name'] == 'UCSDped2':
            if clip_idx == 3:
                indices = [16, 70, 136]
                orig_imgs = load_images_single_clip(dataset, clip_idx, indices, train=train)
                plot_score_sequence(orig_imgs, score_full_seq, frame_gt, 0, len(score_full_seq), out_file, frame_to_vis=indices, pos=None)
        elif dataset['name'] == 'Avenue':
            if clip_idx in (0, 5):
                indices = [200, 600, 950, 1200] if clip_idx == 0 else [147, 421, 973, 1116]
                orig_imgs = load_images_single_clip(dataset, clip_idx, indices, train=train)
                plot_score_sequence(orig_imgs, score_full_seq, frame_gt, 0, len(score_full_seq), out_file, frame_to_vis=indices,
                                    pos=[-1, -1, 1, 1] if clip_idx == 0 else [1, 1, -1, -1])
        elif dataset['name'] == 'Train' and clip_idx == 0:
            # plot_score_sequence(orig_imgs[:-2],score_full_seq, frame_gt, 2100, 3200, out_file, frame_to_vis=[2233,2565,3033],pos=[2.7,3,1.6])
            indices = [2258, 2590, 3033]
            orig_imgs = load_images_single_clip(dataset, clip_idx, indices, train=train)
            plot_score_sequence(orig_imgs, score_clf_seq, frame_gt, 2100, 3200, out_file, frame_to_vis=indices, pos=[2.7, 3, 1.6])
        elif dataset['name'] == 'Belleview' and clip_idx == 0:
            # plot_score_sequence(orig_imgs[:-2],score_full_seq, frame_gt, 50, 443+157, out_file, frame_to_vis=[103,241,431,490],pos=[-2,2,1.4,-1.2])
            indices = [103, 234, 513]
            orig_imgs = load_images_single_clip(dataset, clip_idx, indices, train=train)
            plot_score_sequence(orig_imgs, score_clf_seq, frame_gt, 50, 443+157, out_file, frame_to_vis=indices, pos=[-2, -1.7, -1.4])
        return

    #
    idx_start, idx_end = 0, len(score_full_seq)
    if frame_range is not None:
        idx_start = frame_range[0]
        idx_end = frame_range[1]
    print('write video from frame %d to %d' % (idx_start, idx_end))
    # draw chart and frame
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    get_only_top = None
    for i in range(idx_start, idx_end):
        #
        datum_appe = np.zeros((480, 210, 3), dtype=np.uint8)
        datum_appe[24:24+120, 10:10+160] = image_data[i]*255
        cv2.putText(datum_appe, 'Input frame %s' % ('(grayscale)' if dataset['name'] in ['Avenue', 'Train'] else ''),
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), lineType=cv2.LINE_AA)
        datum_appe[176:176+120, 10:10+160] = scale_range(score_appe_maps[i], val_to_show=score_appe_seq[i],
                                                         get_only_top=get_only_top, skip=scale_video)*255
        cv2.putText(datum_appe, 'Reconstruction score map S_R', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), lineType=cv2.LINE_AA)
        datum_appe[320:320+120, 10:10+160] = scale_range(score_row_maps[i], val_to_show=score_row_seq[i], skip=scale_video)*255
        cv2.putText(datum_appe, 'Classification score map S_y', (10, 314), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), lineType=cv2.LINE_AA)
        #
        datum_flow = np.zeros((480, 210, 3), dtype=np.uint8)
        datum_flow[24:24+120, 10:10+160] = image_data[i]*255*0.7 + \
            scale_range(score_full_maps[i] if dataset['name'] in ['Avenue', 'UCSDped2'] else score_clf_maps[i],
                        get_only_top=None,
                        val_to_show=score_full_seq[i] if dataset['name'] in ['Avenue', 'UCSDped2'] else score_clf_seq[i],
                        skip=scale_video)*255*0.3 + 0

        cv2.putText(datum_flow, 'Superimposed to %s' % ('S_R,x,y' if not show_clf else 'S_x,y'), (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), lineType=cv2.LINE_AA)
        datum_flow[176:176+120, 10:10+160] = \
            scale_range(score_full_maps[i] if dataset['name'] in ['Avenue', 'UCSDped2'] else score_clf_maps[i], get_only_top=None,
                        val_to_show=score_full_seq[i] if dataset['name'] in ['Avenue', 'UCSDped2'] else score_clf_seq[i],
                        skip=scale_video)*255

        cv2.putText(datum_flow, 'Combined score map %s' % ('S_R,x,y' if not show_clf else 'S_x,y'), (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), lineType=cv2.LINE_AA)
        datum_flow[320:320+120, 10:10+160] = scale_range(score_col_maps[i], val_to_show=score_col_seq[i], skip=scale_video)*255
        cv2.putText(datum_flow, 'Classification score map S_x', (10, 314), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), lineType=cv2.LINE_AA)
        #
        fig = Figure(dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        if show_all_score:
            if show_clf:
                ax.plot(np.arange(idx_start, idx_end), score_clf_seq[idx_start:idx_end], linewidth=2.5)  # 4
            else:
                ax.plot(np.arange(idx_start, idx_end), score_full_seq[idx_start:idx_end], linewidth=2.5)  # 4
        else:
            if show_clf:
                ax.plot(np.arange(idx_start, i+1), score_clf_seq[idx_start:i+1], linewidth=2.5)
            else:
                ax.plot(np.arange(idx_start, i+1), score_full_seq[idx_start:i+1], linewidth=2.5)
            ax.set_ylim([-0.03, 1.05])
            ax.set_xlim([idx_start, idx_end])
        ax.set_title('Score estimated from %s' % ('$\mathcal{S}_{R,x,y}$' if not show_clf else '$\mathcal{S}_{x,y}$'))
        ax.set_xlabel('frame index')
        ax.set_ylabel('frame-level score')
        ax.axvline(x=i, c='r')
        if frame_gt is not None:
            for idx in range(0, len(frame_gt), 2):
                ax.axvspan(frame_gt[idx], np.amin([frame_gt[idx+1], idx_end]), facecolor='r', alpha=0.35)
        canvas.draw()
        s, (w_chart, h_chart) = canvas.print_to_buffer()
        # print(w_chart, h_chart)
        img_chart = np.fromstring(s, np.uint8).reshape((h_chart, w_chart, 4))[..., :3]
        if i == -2:
            plt.imshow(img_chart)
            plt.show()
        out.writeFrame(np.concatenate((np.concatenate((datum_appe, datum_flow), axis=1), img_chart), axis=1))
    #
    out.close()


def write_example(dataset, image_size, cube_size, clip_idx, model_idx, operation=np.std, scale_video=True, wrapall=True):
    from custom_cmap import parula_map

    def scale_range(img, new_size=(160, 120), skip=False):
        cm = plt.get_cmap(parula_map())
        # cm = plt.get_cmap('viridis')
        if len(img.shape) == 3:
            if not skip:
                for i in range(img.shape[-1]):
                    img[..., i] = (img[..., i] - np.min(img[..., i]))/(np.max(img[..., i]) - np.min(img[..., i]))
        else:
            if not skip:
                img = (img - np.min(img))/(np.max(img) - np.min(img))
            img = cm(img)[..., :3]
        if new_size:
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return img

    saved_video_path = './video_results/%s/results_test/%d_epoch/examples' % (dataset['name'], model_idx)
    if not os.path.exists(saved_video_path):
        pathlib.Path(saved_video_path).mkdir(parents=True, exist_ok=True)

    # load score data
    score_appe_maps, score_row_maps, score_col_maps = \
        calc_score_one_clip(dataset, image_size, cube_size, model_idx, clip_idx, train=False, force_calc=False)
    print('maps:', score_appe_maps.shape, score_row_maps.shape, score_col_maps.shape)
    w_appe, w_row, w_col = get_weights(dataset, image_size, cube_size, model_idx, operation)
    print('weights:', w_appe.shape, w_row.shape, w_col.shape)
    way = 1
    score_appe_maps = score_appe_maps[..., way]

    score_full_maps = np.array([(score_appe_maps[i]**0.5 * (1-w_appe) + score_row_maps[i]**1 * (1-w_row) + score_col_maps[i]**1 * (1-w_col))
                               for i in range(len(score_appe_maps))])
    score_clf_maps = np.array([(score_row_maps[i]**1 * (1-w_row) + score_col_maps[i]**1 * (1-w_col)) for i in range(len(score_appe_maps))])
    score_full_seq = score_maps_to_score_seq(score_full_maps, operation)
    score_clf_seq = score_maps_to_score_seq(score_clf_maps, operation)
    score_full_seq /= np.max(score_full_seq)
    score_clf_seq /= np.max(score_clf_seq)
    print(score_full_seq.shape, score_clf_seq.shape)

    # load image data
    saved_image_file = '%s/test_image_clip_%d.npz' % (dataset['cube_dir'], clip_idx + 1)
    if not os.path.isfile(saved_image_file):
        print('image file not found! (%s)' % saved_image_file)

    image_data = np.load(saved_image_file)['image']
    print('loaded image data:', image_data.shape)
    image_data = image_data[:len(score_full_seq)]
    image_data = np.array([np.dstack((img, img, img)) for img in image_data])
    print('image data:', image_data.shape)

    if scale_video:
        score_appe_maps /= np.max(score_appe_maps)
        score_row_maps /= np.max(score_row_maps)
        score_col_maps /= np.max(score_col_maps)
        score_clf_maps /= np.max(score_clf_maps)
        score_full_maps /= np.max(score_full_maps)

    # draw chart and frame
    if dataset['name'] == 'UCSDped2':
        selected_idx = 136
    elif dataset['name'] == 'Avenue':
        selected_idx = 972
    elif dataset['name'] == 'Belleview':
        selected_idx = 135
    elif dataset['name'] == 'Train':
        selected_idx = 2734
    else:
        selected_idx = 0
        print('WARNING: unknown dataset')
    in_frame = image_data[selected_idx]*255
    recon_map = scale_range(score_appe_maps[selected_idx], skip=scale_video)*255
    row_map = scale_range(score_row_maps[selected_idx], skip=scale_video)*255
    col_map = scale_range(score_col_maps[selected_idx], skip=scale_video)*255
    clf_map = scale_range(score_clf_maps[selected_idx], skip=scale_video)*255
    full_map = scale_range(score_full_maps[selected_idx], skip=scale_video)*255
    if wrapall:
        w0 = 0.6
        recon_map = in_frame*w0 + recon_map*(1-w0) + 0
        row_map = in_frame*w0 + row_map*(1-w0) + 0
        col_map = in_frame*w0 + col_map*(1-w0) + 0
        clf_map = in_frame*w0 + clf_map*(1-w0) + 0
        full_map = in_frame*w0 + full_map*(1-w0) + 0
    cv2.imwrite('%s/%s_in_frame.png' % (saved_video_path, dataset['name']), in_frame)
    cv2.imwrite('%s/%s_R_map.png' % (saved_video_path, dataset['name']), recon_map)
    cv2.imwrite('%s/%s_x_map.png' % (saved_video_path, dataset['name']), col_map)
    cv2.imwrite('%s/%s_y_map.png' % (saved_video_path, dataset['name']), row_map)
    cv2.imwrite('%s/%s_clf_map.png' % (saved_video_path, dataset['name']), clf_map)
    cv2.imwrite('%s/%s_full_map.png' % (saved_video_path, dataset['name']), full_map)

    plt.subplot(2, 3, 1)
    plt.imshow(recon_map/255.)
    plt.subplot(2, 3, 2)
    plt.imshow(row_map/255.)
    plt.subplot(2, 3, 3)
    plt.imshow(col_map/255.)
    plt.subplot(2, 3, 4)
    plt.imshow(full_map/255.)
    plt.subplot(2, 3, 5)
    plt.imshow(clf_map/255.)
    plt.show()
    #


def manual_assess_AUC(dataset, image_size, cube_size, model_idx, frame_labels, plot_pr_idx=None,
                      selected_score_estimation_way=3, operation=np.std):

    def basic_assess_AUC_2(score_maps, labels, plot_pr_idx=None):
        def calc_auc_from_tpr_fpr(tpr, fpr):
            idx = np.argsort(fpr)
            tpr, fpr = tpr[idx], fpr[idx]
            if fpr[0] > 0:
                fpr = np.insert(fpr, 0, 0)
                tpr = np.insert(tpr, 0, 0)
            if fpr[-1] < 1:
                fpr = np.append(fpr, 1)
                tpr = np.append(tpr, 1)
            auc = np.sum([0.5*(tpr[i+1]+tpr[i])*(fpr[i+1]-fpr[i]) for i in range(len(fpr)-1)])
            return auc

        def calc_apr_from_ppv_tpr(ppv, tpr):
            idx = np.argsort(tpr)
            tpr, ppv = tpr[idx], ppv[idx]
            average_precision = np.sum((tpr[1:]-tpr[:-1])*ppv[1:])
            return average_precision

        print(len(score_maps), len(labels))
        assert len(score_maps) == len(labels)

        P = len(np.where(labels == 1)[0])
        N = len(np.where(labels == 0)[0])

        threshs = np.unique(score_maps)
        print('no. of thresholds:', len(threshs))

        threshs = threshs[np.arange(0, len(threshs), 100//2)]
        progress = ProgressBar(len(threshs), fmt=ProgressBar.FULL)
        tpr, fpr, ppv = [], [], []
        for thresh in threshs:
            progress.current += 1
            progress()

            label_pred = np.array([np.any(score_map >= thresh).astype(int) for score_map in score_maps])
            TP = len(np.intersect1d(np.where(label_pred == 1)[0], np.where(labels == 1)[0]))
            FP = len(np.intersect1d(np.where(label_pred == 1)[0], np.where(labels == 0)[0]))
            tpr.append(TP/P)
            fpr.append(FP/N)
            ppv.append(TP/(TP+FP))

        progress.done()
        tpr, fpr, ppv = np.array(tpr), np.array(fpr), np.array(ppv)

        if plot_pr_idx is not None:
            idx = np.argsort(tpr)
            tmp_tpr, tmp_ppv = tpr[idx], ppv[idx]
            step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
            plt.step(tmp_tpr, tmp_ppv, color='b', alpha=0.2, where='post')
            plt.fill_between(tmp_tpr, tmp_ppv, alpha=0.2, color='b', **step_kwargs)
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.show()

        return calc_auc_from_tpr_fpr(tpr, fpr), calc_apr_from_ppv_tpr(ppv, tpr)

    way_names = ['mean', 'max', 'median', 'std', 'PSNR', 'SSIM']
    # load scores and calc weights (all converted to sequence)
    score_appe_maps, score_row_maps, score_col_maps = \
        calc_score_full_clips(dataset, image_size, cube_size, model_idx, train=False, force_calc=False)
    w_appe, w_row, w_col = get_weights(dataset, image_size, cube_size, model_idx, operation)
    print('====================== %s (manual) =====================' % way_names[selected_score_estimation_way])

    # only consider selected score types
    score_appe_maps = score_appe_maps[..., selected_score_estimation_way]
    print('weights:', w_appe.shape, w_row.shape, w_col.shape)
    print(np.min(w_appe), np.max(w_appe), np.min(w_row), np.max(w_row), np.min(w_col), np.max(w_col))
    # calc combined score maps
    # w_appe, w_row, w_col = 1, 5, 0.5
    score_full_maps = np.array([(score_appe_maps[i]**0.5 * (1-w_appe) + score_row_maps[i]**1 * (1-w_row) + score_col_maps[i]**1 * (1-w_col))
                               for i in range(len(score_appe_maps))])

    auc, prc = basic_assess_AUC_2(score_appe_maps, frame_labels, plot_pr_idx=plot_pr_idx) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('appearance AUCs: %.3f, PRscore: %.3f' % (auc, prc))

    auc, prc = basic_assess_AUC_2(score_row_maps, frame_labels, plot_pr_idx=plot_pr_idx) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('row index  AUCs: %.3f, PRscore: %.3f' % (auc, prc))

    auc, prc = basic_assess_AUC_2(score_col_maps, frame_labels, plot_pr_idx=plot_pr_idx) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('col index  AUCs: %.3f, PRscore: %.3f' % (auc, prc))

    auc, prc = basic_assess_AUC_2(score_full_maps, frame_labels, plot_pr_idx=plot_pr_idx) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('combinatio AUCs: %.3f, PRscore: %.3f' % (auc, prc))


def full_assess_AUC_multiple_scale(dataset, image_sizes, cube_size, model_idx, frame_labels,
                                   sequence_n_frame=None, selected_score_estimation_way=3, operation=np.std):
    weights = [1.0, 0.0, 0.0]
    # weights = [0.0, 1.0, 0.0]
    # weights = [0.0, 0.0, 1.0]
    weights = [4, -2, -1]
    p = 0.5*8
    assert not len(image_sizes) > len(weights)
    for i in range(len(image_sizes)):
        tmp_score_appe_seq, tmp_score_row_seq, tmp_score_col_seq, tmp_score_full_seq = \
            full_assess_AUC(dataset, image_sizes[i], cube_size, model_idx, frame_labels,
                            sequence_n_frame=None,
                            selected_score_estimation_way=selected_score_estimation_way,
                            operation=operation, show=False)
        if i == 0:
            score_appe_seq = tmp_score_appe_seq * weights[i]**p
            score_row_seq = tmp_score_row_seq * weights[i]**p
            score_col_seq = tmp_score_col_seq * weights[i]**p
            score_full_seq = tmp_score_full_seq * weights[i]**p
        else:
            score_appe_seq += tmp_score_appe_seq * weights[i]**p
            score_row_seq += tmp_score_row_seq * weights[i]**p
            score_col_seq += tmp_score_col_seq * weights[i]**p
            score_full_seq += tmp_score_full_seq * weights[i]**p

    # split to clips
    if sequence_n_frame is not None:
        accumulated_n_frame = np.cumsum(sequence_n_frame - cube_size[2] + 1)[:-1]
        score_appe_seq = np.split(score_appe_seq, accumulated_n_frame, axis=0)
        score_row_seq = np.split(score_row_seq, accumulated_n_frame, axis=0)
        score_col_seq = np.split(score_col_seq, accumulated_n_frame, axis=0)
        score_full_seq = np.split(score_full_seq, accumulated_n_frame, axis=0)
        # normalize score in each clip
        score_appe_seq = [item/np.max(item, axis=0) for item in score_appe_seq]
        score_row_seq = [item/np.max(item, axis=0) for item in score_row_seq]
        score_col_seq = [item/np.max(item, axis=0) for item in score_col_seq]
        score_full_seq = [item/np.max(item, axis=0) for item in score_full_seq]

        # test if keyframe assessment is better
        perform_check = False
        if perform_check:
            frame_labels = np.split(frame_labels, accumulated_n_frame, axis=0)
            score_appe_seq = np.array([score_appe_seq[i][np.arange(1, len(score_appe_seq[i]), cube_size[-1])] for i in range(len(score_appe_seq))])
            score_row_seq = np.array([score_row_seq[i][np.arange(1, len(score_row_seq[i]), cube_size[-1])] for i in range(len(score_row_seq))])
            score_col_seq = np.array([score_col_seq[i][np.arange(1, len(score_col_seq[i]), cube_size[-1])] for i in range(len(score_col_seq))])
            score_full_seq = np.array([score_full_seq[i][np.arange(1, len(score_full_seq[i]), cube_size[-1])] for i in range(len(score_full_seq))])
            frame_labels = np.array([frame_labels[i][np.arange(1, len(frame_labels[i]), cube_size[-1])] for i in range(len(frame_labels))])
            frame_labels = np.concatenate(frame_labels, axis=0)
        # concatenate again
        score_appe_seq = np.concatenate(score_appe_seq, axis=0)
        score_row_seq = np.concatenate(score_row_seq, axis=0)
        score_col_seq = np.concatenate(score_col_seq, axis=0)
        score_full_seq = np.concatenate(score_full_seq, axis=0)

    print('========= multiple scale ==========')
    print(score_appe_seq.shape, score_row_seq.shape, score_col_seq.shape, score_full_seq.shape)

    auc, eer, eer_expected, prc = basic_assess_AUC(score_appe_seq, frame_labels, plot_pr_idx=None) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('appearance AUCs: %.3f (%.3f, %.3f), PRscore: %.3f' % (auc, eer, eer_expected, prc))

    auc, eer, eer_expected, prc = basic_assess_AUC(score_row_seq, frame_labels, plot_pr_idx=None) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('row index  AUCs: %.3f (%.3f, %.3f), PRscore: %.3f' % (auc, eer, eer_expected, prc))

    auc, eer, eer_expected, prc = basic_assess_AUC(score_col_seq, frame_labels, plot_pr_idx=None) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('col index  AUCs: %.3f (%.3f, %.3f), PRscore: %.3f' % (auc, eer, eer_expected, prc))

    auc, eer, eer_expected, prc = basic_assess_AUC(score_full_seq, frame_labels, plot_pr_idx=None) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    print('combinatio AUCs: %.3f (%.3f, %.3f), PRscore: %.3f' % (auc, eer, eer_expected, prc))


def get_segments(seq):
    def find_ends(seq):
        tmp = np.insert(seq, 0, -10)
        diff = tmp[1:] - tmp[:-1]
        peaks = np.where(diff != 1)[0]
        #
        ret = np.empty((len(peaks), 2), dtype=int)
        for i in range(len(ret)):
            ret[i] = [peaks[i], (peaks[i+1]-1) if i < len(ret)-1 else (len(seq)-1)]
        return ret
    #
    ends = find_ends(seq)
    return np.array([[seq[curr_end[0]], seq[curr_end[1]]] for curr_end in ends]).reshape(-1) + 1  # +1 for 1-based index (same as UCSD data)


def load_ground_truth_Avenue(folder, n_clip):
    ret = []
    for i in range(n_clip):
        filename = '%s/%d_label.mat' % (folder, i+1)
        data = loadmat(filename)['volLabel']
        n_bin = np.array([np.sum(data[0, i]) for i in range(len(data[0]))])
        abnormal_frames = np.where(n_bin > 0)[0]
        ret.append(get_segments(abnormal_frames))
    return ret


def load_ground_truth_Boat(dataset, n_clip=1, keep_patch_struct=False, write_to_bin=False):
    ret = []
    if write_to_bin:
        bin_data = np.array([])
    for i in range(n_clip):
        filenames = sorted(glob.glob('%s/GT%s/*.png' % (dataset['path_test'], str(i+1).zfill(3))))
        print(len(filenames))
        data = np.array([imread(filename, 'L')/255. for filename in filenames])
        if keep_patch_struct:
            ret.append(data)
        else:
            n_bin = np.array([np.sum(data[i]) for i in range(len(data))])
            if write_to_bin:
                bin_data = np.concatenate((bin_data, n_bin), axis=0)
            abnormal_frames = np.where(n_bin > 0)[0]
            ret.append(get_segments(abnormal_frames))
    if write_to_bin:
        print('loaded', len(bin_data), 'label groundtruths')
        bin_file = open('%s/labels.bin' % dataset['path_test'], 'wb')
        bin_file.write(bytearray(bin_data))
        bin_file.close()
    return ret


def load_ground_truth_Subway(dataset):
    labels = np.load('../dataset/Amit-Subway/subway_labels/%s_clip_labels.npz' % dataset['name'].lower())['label']
    print(len(labels), dataset['n_clip_test'], dataset['n_clip_train'])
    return labels


def write_sequence_to_bin(bin_file, data, reload_to_check=True):
    data = data.astype(np.int32)
    writer = open(bin_file, 'wb')  # file to write to
    writer.write(data)
    writer.close()
    if reload_to_check:
        import struct
        loaded_data = []
        with open(bin_file, 'rb') as f:
            while True:
                datum = f.read(4)
                if not datum:
                    break
                loaded_data.append(struct.Struct('i').unpack_from(datum))
        loaded_data = np.array(loaded_data).flatten()
        print(loaded_data[:5], data[:5])
        print('equal data:', np.array_equal(data, np.array(loaded_data)), np.sum(data-loaded_data), 'shape:', data.shape)


# PIXEL-WISE SECTION
def get_pixel_gt(dataset, cube_size, sequence_n_frame, select_frame='mid'):
    assert select_frame in ['last', 'first', 'mid']
    pixel_gt = np.array([])
    for idx in dataset['ground_truth_mask']:
        gt_clip_path = '%s/Test%s_gt' % (dataset['path_test'], str(idx).zfill(3))
        image_paths = sorted(glob.glob('%s/*.bmp' % gt_clip_path))
        # select frames for gt
        if select_frame == 'last':
            n_removed_frame = cube_size[2] - 1
            image_paths = image_paths[n_removed_frame:]
        elif select_frame == 'first':
            n_removed_frame = cube_size[2] - 1
            image_paths = image_paths[:-n_removed_frame]
        else:
            seq_length = sequence_n_frame[idx-1] + 1 - cube_size[2]
            start_idx = cube_size[2]//2
            stop_idx = start_idx + seq_length
            image_paths = image_paths[start_idx:stop_idx]
        # load groundtruth
        gt = np.array([imread(path, 'L').astype(np.float32)/255. for path in image_paths])
        pixel_gt = np.append(pixel_gt, gt)
        print(pixel_gt.shape)
    return np.concatenate(pixel_gt, axis=0)


# TODO (only need to check again)
def pixel_wise_assessment(score_maps, gt_masks, image_size, frame_labels):
    def calc_auc_from_tpr_fpr(tpr_fpr_matrix):
        tpr, fpr = tpr_fpr_matrix[:, 0], tpr_fpr_matrix[:, 1]
        idx = np.argsort(fpr)
        tpr, fpr = tpr[idx], fpr[idx]
        if fpr[0] > 0:
            fpr = np.insert(fpr, 0, 0)
            tpr = np.insert(tpr, 0, 0)
        if fpr[-1] < 1:
            fpr = np.append(fpr, 1)
            tpr = np.append(tpr, 1)
        auc = np.sum([0.5*(tpr[i+1]+tpr[i])*(fpr[i+1]-fpr[i]) for i in range(len(fpr)-1)])
        return auc

    # Anomaly Detection and Localization in Crowded Scenes, section 6.2
    def calc_tpr_fpr_pixel_wise(score_maps, gt_masks, frame_labels, thresh):
        def decide_pixel_based_frame(gt_mask, pred_mask):
            # return -1: TN or FN (dont care), 0: FP, 1: TP
            if np.sum(gt_mask) > 0 and np.sum(pred_mask*gt_mask)/np.sum(gt_mask) >= 0.4:
                return 1
            if np.sum(gt_mask) == 0 and np.sum(pred_mask) > 0:
                return 0
            return -1

        assert len(score_maps) == len(gt_masks)
        pred_masks = np.ones_like(score_maps)
        pred_masks[score_maps < thresh] = 0
        #
        pred_frame = np.array([decide_pixel_based_frame(gt_masks[i], pred_masks[i]) for i in range(len(pred_masks))])
        tpr = len(np.where(pred_frame == 1)[0])/len(np.where(frame_labels == 1)[0])
        fpr = len(np.where(pred_frame == 0)[0])/len(np.where(frame_labels == 0)[0])
        return [tpr, fpr]

    # calc score values for using as threshold
    score_vals = np.unique(score_maps)
    # resize score map to image size
    score_maps = np.array([cv2.resize(score_map, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST) for score_map in score_maps])
    # calc tpr & fpr for multiple thresholds
    tpr_fpr = np.array([calc_tpr_fpr_pixel_wise(score_maps, gt_masks, frame_labels, thresh) for thresh in score_vals])
    return calc_auc_from_tpr_fpr(tpr_fpr)


# TODO
# apply for Boat, Belleview, Train
def get_patch_gt(dataset, cube_size, sequence_n_frame, select_frame='mid'):
    assert select_frame in ['last', 'first', 'mid']
    raw_gt_masks_clips = load_ground_truth_Boat(dataset, n_clip=dataset['n_clip_test'], keep_patch_struct=True)
    gt_masks = []
    for i in range(len(raw_gt_masks_clips)):
        raw_gt_masks = raw_gt_masks_clips[i]
        # select frames for gt
        if select_frame == 'last':
            n_removed_frame = cube_size[2] - 1
            raw_gt_masks = raw_gt_masks[n_removed_frame:]
        elif select_frame == 'first':
            n_removed_frame = cube_size[2] - 1
            raw_gt_masks = raw_gt_masks[:-n_removed_frame]
        else:
            seq_length = sequence_n_frame[i] + 1 - cube_size[2]
            start_idx = cube_size[2]//2
            stop_idx = start_idx + seq_length
            raw_gt_masks = raw_gt_masks[start_idx:stop_idx]
        # load groundtruth
        gt_masks.append(raw_gt_masks)
    return np.concatenate(gt_masks, axis=0)


def patch_wise_assessment(score_maps, gt_masks, pr_calc=True, title=''):
    def process_one_map_mask(in_score_map, gt_mask):
        #
        score_map = cv2.resize(in_score_map, tuple([gt_mask.shape[1], gt_mask.shape[0]]), interpolation=cv2.INTER_LINEAR)
        #
        scores = np.zeros(len(gt_mask.flatten()))
        labels = np.ones(len(gt_mask.flatten()), dtype=int)
        idx = 0
        step_h = int(np.ceil(score_map.shape[0]/gt_mask.shape[0]))
        step_w = int(np.ceil(score_map.shape[1]/gt_mask.shape[1]))
        for i in range(gt_mask.shape[0]):
            for j in range(gt_mask.shape[1]):
                tmp = score_map[i*step_h:np.min([(i+1)*step_h, score_map.shape[0]]),
                                j*step_w:np.min([(j+1)*step_w, score_map.shape[1]])].flatten()
                vals = np.unique(tmp)
                freq = np.array([len(np.where(tmp == val)[0]) for val in vals])
                scores[idx] = vals[np.where(freq == np.max(freq))[0][0]]
                if gt_mask[i, j] == 0:
                    labels[idx] = 0
                idx += 1
        return scores, labels

    assert len(score_maps) == len(gt_masks)
    patch_data = np.array([process_one_map_mask(score_maps[i], gt_masks[i]) for i in range(len(score_maps))])
    scores = np.concatenate([patch_datum[0] for patch_datum in patch_data], axis=0)
    labels = np.concatenate([patch_datum[1] for patch_datum in patch_data], axis=0)
    #
    if pr_calc:
        precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
        average_precision = average_precision_score(labels, scores, pos_label=1)
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('[%s] 2-class Precision-Recall curve: AP={%0.2f}' % (title, average_precision))
    else:
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        auc = roc_auc_score(labels, scores)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='[%s] ROC curve (area = %0.2f)' % (title, auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
    plt.show()
