# -*- coding: utf-8 -*-
# phoenixyli 李岩 @2020-04-02 17:03:11

import os

def return_ucf101(modality):
    filename_categories = 101
    if modality == 'RGB':
        filename_imglist_train = '/data2/v_jasonbji/ucfTrainTestlist/ucf101_rgb_train_split_3.txt'
        filename_imglist_val = '/data2/v_jasonbji/ucfTrainTestlist/ucf101_rgb_val_split_3.txt'
        prefix = 'image_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, prefix

def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        filename_imglist_train = '/data2/v_jasonbji/hmdb_tsn_split/hmdb51_rgb_train_split_3.txt'
        filename_imglist_val = '/data2/v_jasonbji/hmdb_tsn_split/hmdb51_rgb_val_split_3.txt'
        prefix = 'image_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, prefix

def return_something(modality):
    filename_categories = 174
    if modality == 'RGB':
        filename_imglist_train = '/data1/phoenixyli/DeepLearning/' \
                'something-something-v1/TrainTestlist/train_videofolder_new.txt'
        filename_imglist_val = '/data1/phoenixyli/DeepLearning/' \
                'something-something-v1/TrainTestlist/val_videofolder_new.txt'
        prefix = '{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, prefix

def return_somethingv2(modality):
    filename_categories = 174
    if modality == 'RGB':
        filename_imglist_train = '/data2/v_jasonbji/ft_local/Something-Something-V2/train_videofolder.txt'
        filename_imglist_val = '/data2/v_jasonbji/ft_local/Something-Something-V2/test_videofolder.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, prefix

def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        filename_imglist_train = '/data2/v_jasonbji/v_jasonbji_data/ft_local/kinetics_400_train.txt'
        filename_imglist_val = '/data2/v_jasonbji/v_jasonbji_data/ft_local/kinetics_400_val.txt'
        prefix = 'image_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, prefix


def return_dataset(dataset, modality):
    dict_single = {
        'something': return_something,
        'somethingv2': return_somethingv2,
        'ucf101': return_ucf101,
        'hmdb51': return_hmdb51,
        'kinetics': return_kinetics
    }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, prefix
