# -*- coding: utf-8 -*-
# phoenixyli 李岩 @2020-04-02 14:33:59
import os
import numpy as np
from numpy.random import randint

import torch.utils.data as data
from PIL import Image

class VideoRecord(object):
    """Store the basic information of the video

    _data[0]: the absolute path of the video frame folder
    _data[1]: the frame number
    _data[2]: the label of the video
    """

    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    """The torch dataset for the video data.

    :param list_file: the list file is utilized to specify the data sources.
    Each line of the list file contains a tuple of extracted video frame folder path (absolute path),
    video frame number, and video groundtruth class. An example line looks like:
    /data/xxx/xxx/Dataset/something-somthing-v1/100218 42 134
    """

    def __init__(
            self, list_file, num_segments=8, new_length=1, modality='RGB',
            image_tmpl='img_{:05d}.jpg', transform=None, random_shift=True,
            test_mode=False, remove_missing=False, multi_clip_test=False,
            dense_sample=False):
        
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.multi_clip_test = multi_clip_test
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(directory, self.image_tmpl.format(1))).convert('RGB')]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """Random Sampling from each video segment

        :param record: VideoRecord
        :return: list
        """

        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration)\
                          + randint(average_duration, size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        """Sampling for validation set

        Sample the middle frame from each video segment
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        record = self.video_list[index]
        # check this is a legit video folder
        file_name = self.image_tmpl.format(1)
        full_path = os.path.join(record.path, file_name)

        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(record.path, file_name)

        if not self.test_mode:  # training or validation set
            if self.random_shift:  # training set
                segment_indices = self._sample_indices(record)
            else:  # validation set
                segment_indices = self._get_val_indices(record)
        else:  # test set
            # for mulitple clip test, use random sampling;
            # for single clip test, use middle sampling
            if self.multi_clip_test:
                segment_indices = self._sample_indices(record)
            else:
                segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

if __name__ == "__main__":
    # test dataset
    test_train_list ='/data1/phoenixyli/DeepLearning/something-something-v1/TrainTestlist/val_videofolder_new.txt'
    test_num_segments = 8
    data_length = 1
    test_modality = 'RGB'
    prefix = '{:05d}.jpg'
    train_dataset = TSNDataSet(
        test_train_list, num_segments=test_num_segments,
        new_length=data_length, modality=test_modality,
        image_tmpl=prefix, multi_clip_test=False, dense_sample=False)
    data, label = train_dataset.__getitem__(10)

