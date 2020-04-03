# -*- coding: utf-8 -*-
# phoenixyli 李岩 @2020-04-02 21:15:33

import os
import argparse
import time

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F
import pickle

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # import pdb; pdb.set_trace()
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res


parser = argparse.ArgumentParser(description="TEA testing on the full validation set")
parser.add_argument('dataset', type=str)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--num_clips', type=int, default=10)

args = parser.parse_args()

output_dir = './result_file'
output_filepath = os.path.join(output_dir, '0'+'_'+'crop'+str(args.test_crops)+'.pkl')
with open(output_filepath, 'rb') as f:
    output_file = pickle.load(f)
    num_videos = len(output_file)
    num_classes = output_file[0][0].shape[1]

num_clips = args.num_clips
ens_pred_numpy = np.zeros((num_videos, num_classes))
ens_label_numpy = np.zeros((num_videos,))

for clip_index in range(num_clips):
    output_filepath = os.path.join(output_dir, str(clip_index)+'_'+'crop'+str(args.test_crops)+'.pkl')
    with open(output_filepath, 'rb') as f:
        output_file = pickle.load(f)
        pred_numpy = output_file[0]
        ens_pred_numpy  = ens_pred_numpy + pred_numpy
        label_numpy = output_file[1]
        ens_label_numpy = ens_label_numpy + label_numpy

ens_pred_numpy = ens_pred_numpy / num_clips
ens_label_numpy = ens_label_numpy / int(num_clips)

prec1, prec5 = accuracy(torch.from_numpy(ens_pred_numpy), torch.from_numpy(ens_label_numpy).type(torch.LongTensor), topk=(1, 5))

# import pdb; pdb.set_trace()

video_pred = [np.argmax(x) for x in ens_pred_numpy]
video_labels = [x for x in ens_label_numpy]

cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt
# print(cls_acc)
upper = np.mean(np.max(cf, axis=1) / cls_cnt)
# print('upper bound: {}'.format(upper))

print('-----Evaluation is finished------')
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(prec1, prec5))
