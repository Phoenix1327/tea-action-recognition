# -*- coding: utf-8 -*-
# phoenixyli 李岩 @2020-04-02 20:06:15

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

# options
parser = argparse.ArgumentParser(description="TEA testing on the full validation set")
parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weight', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--modality', type=str, default='RGB')
parser.add_argument('--arch', type=str, default='resnet50')
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--multi_clip_test', default=False, action="store_true", help='multi clip test')

# for true test
parser.add_argument('--csv_file', type=str, default=None)

parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')


# add model params
parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')

parser.add_argument('--clip_index', type=int, default=0)

args = parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res

def eval_video(data, label, net, args):
    net.eval()
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        batch_size = label.numel()
        num_crop = args.test_crops
        if args.dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample

        if args.modality == 'RGB':
            length = 3
        elif args.modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality "+ modality)

        data_in = data.view(-1, length, data.size(2), data.size(3))
        if args.shift:
            data_in = data_in.view(batch_size * num_crop, args.test_segments, length, data_in.size(2), data_in.size(3))
        rst = net(data_in)
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        if args.softmax:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst, dim=1)

        rst = rst.data.cpu().numpy().copy()

        if net.module.is_shift:
            rst = rst.reshape(batch_size, num_class)
        else:
            rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))

        return rst

num_class, args.train_list, val_list, prefix = dataset_config.return_dataset(args.dataset,
                                                                             args.modality)

net = TSN(num_class, args.test_segments, args.modality,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          img_feature_dim=args.img_feature_dim,
          pretrain=args.pretrain,
          is_shift=args.shift, shift_div=args.shift_div,
          shift_place=args.shift_place,)

# import pdb; pdb.set_trace()
'''
checkpoint = torch.load(args.weight)
checkpoint = checkpoint['state_dict']

base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
replace_dict = {
    'base_model.classifier.weight': 'new_fc.weight',
    'base_model.classifier.bias': 'new_fc.bias',
}
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)

net.load_state_dict(base_dict)
'''

input_size = net.scale_size if args.full_res else net.input_size
if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(input_size),
    ])
elif args.test_crops == 3:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupFullResSample(input_size, net.scale_size, flip=False)
    ])
elif args.test_crops == 5:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net.scale_size, flip=False)
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
    TSNDataSet(val_list, num_segments=args.test_segments,
               new_length=1 if args.modality == "RGB" else 5,
               modality=args.modality, image_tmpl=prefix,
               test_mode=True, remove_missing=True,
               multi_clip_test=args.multi_clip_test,
               transform=torchvision.transforms.Compose([
                   cropping,
                   Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                   ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                   GroupNormalize(net.input_mean, net.input_std),
               ]),
               dense_sample=args.dense_sample,),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))

net = torch.nn.DataParallel(net.cuda())
net.eval()

total_num = len(data_loader.dataset)

proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else total_num

top1 = AverageMeter()
top5 = AverageMeter()

all_results = np.zeros((0, num_class), dtype=np.float)
all_labels = np.zeros((0,), dtype=np.int)
import pdb; pdb.set_trace()
with torch.no_grad():
    for i, (data, label) in enumerate(data_loader):
        all_labels = np.hstack((all_labels, label))
        if i >= max_num:
            break
        rst = eval_video(data, label, net, args)
        all_results = np.vstack((all_results, rst))
        cnt_time = time.time() - proc_start_time
        prec1, prec5 = accuracy(torch.from_numpy(rst), label, topk=(1, 5))
        top1.update(prec1.item(), label.numel())
        top5.update(prec5.item(), label.numel())
        if i % 20 == 0:
            print('video {} done, total {}/{}, average {:.3f} sec/video, '
                  'moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i * args.batch_size,
                                                              i * args.batch_size, total_num,
                                                              float(cnt_time) / (i+1) / args.batch_size,
                                                              top1.avg, top5.avg))

import pdb; pdb.set_trace()
video_pred = [np.argmax(x) for x in all_results]
video_pred_top5 = [np.argsort(x)[::-1][:5] for x in all_results]
video_labels = [x for x in all_labels]


if args.csv_file is not None:
    print('=> Writing result to csv file: {}'.format(args.csv_file))
    with open(test_file_list[0].replace('test_videofolder.txt', 'category.txt')) as f:
        categories = f.readlines()
    categories = [f.strip() for f in categories]
    with open(test_file_list[0]) as f:
        vid_names = f.readlines()
    vid_names = [n.split(' ')[0] for n in vid_names]
    assert len(vid_names) == len(video_pred)
    if args.dataset != 'somethingv2':  # only output top1
        with open(args.csv_file, 'w') as f:
            for n, pred in zip(vid_names, video_pred):
                f.write('{};{}\n'.format(n, categories[pred]))
    else:
        with open(args.csv_file, 'w') as f:
            for n, pred5 in zip(vid_names, video_pred_top5):
                fill = [n]
                for p in list(pred5):
                    fill.append(p)
                f.write('{};{};{};{};{};{}\n'.format(*fill))

if args.multi_clip_test:
    output_dir = './result_file'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print("Store results matrix into {}".format(output_dir))
    
    output_filepath = os.path.join(output_dir, str(args.clip_index)+'_'+'crop'+str(args.test_crops)+'.pkl')
    with open(output_filepath, 'wb') as f:
        pickle.dump(all_results, f, pickle.HIGHEST_PROTOCOL)

cf = confusion_matrix(video_labels, video_pred).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt
print(cls_acc)
upper = np.mean(np.max(cf, axis=1) / cls_cnt)
print('upper bound: {}'.format(upper))

print('-----Evaluation is finished------')
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))


