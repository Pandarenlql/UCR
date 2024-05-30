import os
import random
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchtoolbox.transform import Cutout
from data_loader.rects import *
from data_loader.ctw import *
import models.ucr as ucr
from trainer import Trainer
from utils.logger import *
from utils.acc import *
from torchnet.meter import ClassErrorMeter
import torch.backends.cudnn as cudnn

def load_state_dict(model_dir, is_multi_gpu):
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]       # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict


# Edit loss
class EditDistanceLoss(nn.Module):
    def __init__(self):
        super(EditDistanceLoss, self).__init__()
        
    def forward(self, input, target):
        diff_list = (input - target).abs()>0
        return diff_list.float().sum()


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu   
    is_use_cuda = torch.cuda.is_available()
    if is_use_cuda: 
        print("Use GPU!")
    cudnn.benchmark = True
    
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    
    if args.resume:
        logger = Logger('./logs/' + args.model + '.log', True)
    else:
        logger = Logger('./logs/' + args.model + '.log')

    logger.append(vars(args))
    gpus = args.gpu.split(',')
  
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomApply([transforms.RandomRotation(50)], p=0.5),
            transforms.RandomApply([Cutout()], p=0.5),
            transforms.ToTensor(),
            transforms.RandomApply([RandomGaussianNoise()], p=0.4)
        ]),
        'val': transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])}

    if args.dataset == 'rects':
        train_datasets = ReCTS(args.data_root, id_file=args.id_file, load_type='train', transform=data_transforms['train'],
                               file_path=args.files_path)
        val_datasets = ReCTS(args.data_root, id_file=args.id_file, load_type='val', transform=data_transforms['val'],
                             file_path=args.files_path)
        
    elif args.dataset == 'ctw':
        train_datasets = CTW(args.data_root, id_file=args.id_file, load_type='train', transform=data_transforms['train'], 
                             file_path=args.files_path)
        val_datasets = CTW(args.data_root, id_file=args.id_file, load_type='val', transform=data_transforms['val'],
                           file_path=args.files_path)
    else:
        raise ValueError("Unsupport this dataset")

    init_lr = 1e-3
    args.radical_nums = train_datasets.radical_nums
    args.all_ids_sequence = train_datasets.ids_sequence
    args.start = train_datasets.ids_encode['<start>']
    args.eos = train_datasets.ids_encode['<eos>']
    args.pad = train_datasets.ids_encode['<pad>']
    args.rev_ids_encode = {v: k for k, v in train_datasets.ids_encode.items()}
    # 将start和eos标志位加入到所有ids-to-label字典中, 目的是在训练中提供终止位, 
    # 否则预测向量包含了eos位, 而标签向量没有标志位的话, 预测会出错
    for key in args.all_ids_sequence.keys():
        args.all_ids_sequence[key] = [args.start] + args.all_ids_sequence[key]
        args.all_ids_sequence[key].append(args.eos)
    args.label_id = train_datasets.label_id
    args.class_num = len(train_datasets.label_id.keys())

    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size*len(gpus), shuffle=True, num_workers=8)
    val_dataloaders = torch.utils.data.DataLoader(val_datasets, batch_size=args.batch_size, shuffle=False, num_workers=8)

    if args.model == 'UCR':
        my_model = ucr.ucr_large(clsss_num=args.class_num, ids_class=args.radical_nums)
    else:
        raise ModuleNotFoundError

    if is_use_cuda and len(gpus) == 1:
        my_model = my_model.cuda()
    elif is_use_cuda and len(gpus) > 1:
        my_model = nn.DataParallel(my_model.cuda())

    loss_fn = [nn.CrossEntropyLoss(), nn.MSELoss(), EditDistanceLoss()]
    args.start_epoch = 0
    args.best_acc = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['cur_epoch']
            args.best_acc = checkpoint['best_acc']
            my_model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['cur_epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    optimizer = optim.Adam(my_model.parameters(), lr=init_lr)
    lr_schedule = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    metric = [ClassErrorMeter([1,5], True), AverageMeter_ids(args.eos)]

    my_trainer = Trainer(my_model, loss_fn, optimizer, lr_schedule, is_use_cuda, train_dataloaders, \
                        val_dataloaders, metric, logger, args)
    my_trainer.fit()
    logger.append('Optimize Done!')


if __name__ == '__main__':
    """
    Train command:
    python train.py --gpu 0 -p 1500 --dataset ctw  --batch_size 52
    """
    parser = argparse.ArgumentParser(description='PyTorch Template')
    # ----------------Load Data-------------------
    parser.add_argument('--dataset',  default='ctw', type=str,
                        help='dataset select: [rects, ctw]')
    parser.add_argument('-d', '--data_root', default='/home/qilong/datasets/CTW/',
                         type=str, help='data root')
    parser.add_argument('-id', '--id_file', default='./files/id_file.txt', type=str, help='dataset id file')
    parser.add_argument('--files_path', default='./files', type=str, help='IDS files')
    parser.add_argument('-b', '--batch_size', default=384,
                         type=int, help='model train batch size')
    parser.add_argument('--image_size', default=32,
                         type=int, help='The size of input image, must be 32 for the input size is 8 for vit')
    # ----------------Load Model-------------------
    parser.add_argument('-m', '--model', default='UCR', type=str)
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='path to latest checkpoint (default: None)')
    # ----------------Training control-------------------
    parser.add_argument('-n', '--num_epochs', default=20, type=int,
                         help='How many epochs in training stage.') 
    parser.add_argument('-g', '--gpu', default='0', type=str,
                        help='GPU ID Select')                    
    parser.add_argument('-p', '--print_num', default=1, type=int,
                         help='How many epochs print one str.')
    parser.add_argument('-s', '--switch_align', default=2, type=int,
                         help='How many epochs train the model withot align loss and edit loss.')
    parser.add_argument('-al', '--alpha_local', default=5, type=int,
                         help='the weight of ids loss. defualt is 5.')
    parser.add_argument('-aa', '--alpha_algin', default=1, type=int,
                         help='the weight of algin loss. defualt is 5.')
    parser.add_argument('-ae', '--alpha_edit', default=0.5, type=float,
                         help='the weight of edit loss.')
    args = parser.parse_args()
    main(args)

# 有时间的时候, 参考长尾学习的那个代码, 设立一个best_pred变量, 每轮输出一次目前最好的结果。