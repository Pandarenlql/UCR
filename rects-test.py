"""
Test UCR on ReCTS
"""

import os
import math
import argparse
from collections import OrderedDict
from torchvision import transforms
from data_loader.rects import *
from data_loader.combine import *
import torch.nn.functional as F
from tqdm import tqdm
import random
from torch.autograd import Variable
import models.ucr as ucr

def create_masks(trg, is_use_cuda=True):
    # trg shape is [batch, seqLen]
    # 把非pad位全部置为1, 即True
    size = trg.size(1) # get seq_len for matrix
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)

    if is_use_cuda:
        np_mask = np_mask.cuda()
    return np_mask

def load_files(file_path):
    ids_encode_all = os.path.join(file_path, "IDS-Encode-based-radical-V2-Combine.txt")
    ids_encode_combine = os.path.join(file_path, "IDS-Encode-ALL.txt")
    ids_encode = {}
    all_ids_sequence = {}
    with open(ids_encode_all, 'r') as fd:
        for i, _line in enumerate(fd.readlines()):
            # skip the first two lines
            if i<=1:
                continue
            infos = _line.replace('\n', '').split('\t')
            ids_encode[infos[0]] = int(infos[1])
    
    basic_radical_num = len(ids_encode.keys())
    ids_encode['<start>'] = basic_radical_num      # 574
    ids_encode['<eos>'] = basic_radical_num + 1    # 575
    ids_encode['<pad>'] = basic_radical_num + 2    # 576
    radical_nums = len(ids_encode.keys())

    with open(ids_encode_combine, 'r') as fd:
        for i, _line in enumerate(fd.readlines()):
            # skip the first two lines
            if i<=1:
                continue
            infos = _line.replace('\n', '').split('\t')
            _ids_sequence = infos[1].split()
            _ids_sequence = [int(x) for x in _ids_sequence]
            # NOTE：ids_encode_combine format: {'说':[0, 12, 1, 13, 1, 14, 15], '⿱': 1, ...}
            all_ids_sequence[infos[0]] = _ids_sequence

    max_ids_len = max([len(x) for x in all_ids_sequence.values()])
    max_ids_len += 2

    return ids_encode, all_ids_sequence, radical_nums, max_ids_len

def test(args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    is_use_cuda = torch.cuda.is_available()

    #-----------------Load Data-------------------
    data_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
        ]) 

    if args.dataset == 'rects':
        test_datasets = ReCTS(args.data_root, id_file=args.id_file, load_type='test', transform=data_transform)
        label_id = test_datasets.label_id
        args.class_num = len(label_id.keys())
        ids_encode, all_ids_sequence, radical_nums, max_ids_len = load_files(args.files_path)
        ids_max_len = max_ids_len - 1
        # Insert the <eos> to all ids.
        for key in all_ids_sequence.keys():
            all_ids_sequence[key].append(ids_encode['<eos>'])
    test_data_loader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    #-----------------Load Model-------------------
    args.resume = os.path.join(args.resume, 'UCR/Models_best.ckpt')   
    model = ucr.ucr_large(clsss_num=args.class_num, ids_class=radical_nums)
    if is_use_cuda:
        model = model.cuda()

    state_dict = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())['state_dict']
    model.load_state_dict(state_dict)

    #---------------Combined Predictor paras-------
    ids_confidence = 0.96
    alpha = 0.184
    
    #-----------------Testing...-------------------
    model.eval()
    combined_pred_list = []
    print("Testing...")
    with torch.no_grad():
        for i, (inputs, imgs_name) in tqdm(enumerate(test_data_loader), desc="Prediction..."):
            y_start = torch.empty(1,1).int().fill_(ids_encode['<start>'])
            
            if is_use_cuda:
                inputs, y_start = inputs.cuda(), y_start.cuda()

            backbone_feature = model.cnn(inputs)
            decoder_mask = create_masks(y_start, is_use_cuda)
            predictions_ids = torch.zeros(1, max_ids_len).cuda()
            next_input = y_start
            next_mask = decoder_mask

            words_value = []
            for t in range(max_ids_len):
                global_pred, local_pred = model.vit(backbone_feature, next_input, next_mask, train=False)
                local_pred = F.softmax(local_pred, dim=2).data
                _loacl_value, local_pred = local_pred.topk(1, 2, True, True)
                words_value.append(_loacl_value.mean().cpu())
                next_word = local_pred[:, -1]
                next_input = torch.cat([next_input, next_word], dim=1)
                predictions_ids[0, t] = next_word[0][0]
                next_mask = create_masks(next_input, is_use_cuda)
                
                if local_pred[0, -1] == ids_encode['<eos>']:
                    break
            
            # Global pred
            global_pred = F.softmax(global_pred, dim=1).data.cpu()
            _value, cls_pred = global_pred.topk(1, 1, True, True)
            cls_pred = cls_pred.squeeze(1).tolist()[0]
            confidence_cls = _value.mean().cpu()
            _value = _value.tolist()
            
            # Combined Predictor
            multi_pred_confidence = 1
            mean_pred_confidence = 0
            penalty_num = 0
            for _each_value in words_value:
                if _each_value<ids_confidence:
                    penalty_num = penalty_num+1
                multi_pred_confidence = multi_pred_confidence*_each_value
                mean_pred_confidence = mean_pred_confidence + _each_value
            
            mean_pred_confidence = mean_pred_confidence/len(words_value)
            init_confidence_ids = (mean_pred_confidence + multi_pred_confidence)/2

            pred_len = len(words_value)
            penalty_factor = math.e ** -(max(penalty_num+ alpha*ids_max_len/pred_len - 1, 0))
            confidence_ids = penalty_factor * init_confidence_ids
            
            if confidence_cls > confidence_ids:
                final_pred_name = [label_id[str(cls_pred)]]
                pred_confidence = confidence_cls.tolist()
            else:
                final_pred_name = _trans_ids_name(predictions_ids, ids_encode, all_ids_sequence)
                pred_confidence = confidence_ids.tolist()
            combined_pred_list.extend(list(zip(imgs_name, final_pred_name)))
        combined_pred_list = sorted(combined_pred_list)  

        write_pred(combined_pred_list)
        print("Test has done!")

def write_pred(combined_pred):
    print("Writing txt...")
    save_path = args.files_path + 'rects-test'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    combined_path = os.path.join(save_path, 'UCR_ReCTSTest.txt')

    with open(combined_path, 'w') as f:
        for i in range(len(combined_pred)):
            img_name, id = combined_pred[i]
            new_img_name = img_name.split('_')[0]+'_'+img_name.split('_')[-1]
            f.writelines(new_img_name + ',' + id + '\n')
            
def _trans_ids_name(pred_ids, ids_encode, all_ids_sequence):
        """
        input: pred ids sequence or real ids sequence, shape: [b, length], type: tensor
        output: name, type: list, shape: [b,]
        """
        batch_num = pred_ids.shape[0]
        pred_ids = pred_ids.int().tolist()
        ids_name = []
        for i in range(batch_num):
            _ids = pred_ids[i]
            for ind, pred_label in enumerate(_ids):
                if pred_label == ids_encode['<eos>']:
                    del _ids[ind+1:]
            _name = [key for key, value in all_ids_sequence.items() if value == _ids]
            if _name:
                pass
            else:
                # Give a default name for unrecognized ids
                _name = '无'
            ids_name.extend(_name)
        return ids_name


if __name__ == '__main__':
    """
    NOTE: batch must equal 1
    python rects-test.py --gpu 0 --batch_size 1
    """
    
    parser = argparse.ArgumentParser(description='PyTorch Template')  
    parser.add_argument('--dataset',  default='rects', type=str)
    parser.add_argument('-d', '--data_root', default='/home/qilong/datasets/ReCTSTask1/',
                         type=str, help='data root')
    parser.add_argument('-id', '--id_file', default='./files/id_file.txt',
                         type=str, help='dataset id file')
    parser.add_argument('--files_path', default='./files/', type=str, help='IDS files')
    parser.add_argument('-b', '--batch_size', default=1,
                         type=int, help='In test procedure, batch must be 1')
    parser.add_argument('--image_size', default=32,
                         type=int, help='The size of input image')
    # ----------------Load Model-------------------
    parser.add_argument('-bv', '--backbone_vit', default='vit_base',
                         type=str, help='Only used if -m is ucr, select from [vit_base, vit_large, vit_huge, vit_test]')
    parser.add_argument('--backbone', default='resnet34', type=str,
                        help='CNN backbone, select from [resnet18, resnet34, resnet50, resnet101]')
    parser.add_argument('-r', '--resume', default='./checkpoint/', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--patch_size', default=32, type=int, 
                         help='The height and width of a patch')
    # ----------------Training control-------------------
    parser.add_argument('-g', '--gpu', default='0', type=str,
                        help='GPU ID Select')
    parser.add_argument('--seed', default=42, type=int,
                         help='random seed for easy repeat')
    args = parser.parse_args()
    test(args)
