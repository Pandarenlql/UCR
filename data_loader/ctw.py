import torch
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import os
from glob import glob
from copy import deepcopy


class CTW(data.Dataset):
    def __init__(self, img_root, id_file="", load_type='test', transform=None, file_path='./files',
                 max_ids_len = None):
        self.root = img_root
        self.id_file = id_file
        self.load_type = load_type
        self.transform = transform
        self.ids_encode_all = os.path.join(file_path, "IDS-Encode-based-radical-V2-Combine.txt")
        self.ids_encode_ctw = os.path.join(file_path, "IDS-Encode-ALL.txt")
        self.max_ids_len = max_ids_len
        
        self.imgs = []
        self.ids_encode = {}
        self.ids_sequence = {}
        self.label_id = {}
        self.id_label = {}
        self.img_file = []

        self._load_files()
        self._loader() 

    def _load_files(self):

        # 注：该字典的数据格式均为：{int: 'str', ...}
        real_path = os.path.join(self.root, self.load_type)       
        if self.load_type == 'test':
            self.filenames = sorted(glob(os.path.join(real_path,'*')))
            self.filenums = [x.split('/')[-1] for x in self.filenames]
            for filename in self.filenames:
                img_names = sorted(glob(os.path.join(filename,'*')))
                self.imgs.extend(img_names)
                self.img_file.append(len(img_names))
            self.file_img_pair = list(zip(self.filenums, self.img_file))

            with open(self.id_file, 'r') as fd:
                for i, _line in enumerate(fd.readlines()):
                    # skip first line
                    if i==0:
                        continue
                    infos = _line.replace('\n', '').split('\t')
                    self.label_id[str(i-1)] = infos[0]
        else:
            with open(self.id_file, 'r') as fd:
                for i, _line in enumerate(fd.readlines()):
                    # skip first line
                    if i==0:
                        continue
                    infos = _line.replace('\n', '').split('\t')
                    class_path = os.path.join(real_path, infos[0])
                    self.label_id[str(i-1)] = infos[0]
                    img_path = sorted(glob(os.path.join(class_path,'*')))
                    for j in range(len(img_path)):
                        self.imgs.append((img_path[j], str(i-1)))
        
        with open(self.ids_encode_all, 'r') as fd:
            for i, _line in enumerate(fd.readlines()):
                # skip the first two lines
                if i<=1:
                    continue
                infos = _line.replace('\n', '').split('\t')
                # NOTE：self.ids_encode format: {'⿰':0, '⿱':1, ...}
                self.ids_encode[infos[0]] = int(infos[1])
        
        basic_radical_num = len(self.ids_encode.keys())     # num is 511
        self.ids_encode['<start>'] = basic_radical_num      # 511
        self.ids_encode['<eos>'] = basic_radical_num + 1    # 512
        self.ids_encode['<pad>'] = basic_radical_num + 2    # 513
        self.radical_nums = len(self.ids_encode.keys())

        with open(self.ids_encode_ctw, 'r') as fd:
            for i, _line in enumerate(fd.readlines()):
                # skip the first lines
                if i<1:
                    continue
                infos = _line.replace('\n', '').split('\t')
                ids_sequence = infos[1].split()
                ids_sequence = [int(x) for x in ids_sequence]
                # 注：self.ids_encode_ctw format: {'说':[0, 12, 1, 13, 1, 14, 15], '⿱':'1', ...}
                self.ids_sequence[infos[0]] = ids_sequence

    def _loader(self):       
        all_ids_sequence = list(self.ids_sequence.values())   
        # max len is 21       
        self._max_ids_len = max([len(x) for x in all_ids_sequence])
        self._max_ids_len += 2

        if not self.max_ids_len:
            self.max_ids_len = self._max_ids_len
        
        if self.load_type == 'test':
            pass
        else:
            self.all_datas = [[]]
            for i, (path, label) in enumerate(self.imgs):
                try:
                    real_id = self.label_id[label]
                except KeyError:
                    real_id = path.split('/')[-2]
                real_id_ids, real_len = self._trans_to_ids(real_id)
                packed_list = [path, real_id_ids, real_len, label]
                
                if i==0:
                    self.all_datas[0] = packed_list
                else:
                    self.all_datas.append(packed_list)


    def _trans_to_ids(self, real_id):
        ids = deepcopy(self.ids_sequence[real_id])
        ids = [self.ids_encode['<start>']] + ids

        ids.extend([self.ids_encode['<eos>']])
        real_len = len(ids) 
        
        for i in range(real_len, self.max_ids_len):
            ids.extend([self.ids_encode['<pad>']])
        return ids, real_len

    def __getitem__(self, index):
        if self.load_type == 'test':
            img = Image.open(self.imgs[index])
            if self.transform is not None:
                img = self.transform(img)
            return img, self.imgs[index].split("/")[-2:]
        else:
            path, real_id_ids, caption_length, label = self.all_datas[index]
            img = Image.open(path)
            if self.transform is not None:
                img = self.transform(img)
            real_id_ids = torch.from_numpy(np.array(real_id_ids).astype(int))
            label = torch.from_numpy(np.array(label).astype(int))             
            return img, real_id_ids, caption_length, label
    
    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    ctw_path = "/home/qilong/datasets/CTW"
    ctw_id_path = "/home/qilong/UCR/UCR/files/id_file.txt"
    _file = "/home/qilong/UCR/UCR/files/"

    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
        ])

    ctw_train = CTW(ctw_path, id_file=ctw_id_path, load_type='train', 
                            transform=data_transform, file_path=_file)
    ctw_val = CTW(ctw_path, id_file=ctw_id_path, load_type='val', 
                            transform=data_transform, file_path=_file)

    train_dataloaders = torch.utils.data.DataLoader(ctw_train, batch_size=1, 
                                                    shuffle=True)
    val_dataloaders   = torch.utils.data.DataLoader(ctw_val, batch_size=1, 
                                                    shuffle=True) 
    
    print("len train images is ", ctw_train.__len__())
    print("len val images is ", ctw_val.__len__())
    for i, (train_data, real_id_ids, caption_length, label) in enumerate(train_dataloaders): 
        print("Train data shape is ", train_data.shape)
        print("real_id_ids is ", real_id_ids, " shape is ", real_id_ids.shape)
        print("caption_length is ", caption_length, " shape is ", caption_length.shape)
        print("label is ", label, " shape is ", label.shape)
        print("max len is ", ctw_train.max_ids_len)

        raise ValueError("Debug")
    
    for i, (test_data, real_id_ids, caption_length, label) in enumerate(val_dataloaders): 
        print("Test data shape is ", test_data.shape)
        print("real_id_ids is ", real_id_ids)
        raise ValueError("Debug")
