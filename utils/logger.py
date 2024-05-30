import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import traceback
import torch
import torch.nn as nn

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, resume=False): 
        self.file = None
        self.resume = resume
        if os.path.isfile(fpath):
            if resume:
                self.file = open(fpath, 'a') 
            else:
                self.file = open(fpath, 'w')
        else:
            self.file = open(fpath, 'w')

    def append(self, target_str):
        if not isinstance(target_str, str):
            try:
                target_str = str(target_str)
            except:
                traceback.print_exc()
            else:
                print(target_str)
                self.file.write(target_str + '\n')
                self.file.flush()
        else:
            print(target_str)
            self.file.write(target_str + '\n')
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


class RandomGaussianNoise(nn.Module):
    def __init__(self, p=0.5, s=0.1):
        super().__init__()
        self.p = p
        self.std = s ** 0.5

    def forward(self, img):
        v = torch.rand(1)[0]
        if v < self.p:
            img += torch.randn(img.shape, device=img.device) * self.std
        return img
