import numpy as np
import torch
import torch.nn.functional as F
import time
import os
from utils.acc import *
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class Trainer():
    def __init__(self, model, loss_fn, optimizer, lr_schedule, is_use_cuda, train_data_loader, valid_data_loader,\
                 metric, logger, args):
        self.model = model
        self.loss_fn  = loss_fn
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule      
        self.is_use_cuda = is_use_cuda
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.metric = metric
        self.logger = logger
        self.model_name = args.model
        self.best_acc = args.best_acc
        self.log_batchs = args.print_num
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.cur_epoch = args.start_epoch
        self.dataset = args.dataset
        self.radical_nums = args.radical_nums
        self.all_ids_sequence = args.all_ids_sequence
        self.alpha_l = args.alpha_local
        self.alpha_align = args.alpha_algin
        self.alpha_edit = args.alpha_edit
        self.start = args.start
        self.eos = args.eos
        self.pad = args.pad
        self.id_label = {value: key for key, value in args.label_id.items()}
        self.label_id = args.label_id
        self.rev_ids_encode = args.rev_ids_encode
        self.switch_align = args.switch_align

    def fit(self):
        # Reset lr_schedule when use checkpoint
        for epoch in range(0, self.start_epoch):
            self.lr_schedule.step()
        self._init_infos()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.cur_epoch = epoch
            self.logger.append('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            self.logger.append('-' * 60)
            self.logger.append('current lr is {}'.format(self.lr_schedule.get_lr()))
            self._train()
            self._valid()
            self.lr_schedule.step()

    def _dump_infos(self):
        self.logger.append('---------------------Current Parameters---------------------')
        self.logger.append('is use GPU: ' + ('True' if self.is_use_cuda else 'False'))
        self.logger.append('lr: %f' % (self.lr_schedule.get_lr()[0]))
        self.logger.append('model_name: %s' % (self.model_name))
        self.logger.append('current epoch: %d' % (self.cur_epoch))
        self.logger.append('best accuracy: %f' % (self.best_acc))
        self.logger.append('------------------------------------------------------------')

    def _init_infos(self):
        self.logger.append('---------------------Current Parameters---------------------')
        self.logger.append('is use GPU: ' + ('True' if self.is_use_cuda else 'False'))
        self.logger.append('init lr: %f' % (self.lr_schedule.get_lr()[0]))
        self.logger.append('model_name: %s' % (self.model_name))
        self.logger.append('------------------------------------------------------------')

    def _create_masks(self, trg):
        trg_mask = (trg != self.pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        np_mask = Variable(torch.from_numpy(np_mask) == 0)

        if self.is_use_cuda:
            np_mask = np_mask.cuda()
            trg_mask = trg_mask.cuda()
        trg_mask = trg_mask & np_mask
        return trg_mask
    
    def _train(self):
        self.model.train()
        losses = []
        self.metric[0].reset()

        for i, (inputs, real_id_ids, caption_lengths, labels) in enumerate(self.train_data_loader):
            if self.is_use_cuda:
                inputs, real_id_ids, caption_lengths, labels = inputs.cuda(), real_id_ids.cuda(), caption_lengths.cuda(), labels.cuda()

            ids_masks = self._create_masks(real_id_ids)
            self.optimizer.zero_grad()
            
            trg_ids = real_id_ids[:, 1:]
            global_pred, local_pred, global_feature, char_reconstruct, sort_ind = self.model(inputs, real_id_ids, ids_masks, caption_lengths)
            caption_lengths = caption_lengths[sort_ind]
            labels = labels[sort_ind]
            trg_ids = trg_ids[sort_ind]
            _, pred_ids = local_pred.topk(1, 2, True, True)
            pred_ids = pred_ids.squeeze(-1)
            decode_lengths = [x-1 for x in caption_lengths]

            preds_pad = pack_padded_sequence(local_pred, decode_lengths, batch_first=True)
            pred_ids_pad = pack_padded_sequence(pred_ids, decode_lengths, batch_first=True)
            targets_pad = pack_padded_sequence(trg_ids, decode_lengths, batch_first=True)

            global_loss = self.loss_fn[0](global_pred, labels)
            local_loss = self.loss_fn[0](preds_pad.data, targets_pad.data)
            if self.cur_epoch >= self.switch_align:
                algin_loss = self.loss_fn[1](global_feature, char_reconstruct)
                edit_loss = self.loss_fn[2](pred_ids_pad.data, targets_pad.data)
            else:
                algin_loss = 0
                edit_loss = 0

            loss = global_loss + self.alpha_l * local_loss + \
                self.alpha_align *algin_loss + self.alpha_edit * edit_loss

            if self.metric is not None:
                global_prob = F.softmax(global_pred, dim=1).data.cpu()
                self.metric[0].add(global_prob, labels.data.cpu())
                preds_words = nn.Softmax(dim=2)(local_pred).topk(1, dim=2)[1].squeeze(-1)
                preds_words = preds_words.int().cpu().tolist()
                self.metric[1].cal_acc(preds_words, trg_ids.cpu().tolist())
            
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if 0 == i % self.log_batchs or (i == len(self.train_data_loader) - 1):
                local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print_str = '[%s]\tTraining Batch[%d/%d]  Total Loss: %.4f | IDS Loss: %.4f | '\
                            'Global Loss: %.4f | Algin Loss: %.4f | Edit Loss: %.4f | '\
                            'IDS acc: %.2f | Global acc: %.2f'  \
                            % (local_time_str, i, len(self.train_data_loader) - 1, 
                            loss, local_loss, global_loss, algin_loss, edit_loss,
                            self.metric[1].val, self.metric[0].value()[0])
                if i == len(self.train_data_loader) - 1:
                    top1_acc_score = self.metric[0].value()[0]
                    top5_acc_score = self.metric[0].value()[1]
                    print_str += ' Global: @Top-1 Score: %.2f | @Top-5 Score: %.2f | ' % (top1_acc_score, top5_acc_score)
                    print_str += ' IDS: @Top-1 Score: %.2f' % (self.metric[1].avg)
                self.logger.append(print_str)

    def _valid(self):
        self.model.eval()
        losses = []
        if self.metric is not None:
            self.metric[0].reset()

        with torch.no_grad():
            for i, (inputs, real_id_ids, caption_lengths, labels) in enumerate(self.valid_data_loader):
                if self.is_use_cuda:
                    inputs, real_id_ids, caption_lengths, labels = inputs.cuda(), real_id_ids.cuda(), caption_lengths.cuda(), labels.cuda()
                
                ids_masks = self._create_masks(real_id_ids)
                trg_ids = real_id_ids[:, 1:]
                global_pred, local_pred, global_feature, char_reconstruct, sort_ind = self.model(inputs, real_id_ids, ids_masks, caption_lengths)
                caption_lengths = caption_lengths[sort_ind]
                labels = labels[sort_ind]
                trg_ids = trg_ids[sort_ind]
                _, pred_ids = local_pred.topk(1, 2, True, True)
                pred_ids = pred_ids.squeeze(-1)
                decode_lengths = [x-1 for x in caption_lengths]
                
                preds_pad = pack_padded_sequence(local_pred, decode_lengths, batch_first=True)
                pred_ids_pad = pack_padded_sequence(pred_ids, decode_lengths, batch_first=True)
                targets_pad = pack_padded_sequence(trg_ids, decode_lengths, batch_first=True)
                
                global_loss = self.loss_fn[0](global_pred, labels)
                local_loss = self.loss_fn[0](preds_pad.data, targets_pad.data)
                if self.cur_epoch >= self.switch_align:
                    algin_loss = self.loss_fn[1](global_feature, char_reconstruct)
                    edit_loss = self.loss_fn[2](pred_ids_pad.data, targets_pad.data)
                else:
                    algin_loss = 0
                    edit_loss = 0

                loss = global_loss + self.alpha_l * local_loss + \
                    self.alpha_align *algin_loss + self.alpha_edit * edit_loss

                if self.metric is not None:
                    global_prob = F.softmax(global_pred, dim=1).data.cpu()
                    self.metric[0].add(global_prob, labels.data.cpu())
                    preds_words = nn.Softmax(dim=2)(local_pred).topk(1, dim=2)[1].squeeze(-1)
                    preds_words = preds_words.int().cpu().tolist()
                    self.metric[1].cal_acc(preds_words, trg_ids.cpu().tolist())
                losses.append(loss.item())

                if 0 == i % self.log_batchs or (i == len(self.valid_data_loader) - 1):
                    local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    print_str = '[%s]\tValid Batch[%d/%d]  Total Loss: %.4f | IDS Loss: %.4f | '\
                            'Global Loss: %.4f | Algin Loss: %.4f | Edit Loss: %.4f | '\
                            'IDS acc: %.2f | Global acc: %.2f'  \
                            % (local_time_str, i, len(self.valid_data_loader) - 1, 
                            loss, local_loss, global_loss, algin_loss, edit_loss,
                            self.metric[1].val, self.metric[0].value()[0])
                    if i == len(self.valid_data_loader) - 1:
                        local_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                        batch_mean_loss = np.mean(losses)
                        print_str = '[%s]\tValidation: \t Mean Loss: %.4f\t' \
                                    % (local_time_str, batch_mean_loss)
                        top1_acc_score = self.metric[0].value()[0]
                        top5_acc_score = self.metric[0].value()[1]
                        print_str += '@Best Global Acc: %.2f | ' % (self.best_acc)
                        print_str += 'Global: @Top-1 Score: %.2f | @Top-5 Score: %.2f | ' % (top1_acc_score, top5_acc_score)
                        print_str += 'IDS: @Top-1 Score: %.2f | ' % (self.metric[1].avg)
                    self.logger.append(print_str)

        if top1_acc_score >= self.best_acc:
            self.best_acc = top1_acc_score
            self._save_best_model()

    def _save_best_model(self):
        # Save Model
        self.logger.append('Saving Model...')
        state = {
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'cur_epoch': self.cur_epoch + 1,
            'num_epochs': self.num_epochs,
            'model name': self.model_name
        }
        save_path = './checkpoint/' + self.model_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(state, os.path.join(save_path, 'Models_best.ckpt'))

