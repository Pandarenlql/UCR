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

def accuracy(output, target, topk=(1,)):    # -> float 百分值
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    output_sorted, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter_ids(object):
    """
    计算IDS序列的预测精度
    """
    def __init__(self, eos):
        self.eos = eos
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def cal_acc(self, output, target):
        """
        Only cal the top1 recognition accuracy.

        input: output and target, type is list
        output: the recognition of the current batch
        """
        batch_size = len(output)
        _equal = 0.
        trgs = []
        preds = []
        
        # 去掉eos之后的填充
        for i in range(batch_size):
            # 去掉start
            # _trg = target[i][1:]

            _trg = target[i]
            # _ids = ids_sequences[i]
            for ind, real_label in enumerate(_trg):
                if real_label == self.eos:
                    del _trg[ind+1:]
            trgs.append(_trg)

            _pred = output[i]
            for pred_ind, pred_label in enumerate(_pred):
                if pred_label == self.eos:
                    del _pred[pred_ind+1:]
            preds.append(_pred)
        
        for i in range(batch_size):
            if output[i] == trgs[i]:
                _equal += 1.
        self.val = _equal/batch_size * 100
        self.sum += _equal
        self.count += batch_size
        self.avg = self.sum / self.count * 100


"""
class AverageMeter_ids(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def cal_acc(self, output, target):
        # Only cal the top1 recognition accuracy.

        # input: output and target, type is list
        # output: the recognition of the current batch
        batch_size = len(output)
        _equal = 0.

        for i in range(batch_size):
            if output[i] == target[i]:
                _equal += 1.
        
        self.val = _equal/batch_size * 100
        self.sum += _equal
        self.count += batch_size
        self.avg = self.sum / self.count * 100
"""


# 测试自己写用于计算精度的的计数器
if __name__ == '__main__':
    # batch_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    # top1.update(1, 64)
    # print("top1 val is ", top1.val)

    ids_top1 = AverageMeter_ids()
    list1 = [157, 2403, 1364, 10, 4633, 4633, 4633, 4633, 4633, 4633, 4633, 4633,
        4633, 4633, 4633, 4633]
    list2 = [157, 2403, 1364, 10, 3347, 3163, 2758, 89, 2640, 316, 3633, 102, 4343, 
             371, 4182, 0]
    ids_top1.cal_acc(list1, list2)
    print("Current batch acc is ", ids_top1.val, " All batch acc is ", ids_top1.avg)

