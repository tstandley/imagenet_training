import argparse
import os
import shutil
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models

import model_definitions as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#import model_summary
#from my_distributed_sampler import DistributedSampler
#import resnet_plus
import imagenet_loader
#import zip_loader
#import sgd_plus
#from fp16util import *
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *

import copy
import numpy as np
import signal
import sys
import math
from collections import defaultdict
from torch.distributed import get_world_size, get_rank




model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--train', '-t', dest='train',
                    help='path to training set')
parser.add_argument('--val', '-v', dest='val',
                    help='path to validation set')
parser.add_argument('--val2', '-v2', dest='val2',default=None,
                    help='path to second validation set')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--model_dir', default='saved_models', dest='model_dir',
                    help='where to save models')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=130, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--image-size', default=224, type=int, metavar='N',
                    help='size of image side (images are square)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--experiment_name', default='', type=str,
                    help='name to prepend to experiment saves.')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('-vb', '--virtual-batch-size', default=1, type=int,
                    metavar='N', help='number of forward/backward passes per gradient update')
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--write_pure_model', action='store_true',
                    help='write pure model file')

# parser.add_argument('--static-loss-scale', type=float, default=1,
#                     help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
# parser.add_argument('--dynamic-loss-scale', action='store_true',
#                     help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
#                     '--static-loss-scale.')

parser.add_argument('--dist-url', default='file://sync.file', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

parser.add_argument('--world-size', default=1, type=int,
                    help='Number of GPUs to use. Can either be manually set ' +
                    'or automatically set by using \'python -m multiproc\'.')
# parser.add_argument('--rank', default=0, type=int,
#                     help='Used for multi-process training. Can either be manually set ' +
#                     'or automatically set by using \'python -m multiproc\'.')

cudnn.benchmark = True
def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
        
    return tensor, targets





def main(args):
    print('starting')
    #args.dist_backend='tcp'
    args.distributed = args.world_size > 1
    #args.gpu = 0
    if args.distributed:
        #args.gpu = args.rank % torch.cuda.device_count()
        #torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size)

    main_stream = torch.cuda.Stream()

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
        print('Got fp16!')
        
    traindir=args.train
    valdir=args.val
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # if args.image_size==299 or 'ception' in args.arch:
    #     print('changing normalization...')
    #     normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                  std=[0.5,0.5,0.5])



    train_transforms = [
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        #transforms.ToTensor(),
        #normalize,
    ]

    # if not args.zip:
    train_dataset = imagenet_loader.ImageFolder(
        traindir,
        transforms.Compose(train_transforms))
    # else:
    #     train_dataset = zip_loader.ZipLoader(
    #         traindir,
    #         transforms.Compose(train_transforms))

    print('Found',len(train_dataset),'training instances.')

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True,num_classes=len(train_dataset.classes))
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=len(train_dataset.classes))
    #print("Model has", model_summary.get_n_params(model), "parameters")



    model = model.cuda()

    # optionally resume from a checkpoint
    progress_table = []
    best_prec5=0
    stats = []
    start_epoch = 0
    resumed = False
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda())

            progress_table = checkpoint['progress_table']
            #print_table(progress_table)
            start_epoch = checkpoint['epoch']
            best_prec5 = checkpoint['best_prec5']
            model.load_state_dict(checkpoint['state_dict'])
            stats = checkpoint['stats']
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            resumed = True
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    o_model = model
    if args.fp16:
        print('making network fp16')
        model = network_to_half(model)
    if args.distributed:
        model = DDP(model)
    if torch.cuda.device_count() >1:
        model = torch.nn.DataParallel(model).cuda()
    
    
    # if args.fp16:
    #     if not resumed:
    #         param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in o_model.parameters()]
    #         for param in param_copy:
    #             param.requires_grad = True
    # else:
    #     param_copy = list(o_model.parameters())


    print('Virtual batch size =', args.batch_size*args.virtual_batch_size*args.world_size)

    # define loss function (criterion) and optimizer
    criteria = []
    #criterion = nn.CrossEntropyLoss().cuda()
    def top5(output,target):
        return float(accuracy(output, target, topk=(5,))[0])
    def top1(output,target):
        return float(accuracy(output, target, topk=(1,))[0])
    criteria.append({"CL":lambda output,target:nn.CrossEntropyLoss().cuda()(output,target).float(),"top1":top1,"top5":top5})

    optimizer = torch.optim.SGD(o_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.fp16:
        sys.stdout = open(os.devnull, "w")
        optimizer = FP16_Optimizer(optimizer,
                                   dynamic_loss_scale=True,
                                   #dynamic_loss_args={'init_scale':2**16},
                                   #static_loss_scale=2**16,
                                   #static_loss_scale=args.static_loss_scale,
                                   #dynamic_loss_scale=args.dynamic_loss_scale
        )
        sys.stdout = sys.__stdout__

    
    if args.resume:
        if os.path.isfile(args.resume):
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('number_of_optimizer_groups=',len(optimizer.param_groups[0]['params']))

    if args.start_epoch!=0:
        start_epoch = args.start_epoch

    
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None #zip_loader.RandomLimitedSampler(train_dataset,12800)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

    val_loader = get_val_loader(None, args.val, train_dataset, args)
    validation_sets = [val_loader]

    if args.evaluate:
        for loader in validation_sets:
            tu.validate(loader, model, criterion,'',confusion_matrix=False)
            print()
        return
    
    train(model,
            optimizer,
            criteria,
            args.epochs,
            train_loader,
            val_loader,
            args,
            start_epoch,
            progress_table,
            best_prec5,
            stats
            )
    

def get_val_loader(normalize, valdir, train_dataset, args):
    if args.image_size == 299:
        val_transforms = [
            transforms.Resize(333),
            transforms.CenterCrop(299),
            #transforms.ToTensor(),
            #normalize,
        ]
    elif args.image_size == 288:
        val_transforms = [
            transforms.Resize(321),
            transforms.CenterCrop(288),
            #transforms.ToTensor(),
            #normalize,
        ]
    else:
        val_transforms = [
            transforms.Resize(int(args.image_size*256/224)),
            transforms.CenterCrop(args.image_size),
            #transforms.ToTensor(),
            #normalize,
        ]
    print(valdir)
    if train_dataset is not None:
        val_dataset = imagenet_loader.ImageFolder(valdir, transforms.Compose(val_transforms), class_to_idx_seed=train_dataset.class_to_idx)
    else:
        val_dataset = imagenet_loader.ImageFolder(valdir, transforms.Compose(val_transforms))
    print('Found',len(val_dataset),'validation instances.')
    if args.distributed:
        val_sampler=torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler=None
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=max(args.batch_size//2,1), shuffle=False,
        num_workers=args.workers, pin_memory=True,sampler=val_sampler, collate_fn=fast_collate)
    return val_loader



program_start_time = time.time()

def on_keyboared_interrupt(x,y):
    #print()
    sys.exit(1)
signal.signal(signal.SIGINT, on_keyboared_interrupt)

def get_average_learning_rate(optimizer):
    try:
        return optimizer.learning_rate
    except:
        s = 0
        for param_group in optimizer.param_groups:
            s+=param_group['lr']
        return s/len(optimizer.param_groups)

def train(model, optimizer, criteria, epochs, train_loader, val_loader,
          args, start_epoch=0, progress_table = [],best_prec5 = 0, stats = []):
    
    print()
    print()
    if args.lr is None:
        if start_epoch==0:
            lr0 = get_average_learning_rate(optimizer)
        else:
            lr0=.1
    else:
        lr0 = args.lr

    for epoch in range(start_epoch,epochs):
        adjust_learning_rate(optimizer, epoch,lr0)

        # train for one epoch
        train_string, train_stats = \
            train_epoch(train_loader, model, criteria, optimizer, epoch,progress_table,fp16=args.fp16,distributed=args.distributed)

        # evaluate on validation set
        progress_string=train_string
        prec5, progress_string, val_stats = validate(val_loader, model, criteria,progress_string,progress_table,fp16=args.fp16,distributed=args.distributed)
        print()

        progress_table.append(progress_string)

        # remember best prec@1 and save checkpoint

        
        is_best = prec5 > best_prec5
        best_prec5 = max(prec5, best_prec5)
        stats.append((train_stats,val_stats))
        save_filename = args.experiment_name+'_'+args.arch+'_checkpoint.pth.tar'
                        
        try:
            to_save = model
            if args.distributed or torch.cuda.device_count() >1:
                to_save=to_save.module
            if args.fp16:
                to_save=to_save[1]
                to_save = to_save.float()
                #copy_in_params(model,param_copy)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': to_save.state_dict(),
                'best_prec5': best_prec5,
                'optimizer' : optimizer.state_dict(),
                'progress_table' : progress_table,
                'stats': stats
            }, False, args.model_dir, save_filename)

            if args.write_pure_model:
                torch.save(to_save.encoder.state_dict(),args.arch+'.encoder.pth.tar')
            if is_best:
                save_checkpoint(None, True,args.model_dir, save_filename)
            if args.fp16:
                to_save = network_to_half(to_save)
        except:
            print('save checkpoint failed...')
            if args.fp16:
                to_save = network_to_half(to_save)



def save_checkpoint(state, is_best,directory='', filename='checkpoint.pth.tar'):
    path = os.path.join(directory,filename)
    if is_best:
        best_path = os.path.join(directory,'best_'+filename)
        shutil.copyfile(path, best_path)
    else:
        torch.save(state, path)

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def train_epoch(train_loader, model, criteria, optimizer, epoch, progress_table, print_freq=1,fp16=False,distributed=False):
    global program_start_time
    average_meters = defaultdict(AverageMeter)
    display_values = []
    for criterion in criteria:
        for name,func in criterion.items():
            display_values.append(name)

    # switch to train mode
    model.train()

    end = time.time()
    epoch_start_time = time.time()
    epoch_start_time2=time.time()
    #print('gonna set epoch')
    if distributed:
        train_loader.sampler.set_epoch(epoch)
    #print('gonna build prefetcher...')
    prefetcher = data_prefetcher(train_loader)
    #print('...prefetcher built')

    batch_num = 0
    num_data_points=len(train_loader)
    while True:
        if batch_num ==0:
            end=time.time()
            epoch_start_time2=time.time()
        if num_data_points==batch_num:
            break

        data_start = time.time()
        input, target = prefetcher.next()
        average_meters['data_time'].update(time.time() - data_start)
        loss_dict = train_batch(model,input,target,optimizer,criteria,fp16=fp16,distributed=distributed)
        update(optimizer,model,fp16=fp16,distributed=distributed)

        for name,value in loss_dict.items():
            try:
                average_meters[name].update(value.data)
            except:
                average_meters[name].update(value)

        # time since start of program:
        #if batch_num==0 and epoch==0:
        #    print('Time since program start:', time.time() - program_start_time, 'seconds.')

        
        
        #eta = (num_data_points - true_batch)*average_meters['batch_time'].avg
        elapsed_time_for_epoch = (time.time()-epoch_start_time2)
        eta = (elapsed_time_for_epoch/(batch_num+.2))*(num_data_points-batch_num)
        if eta >= 24*3600:
            eta = 24*3600-1

        
        batch_num+=1
        if (batch_num-1) % print_freq == 0:

            to_print = {}
            to_print['ep']= ('{0}:').format(epoch)
            to_print['#/{0}'.format(num_data_points)]= ('{0}').format(batch_num)
            to_print['lr']= ('{0:0.3g}').format(get_average_learning_rate(optimizer))
            to_print['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta))))
            
            to_print['d%']=('{0:0.2g}').format(100*average_meters['data_time'].sum/elapsed_time_for_epoch)
            for name in display_values:
                meter = average_meters[name]
                to_print[name]= ('{meter.avg:.4g}').format(meter=meter)
            if batch_num < num_data_points-1:
                to_print['ETA']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta+elapsed_time_for_epoch))))
            print_table(progress_table+[[to_print]])
            

    
    epoch_time = time.time()-epoch_start_time
    stats={'batches':num_data_points,
           'learning_rate':get_average_learning_rate(optimizer),
           'Epoch time':epoch_time,
           }
    for name in display_values:
        meter = average_meters[name]
        stats[name] = meter.avg

    data_time = average_meters['data_time'].sum

    to_print['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(epoch_time))))
    
    return [to_print], stats


def print_table(table_list, go_back=True):

    if go_back:
        print("\033[F",end='')
        print("\033[K",end='')
        for i in range(len(table_list)):
            print("\033[F",end='')
            print("\033[K",end='')


    lens = defaultdict(int)
    for i in table_list:
        for ii,to_print in enumerate(i):
            for title,val in to_print.items():
                lens[(title,ii)]=max(lens[(title,ii)],max(len(title),len(val)))
    

    # printed_table_list_header = []
    for ii,to_print in enumerate(table_list[0]):
        for title,val in to_print.items():

            print('{0:^{1}}'.format(title,lens[(title,ii)]),end=" ")
    for i in table_list:
        print()
        for ii,to_print in enumerate(i):
            for title,val in to_print.items():
                print('{0:^{1}}'.format(val,lens[(title,ii)]),end=" ",flush=True)
    print()

def train_batch(model, input, target, optimizer, criteria,fp16=False,distributed=False):

    loss_dict = {}
    for i, criterion in enumerate(criteria):
        if fp16:
            input = input.half()
        else:
            input = input.float()
        output = model(input)
        #if fp16:
        #    output = output.half()
        output=output.float()
        first_loss=None
        for c_name,criterion_fun in criterion.items():
            if first_loss is None:first_loss=c_name
            loss_dict[c_name]=criterion_fun(output, target)

        loss = loss_dict[first_loss].clone()
        if fp16:
            sys.stdout = open(os.devnull, "w")
        
            optimizer.backward(loss)
            sys.stdout = sys.__stdout__
        else:
            loss.backward()

    return loss_dict

    
def update(optimizer,model,fp16=False,distributed=False):

    sys.stdout = open(os.devnull, "w")
    optimizer.step()
    sys.stdout = sys.__stdout__
    if fp16:
        torch.cuda.synchronize()
    sys.stdout = open(os.devnull, "w")
    optimizer.zero_grad()
    sys.stdout = sys.__stdout__
 

def validate(val_loader, model, criteria, train_table, progress_table,fp16=False,confusion_matrix=False,distributed=False):
    average_meters = defaultdict(AverageMeter)
    model.eval()
    epoch_start_time = time.time()
    batch_num=0
    num_data_points=len(val_loader)

    prefetcher = data_prefetcher(val_loader)
    
    with torch.no_grad():
        for i in range(len(val_loader)):
            input, target = prefetcher.next()


            if batch_num ==0:
                epoch_start_time2=time.time()

            output = model(input)
            

            loss_dict = {}
            for criterion in criteria:
                for c_name,criterion_fun in criterion.items():
                    loss_dict[c_name]=criterion_fun(output, target)
            
            batch_num=i+1

            for name,value in loss_dict.items():    
                try:
                    average_meters[name].update(value.data)
                except:
                    average_meters[name].update(value)
            eta = ((time.time()-epoch_start_time2)/(batch_num+.2))*(len(val_loader)-batch_num)

            to_print = {}
            to_print['#/{0}'.format(num_data_points)]= ('{0}').format(batch_num)
            to_print['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta))))
            for name in criterion.keys():
                meter = average_meters[name]
                to_print[name]= ('{meter.avg:.4g}').format(meter=meter)
            progress=train_table+[to_print]
            print_table(progress_table+[progress])



    epoch_time = time.time()-epoch_start_time

    stats={'batches':len(val_loader),
           'Epoch time':epoch_time,
           }
    ultimate_loss = None
    for name in criterion.keys():
        meter = average_meters[name]
        stats[name]=meter.avg
    prec5 = stats['top5']
    to_print['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(epoch_time))))

    return float(prec5), progress , stats



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
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count
        

def adjust_learning_rate(optimizer, epoch,lr0,period=6):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 20:
        set_learning_rate(optimizer, lr0)
        return
    if epoch < 45:
        set_learning_rate(optimizer, lr0*.1)
        return
    set_learning_rate(optimizer, lr0*.01)
    # if epoch >= 4: epoch +=2
    # lr = lr0 * (0.60 ** (epoch // period))
    # set_learning_rate(optimizer,lr)

def set_learning_rate(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    #mp.set_start_method('forkserver')
    args = parser.parse_args()
    main(args)
