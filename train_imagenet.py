import argparse
import os
import shutil
import time
import platform

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


import imagenet_loader


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import copy
import numpy as np
import signal
import sys
import math
from collections import defaultdict
from torch.distributed import get_world_size, get_rank


import model_definitions as models

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
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--model_dir', default='saved_models', dest='model_dir',
                    help='where to save models')
parser.add_argument('-s','--image-size', default=224, type=int, metavar='N',
                    help='size of image side (images are square)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')


parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-lr','--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-n','--experiment_name', default='', type=str,
                    help='name to prepend to experiment saves.')
parser.add_argument('-e', '--validate', dest='validate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-vb', '--virtual-batch-multiplier', default=1, type=int,
                    metavar='N', help='number of forward/backward passes per parameter update')
parser.add_argument('--write_pure_model', action='store_true',
                    help='write pure model file')


parser.add_argument('--dist-url', default='file://sync.file', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

parser.add_argument('-cl','--channels-last','--channels_last', action='store_true',
                    help='Channels last is a new pytorch memory format and it is supposed to be faster')
parser.add_argument('-alrs','--alternate-learning-rate-schedule',action='store_true',
                    help='Use alternate learning rate schedule with learning rate warmup?')

parser.add_argument('-fp16','--fp16',action='store_true',
                    help='use amp fp16 (O1)')

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
    print(args)
    print('starting on', platform.node())
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print('cuda gpus:',os.environ['CUDA_VISIBLE_DEVICES'])
    
    main_stream = torch.cuda.Stream()

    traindir=args.train
    valdir=args.val

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format
    train_transforms = [
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        #transforms.ToTensor(),
        #normalize,
    ]

    train_dataset = imagenet_loader.ImageFolder(
        traindir,
        transforms.Compose(train_transforms))


    print('Found',len(train_dataset),'training instances.')

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True,num_classes=len(train_dataset.classes))
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=len(train_dataset.classes))




    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            #print(p.size())
            nn=1
            for s in list(p.size()):
                
                nn = nn*s
            pp += nn
        return pp

    print("Model has", get_n_params(model), "parameters")

    model = model.cuda().to(memory_format=memory_format)
    

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer,
                                        opt_level='O1',
                                        loss_scale="dynamic",
                                        verbosity=0
                                        )
        print('Got fp16!')

    args.lr = args.lr*float(args.batch_size*args.virtual_batch_multiplier)/256.

    # optionally resume from a checkpoint
    checkpoint=None
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


    if torch.cuda.device_count() >1:
        print('got device count:',torch.cuda.device_count())
        model = torch.nn.DataParallel(model).cuda().to(memory_format=memory_format)

    print('Virtual batch size =', args.batch_size*args.virtual_batch_multiplier)

    # define loss function (criterion) and optimizer
    criteria = []
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
    def top5(output,target):
        return float(accuracy(output, target, topk=(5,))[0])
    def top1(output,target):
        return float(accuracy(output, target, topk=(1,))[0])
    criteria.append({"CL":lambda output,target:nn.CrossEntropyLoss().cuda()(output,target).float(),"top1":top1,"top5":top5})

    
    if args.resume:
        if os.path.isfile(args.resume):
            optimizer.load_state_dict(checkpoint['optimizer'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, collate_fn=fast_collate)

    val_loader = get_val_loader(None, args.val, train_dataset, args)
    validation_sets = [val_loader]

    trainer=Trainer(train_loader,val_loader,model,optimizer,criteria,args,checkpoint)
    if args.validate:
        trainer.progress_table=[]
        trainer.validate([{}])
        print()
        return
    

    trainer.train()
   

def get_val_loader(normalize, valdir, train_dataset, args):

    print(valdir)
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
    if train_dataset is not None:
        val_dataset = imagenet_loader.ImageFolder(valdir, transforms.Compose(val_transforms), class_to_idx_seed=train_dataset.class_to_idx)
    else:
        val_dataset = imagenet_loader.ImageFolder(valdir, transforms.Compose(val_transforms))
    print('Found',len(val_dataset),'validation instances.')
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=max(args.batch_size//2,1), shuffle=False,
        num_workers=args.workers, pin_memory=True,sampler=None, collate_fn=fast_collate)
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



class data_prefetcher():
    def __init__(self, loader, memory_format=torch.contiguous_format):
        self.inital_loader = loader
        self.memory_format=memory_format
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            # self.next_input = None
            # self.next_target = None
            self.loader = iter(self.inital_loader)
            self.preload()
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std).to(memory_format=self.memory_format)
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


def print_table(table_list, go_back=True):
    if len(table_list)==0:
        print()
        print()
        return
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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.std= 0
        self.sum = 0
        self.sumsq = 0
        self.count = 0
        self.lst = []

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        #self.sumsq += float(val)**2
        self.count += n
        self.avg = self.sum / self.count
        self.lst.append(self.val)
        self.std=np.std(self.lst)


class Trainer:
    def __init__(self,train_loader,val_loader,model,optimizer,criteria,args,checkpoint=None):
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.train_prefetcher=data_prefetcher(self.train_loader)
        self.model=model
        self.optimizer=optimizer
        self.criteria=criteria
        self.args = args
        self.fp16=args.fp16
        self.code_archive=self.get_code_archive()
        if checkpoint:
            print()
            print()
            self.progress_table = checkpoint['progress_table']
            self.start_epoch = checkpoint['epoch']+1
            self.best_prec5 = checkpoint['best_prec5']
            self.stats = checkpoint['stats']
            self.loss_history = checkpoint['loss_history']
        else:
            self.progress_table=[]
            self.best_prec5 = 0
            self.stats = []
            self.start_epoch = 0
            self.loss_history=[]
        

        #self.lr0 = get_average_learning_rate(optimizer)
        self.lr0 = self.args.lr
        print_table(self.progress_table,False)
        self.ticks=0
        self.last_tick=0
        #self.loss_tracking_window = args.loss_tracking_window_initial

    def get_code_archive(self):
        file_contents={}
        for i in os.listdir('.'):
            if i[-3:]=='.py':
                with open(i,'r') as file:
                    file_contents[i]=file.read()
        return file_contents

    def train(self):
        for self.epoch in range(self.start_epoch,self.args.epochs):
            if self.args.alternate_learning_rate_schedule:
                self.adjust_learning_rate2()
            else:
                self.adjust_learning_rate()

            # train for one epoch
            train_string, train_stats = self.train_epoch()

            # evaluate on validation set
            progress_string=train_string
            prec5, progress_string, val_stats = self.validate(progress_string)
            print()

            self.progress_table.append(progress_string)

            self.stats.append((train_stats,val_stats))
            self.checkpoint(prec5)

    def checkpoint(self, prec5):
        is_best = prec5 > self.best_prec5
        self.best_prec5 = max(prec5, self.best_prec5)
        save_filename = self.args.experiment_name+'_'+self.args.arch+'_'+('p' if self.args.pretrained != '' else 'np')+'_checkpoint.pth.tar'

        try:
            to_save = self.model
            if torch.cuda.device_count() >1:
                to_save=to_save.module
            gpus='all'
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                gpus=os.environ['CUDA_VISIBLE_DEVICES']
            self.save_checkpoint({
                'epoch': self.epoch,
                'info':{'machine':platform.node(), 'GPUS':gpus},
                'args': self.args,
                'arch': self.args.arch,
                'state_dict': to_save.state_dict(),
                'best_prec5': self.best_prec5,
                'optimizer' : self.optimizer.state_dict(),
                'progress_table' : self.progress_table,
                'stats': self.stats,
                'loss_history': self.loss_history,
                'code_archive':self.code_archive
            }, False, self.args.model_dir, save_filename)

            if args.write_pure_model:
                torch.save(to_save.encoder.state_dict(),args.arch+'.encoder.pth.tar')
            if is_best:
                self.save_checkpoint(None, True,self.args.model_dir, save_filename)
        except:
            print('save checkpoint failed...')



    def save_checkpoint(self,state, is_best,directory='', filename='checkpoint.pth.tar'):
        path = os.path.join(directory,filename)
        if is_best:
            best_path = os.path.join(directory,'best_'+filename)
            shutil.copyfile(path, best_path)
        else:
            torch.save(state, path)





    def train_epoch(self):
        global program_start_time
        average_meters = defaultdict(AverageMeter)
        display_values = []
        for criterion in self.criteria:
            for name,func in criterion.items():
                display_values.append(name)

        # switch to train mode
        self.model.train()

        end = time.time()
        epoch_start_time = time.time()
        epoch_start_time2=time.time()

        batch_num = 0
        num_data_points=len(self.train_loader)//self.args.virtual_batch_multiplier
        #num_data_points//=10
        starting_learning_rate=get_average_learning_rate(self.optimizer)
        while True:
            if batch_num ==0:
                end=time.time()
                epoch_start_time2=time.time()
            if num_data_points==batch_num:
                break
            self.percent = batch_num/num_data_points
            loss_dict=None
            loss=0

            # accumulate gradients over multiple runs of input
            for _ in range(self.args.virtual_batch_multiplier):
                data_start = time.time()
                input, target = self.train_prefetcher.next()
                average_meters['data_time'].update(time.time() - data_start)
                loss_dict2,loss2 = self.train_batch(input,target)
                loss+=loss2
                if loss_dict is None:
                    loss_dict=loss_dict2
                else:
                    for key,value in loss_dict2.items():
                        loss_dict[key]+=value
            
            # divide by the number of accumulations
            loss/=self.args.virtual_batch_multiplier
            for key,value in loss_dict.items():
                loss_dict[key]=value/self.args.virtual_batch_multiplier
            
            # do the weight updates and set gradients back to zero
            self.update()

            self.loss_history.append(float(loss))

            for name,value in loss_dict.items():
                try:
                    average_meters[name].update(value.data)
                except:
                    average_meters[name].update(value)



            elapsed_time_for_epoch = (time.time()-epoch_start_time2)
            eta = (elapsed_time_for_epoch/(batch_num+.2))*(num_data_points-batch_num)
            if eta >= 24*3600:
                eta = 24*3600-1

            
            batch_num+=1
            if True:

                to_print = {}
                to_print['ep']= ('{0}:').format(self.epoch)
                to_print['#/{0}'.format(num_data_points)]= ('{0}').format(batch_num)
                to_print['lr']= ('{0:0.3g}').format(get_average_learning_rate(self.optimizer))
                to_print['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta))))
                
                to_print['d%']=('{0:0.2g}').format(100*average_meters['data_time'].sum/elapsed_time_for_epoch)
                for name in display_values:
                    meter = average_meters[name]
                    to_print[name]= ('{meter.avg:.4g}').format(meter=meter)
                if batch_num < num_data_points-1:
                    to_print['ETA']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta+elapsed_time_for_epoch))))
                print_table(self.progress_table+[[to_print]])
                

        
        epoch_time = time.time()-epoch_start_time
        stats={'batches':num_data_points,
            'learning_rate':get_average_learning_rate(self.optimizer),
            'Epoch time':epoch_time,
            }
        for name in display_values:
            meter = average_meters[name]
            stats[name] = meter.avg

        data_time = average_meters['data_time'].sum

        to_print['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(epoch_time))))
        
        return [to_print], stats



    def train_batch(self, input, target):

        loss_dict = {}
        
        input = input.float()
        output = self.model(input)

        for i, criterion in enumerate(self.criteria):
            first_loss=None
            for c_name,criterion_fun in criterion.items():
                if first_loss is None:first_loss=c_name
                loss_dict[c_name]=criterion_fun(output, target)

        loss = loss_dict[first_loss].clone()
        loss = loss / self.args.virtual_batch_multiplier
            
        if self.args.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss_dict, loss

    
    def update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


    def validate(self, train_table):
        average_meters = defaultdict(AverageMeter)
        self.model.eval()
        epoch_start_time = time.time()
        batch_num=0
        num_data_points=len(self.val_loader)

        prefetcher = data_prefetcher(self.val_loader)
        torch.cuda.empty_cache()
        with torch.no_grad():
            for i in range(len(self.val_loader)):
                input, target = prefetcher.next()


                if batch_num ==0:
                    epoch_start_time2=time.time()
                input = input.float()
                output = self.model(input)
                

                loss_dict = {}
                for criterion in self.criteria:
                    for c_name,criterion_fun in criterion.items():
                        loss_dict[c_name]=criterion_fun(output, target)
                
                batch_num=i+1

                for name,value in loss_dict.items():    
                    try:
                        average_meters[name].update(value.data)
                    except:
                        average_meters[name].update(value)
                eta = ((time.time()-epoch_start_time2)/(batch_num+.2))*(len(self.val_loader)-batch_num)

                to_print = {}
                to_print['#/{0}'.format(num_data_points)]= ('{0}').format(batch_num)
                to_print['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta))))
                for name in criterion.keys():
                    meter = average_meters[name]
                    to_print[name]= ('{meter.avg:.4g}').format(meter=meter)
                progress=train_table+[to_print]
                print_table(self.progress_table+[progress])

        epoch_time = time.time()-epoch_start_time

        stats={'batches':len(self.val_loader),
            'Epoch time':epoch_time,
            }
        ultimate_loss = None
        for name in criterion.keys():
            meter = average_meters[name]
            stats[name]=meter.avg
        self.prec5 = stats['top5']
        to_print['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(epoch_time))))
        torch.cuda.empty_cache()
        return float(self.prec5), progress , stats


    def adjust_learning_rate(self):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if self.epoch < 30:
            self.set_learning_rate(self.lr0)
            return
        if self.epoch < 60:
            self.set_learning_rate(self.lr0*.1)
            return
        self.set_learning_rate(self.lr0*.01)
    def adjust_learning_rate2(self):
        """Use learning rate warmup and then decrease by half every jump to lr0/512@55 epochs"""
        if self.epoch == 0:
            self.set_learning_rate(self.lr0/16)
        elif self.epoch == 1:
            self.set_learning_rate(self.lr0/8)
        elif self.epoch == 2:
            self.set_learning_rate(self.lr0/4)
        elif self.epoch == 3:
            self.set_learning_rate(self.lr0/2)
        elif self.epoch == 4:
            self.set_learning_rate(self.lr0/1.5)
        elif self.epoch == 5:
            self.set_learning_rate(self.lr0)
        elif self.epoch == 6:
            self.set_learning_rate(self.lr0*1.5)
        elif self.epoch < 8:
            self.set_learning_rate(self.lr0*2)
        elif self.epoch < 12:
            self.set_learning_rate(self.lr0)
        elif self.epoch < 15:
            self.set_learning_rate(self.lr0/2)
        elif self.epoch < 20:
            self.set_learning_rate(self.lr0/4)
        elif self.epoch < 25:
            self.set_learning_rate(self.lr0/8)
        elif self.epoch < 30:
            self.set_learning_rate(self.lr0/16)
        elif self.epoch < 35:
            self.set_learning_rate(self.lr0/32)
        elif self.epoch < 40:
            self.set_learning_rate(self.lr0/64)
        elif self.epoch < 55:
            self.set_learning_rate(self.lr0/128)
        elif self.epoch < 58:
            self.set_learning_rate(self.lr0/256)
        else:
            self.set_learning_rate(self.lr0/512)

    def set_learning_rate(self,lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == '__main__':
    #mp.set_start_method('forkserver')
    args = parser.parse_args()
    main(args)
