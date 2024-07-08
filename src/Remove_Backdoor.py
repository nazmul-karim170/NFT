import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import copy
import math
import networks
import torch.nn.functional as F
import pandas as pd
import data.badnets_blend as poison
from torch.autograd import Variable
from PIL import Image
from data.dataloader_cifar import *
import matplotlib.pyplot as plt
import random 

def main(parser, transform_train, transform_test):
    ## Set the preliminary settings, e.g. radnom seed 
    args = parser.parse_args()
    args_dict = vars(args)
    random.seed(123)
    os.makedirs(args.output_dir, exist_ok=True)
    device =  'cpu'
    torch.cuda.set_device(args.gpuid)

    ## Clean Test Loader (Badnets and Blend)
    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)
    
    ## Triggers 
    triggers = {'badnets': 'checkerboard_1corner',
                'CLB': 'fourCornerTrigger',
                'blend': 'gaussian_noise',
                'SIG': 'signalTrigger',
                'TrojanNet': 'trojanTrigger',
                'FC': 'gridTrigger',
                'benign': None}

    if args.poison_type == 'badnets':
        args.trigger_alpha = 0.6
    elif args.poison_type == 'blend':
        args.trigger_alpha = 0.2
    
    ## Step 1: create datasets -- clean val set, poisoned test set (exclude target labels)
    if args.poison_type in ['badnets', 'blend']:
        trigger_type  = triggers[args.poison_type]
        pattern, mask = poison.generate_trigger(trigger_type=trigger_type)
        backdoor_trigger  = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                        'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.target_label])}

        poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=backdoor_trigger)                   ## To check how many of the poisonous sample is correctly classified to their "target labels"
        poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)

    elif args.poison_type in ['Dynamic']:
        transform_test = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ])
        if args.target_type =='all2one':
            poisoned_data = Dataset_npy(np.load(args.poisoned_data_test_all2one, allow_pickle=True), transform = None)
        else:
            poisoned_data = Dataset_npy(np.load(args.poisoned_data_test_all2all, allow_pickle=True), transform = None)

        poison_test_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=args.batch_size,
                                        shuffle=False)
        clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)


    elif args.poison_type in ['Feature']:

        transform_test = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ])        
        
        poisoned_data = Dataset_Feature_npy(np.load(args.poisoned_data_test_all2one, allow_pickle=True), mode = 'test', transform = transform_test)
        poison_test_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=args.batch_size,
                                        shuffle=True)
        clean_test_loader   = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)
        trigger_info = None


    elif args.poison_type in ['SIG', 'TrojanNet', 'CLB']:
        trigger_type      = triggers[args.poison_type]
        args.trigger_type = trigger_type        

        ## SIG and CLB are Clean-label Attacks 
        if args.poison_type in ['SIG', 'CLB']:
            args.target_type = 'cleanLabel'
        
        _, poison_test_loader = get_test_loader(args)
        clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

    elif args.poison_type in ['Composite']:
        # poison set (for testing)
        poi_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=preprocess)
        poi_set = MixDataset(dataset=poi_set, mixer=mixer, classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                             data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.1, transform=None)
        poison_test_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=True)

    elif args.poison_type == 'benign':
        poison_test_loader  = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)
        clean_test_loader   = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

    ## Step 1.1: Get the dataloader for Mask finetuning 
    cifar10_train = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    _, clean_val = poison.split_dataset(dataset=cifar10_train, val_frac=args.val_ratio,
                                        perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
    sampler = RandomSampler(data_source=clean_val, replacement=True,
                                   num_samples =args.epoch_aggregation * args.batch_size)
    clean_val_loader  = DataLoader(clean_val, batch_size=args.batch_size,
                                  shuffle=False, sampler=sampler, num_workers=0)


    ## Step 2: Load Model Checkpoints
    state_dict = torch.load(args.checkpoint, map_location=device)
    if args.poison_type in ['Dynamic']:
        state_dict = torch.load(args.checkpoint, map_location=device)['netC']

    net = getattr(networks, args.arch)(num_classes=10, BN_layer = networks.Masked_BN2d)                ## For Mask-finetuning 
    load_model(net, orig_state_dict=state_dict)
    net = net.cuda()

    ## Step 3: Training Settings
    criterion = torch.nn.CrossEntropyLoss().cuda()
    parameters  = list(net.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.95)                            ## For Mask-finetuning
    nb_iterations = int(np.ceil(args.nb_epochs / args.epoch_aggregation))


    # # Step 3: train backdoored models
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mask_optimizer, nb_iterations, 0.02)
    N_c = len(clean_val)/args.num_classes  


    ## Step 4: Validate the Given Model 
    cl_test_loss, ACC =NFT_Test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_test_loss, ASR =NFT_Test(model=net, criterion=criterion, data_loader=poison_test_loader)
    print("ASR and ACC Before Purification\t")
    print('-----------------------------------------------------------------')
    print('ASR \t ACC')
    print('{:.4f} \t {:.4f}'.format(100*ASR, 100*ACC))
    print('-----------------------------------------------------------------')
    print("validation Size:", len(clean_val))
    print("Number of Samples per Class:", N_c)


    ## Losses and Accuracy 
    clean_losses  = np.zeros(nb_iterations)
    poison_losses = np.zeros(nb_iterations)
    clean_accs    = np.zeros(nb_iterations)
    poison_accs   = np.zeros(nb_iterations)

    
    ## Step 5: Purification Process Starts
    print('-----------------------------------------------------------------')
    print("ASR and ACC After Purification\t")
    print('-----------------------------------------------------------------')
    print('Iter \t ASR \t \t ACC')
    for i in range(nb_iterations):
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = NFT_Train(args, N_c, model=net, criterion=criterion, data_loader=clean_val_loader,
                                           mask_opt=mask_optimizer)

        clean_loss , ACC = NFT_Test(model=net, criterion=criterion, data_loader=clean_test_loader)
        poison_loss, ASR = NFT_Test(model=net, criterion=criterion, data_loader=poison_test_loader)

        clean_losses[i]  = clean_loss
        poison_losses[i] = poison_loss
        clean_accs[i]    = ACC
        poison_accs[i]   = ASR

        ## Save Stattistics and the Purified model
        np.savez(os.path.join(args.output_dir,'remove_model_'+ args.poison_type + '_' + str(args.dataset) + '_.npz'), cl_loss = clean_losses, cl_test = clean_accs, po_loss = poison_losses, po_acc = poison_accs)
        model_save = args.poison_type + '_' + str(i) + '_' + str(args.dataset) + '.pth'
        torch.save(net.state_dict(), os.path.join(args.output_dir, model_save))
        scheduler.step()

        print('{} \t {:.4f} \t {:.4f}'.format((i + 1) * args.epoch_aggregation, 100*ASR, 100*ACC))

## Loading the Pre-trained Weights to the Current Model
def load_model(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v

    net.load_state_dict(new_state_dict)

## Mask Regularization
def Regularization(model):
    L1=0
    L2=0
    L_inf = 0
    for name, param in model.named_parameters():
        if 'neuron_mask' in name:
            L1 += torch.sum(torch.abs(1-param))
            L2 += torch.norm(param, 2)
            L_inf += torch.max(torch.abs(1-param))
    # for name, module in model.named_parameters():
    return L1, L2, L_inf

## Clip the mask within [mu, 1]
def mask_clip(args, model, upper=1):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    count_layer = 1
    with torch.no_grad():
        for param in params:
            param.clamp_(args.alpha*math.exp(-args.beta*count_layer), upper)
            count_layer += 1

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def NFT_Train(args, N_c, model, criterion, mask_opt, data_loader):
    model.train()
    total_correct = 0
    total_loss    = 0.0
    nb_samples    = 0

    ## Train the model for 1 epoch
    for i, (images, labels) in enumerate(data_loader):
        nb_samples += images.size(0)
        inputs, targets = images.cuda(), labels.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       alpha=1, use_cuda=True)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        mask_opt.zero_grad()
        L1, L2, L_inf = Regularization(model)
        tot_loss     = loss + 0.001*L1/N_c
        
        tot_loss.backward()
        mask_opt.step()
        mask_clip(args,model)

        ## Claculate the train accuracy 
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        total_loss += tot_loss.item()


    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

def NFT_Test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), torch.squeeze(labels.cuda())
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = torch.max(output,1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove Backdoor Through Neural Fine-Tuning')

    # Basic model parameters.
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be pruned')
    parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
    parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')       
    parser.add_argument('--lr', type=float, default=0.05, help='the learning rate for mask optimization')   
    parser.add_argument('--nb-epochs', type=int, default=1000, help='the number of iterations for training')  
    parser.add_argument('--epoch-aggregation', type=int, default=200, help='print results every few iterations')  
    parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
    parser.add_argument('--val-ratio', type=float, default=0.01, help='The fraction of the validate set')  ## Controls the validation size
    parser.add_argument('--output-dir', type=str, default='save/purified_networks/')
    parser.add_argument('--gpuid', type=int, default=0, help='the transparency of the trigger pattern.')

    parser.add_argument('--poison-type', type=str, default='badnets', choices=['badnets', 'Feature', 'FC',  'SIG', 'Dynamic', 'TrojanNet', 'blend', 'CLB', 'benign'],
                        help='type of backdoor attacks used during training')
    parser.add_argument('--trigger-alpha', type=float, default=0.2, help='the transparency of the trigger pattern.')

    parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--load_fixed_data', type=int, default=1, help='load the local poisoned test dataest')
    parser.add_argument('--poisoned_data_test_all2one', type=str, default='./data/dynamic/poisoned_data/cifar10-test-inject0.1-target0-dynamic-all2one.npy', help='random seed')
    parser.add_argument('--poisoned_data_test_all2all', type=str, default='./data/dynamic/poisoned_data/cifar10-test-inject0.1-target0-dynamic-all2all_mask.npy', help='random seed')

    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='squareTrigger', choices=['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                                   'signalTrigger', 'trojanTrigger'], help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=1, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=1, help='height of trigger pattern')    
    parser.add_argument('--alpha', type=float, default=0.8, help='Search area design Parameter')
    parser.add_argument('--beta', type=float, default=0.5, help='Search area design Parameter')
    parser.add_argument('--num_classes', type=float, default=10, help='Number of classes')

    # Linear Transformation
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10  = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    main(parser, transform_train, transform_test)
