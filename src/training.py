import argparse
from datetime import datetime
import time
import glob
import os
from PIL import Image
import numpy as np
import time

# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR

# Custom includes
from model.videoseq import VideoSOD

# Dataloaders includes
from dataloaders import davis, fbms, visal, dutomron, duts, davisseq
from dataloaders import image_transforms as trforms


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-epochs', type=int, default=1)

    ## Model settings
    parser.add_argument('-model_name', type=str, default='VideoSOD')
    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('-input_size', type=int, default=512)
    parser.add_argument('-output_stride', type=int, default=16)
    parser.add_argument('-seq_len', type=int, default=4)

    ## Visualization settings
    parser.add_argument('-load_path', type=str, default=None)
    parser.add_argument('-save_dir', type=str, default='./results')

    parser.add_argument('-train_dataset', type=str, default='DAVIS-Seq', choices=['DAVIS-Seq', 'DAVIS-valset', 'FBMS', 'ViSal', 'DUTOMRON', 'DUTS'])
    parser.add_argument('-train_fold', type=str, default='/train')

    return parser.parse_args()


def softmax_2d(x):
    return torch.exp(x) / torch.sum(torch.sum(torch.exp(x), dim=-1, keepdim=True), dim=-2, keepdim=True)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = VideoSOD(nInputChannels=3, n_classes=1, os=16, img_backbone_type='resnet101', hidden_dim=256, kernel_size=3, padding=1, num_layers=1, bias=False, device=device)
    net.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(net.convLSTM.parameters(), lr=0.0001, momentum=0.0005, weight_decay=0.9)
    #scheduler = ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.1, threshold=0.0001, verbose=True)
    scheduler = CyclicLR(opt, base_lr=0.0001, max_lr=0.01, step_size_up=100, step_size_down=100)
    #scheduler = StepLR(opt, step_size=100, gamma=0.1)
    scaler = GradScaler()

    epoch_start = -1
    train_losses = []
    val_losses = []

    if args.load_path is None:
    # load pre-trained Appearance network weights
        pretrain_weights = torch.load("MGA_trained.pth")
        pretrain_keys = list(pretrain_weights.keys())
        #pretrain_keys = [key for key in pretrain_keys if not key.endswith('num_batches_tracked')]
        net_keys = list(net.state_dict().keys())

        for key in net_keys:
            # key_ = 'module.' + key
            key_ = key
            if key_ in pretrain_keys:
                assert (net.state_dict()[key].size() == pretrain_weights[key_].size())
                net.state_dict()[key].copy_(pretrain_weights[key_])
            else:
                print('missing key: ', key_)
        print('loaded pre-trained weights.')

    # load pre-trained weights
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path)
        epoch_start = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        opt.load_state_dict(checkpoint['opt_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        pretrain_weights = checkpoint['net_state_dict']
        pretrain_keys = list(pretrain_weights.keys())
        pretrain_keys = [key for key in pretrain_keys]
        net_keys = list(net.state_dict().keys())

        for key in net_keys:
            # key_ = 'module.' + key
            key_ = key
            if key_ in pretrain_keys:
                assert (net.state_dict()[key].size() == pretrain_weights[key_].size())
                net.state_dict()[key].copy_(pretrain_weights[key_])
            else:
                print('missing key: ', key_)
        print('loaded pre-trained weights and states')
        print("Current Learning rate: {}".format(opt.param_groups[0]['lr']))

    # set the appearance branch on not requiring grad, since we don't want to train it
    for name, p in net.named_parameters():
        p.requires_grad = False
        if "convLSTM" in name:
            p.requires_grad = True
        print(name, p.requires_grad)
    print("froze appearance branch tensors")

    print("Starting Epoch = {}".format(epoch_start + 1))

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    if args.train_dataset == 'DAVIS-Seq':
        train_data = davisseq.DAVISSeq(dataset='train', transform=composed_transforms_ts, seq_len=args.seq_len, return_size=True)
        val_data = davisseq.DAVISSeq(dataset='val', transform=composed_transforms_ts, seq_len=args.seq_len, return_size=True)
    elif args.train_dataset == 'FBMS':
        train_data = fbms.FBMS(dataset='train', transform=composed_transforms_ts, return_size=True)
    elif args.train_dataset == 'ViSal':
        train_data = visal.ViSal(dataset='train', transform=composed_transforms_ts, return_size=True)
    elif args.train_dataset == 'DUTOMRON':
        train_data = dutomron.DUTOMRON(dataset='train', transform=composed_transforms_ts, return_size=True)
    elif args.train_dataset == 'DUTS':
        train_data = duts.DUTS(dataset='train', transform=composed_transforms_ts, return_size=True)
        val_data = duts.DUTS(dataset='val', transform=composed_transforms_ts, return_size=True)

    #save_dir = args.save_dir + args.train_fold + '-' + args.model_name + '-' + args.train_dataset + '/saliency_map/'
    trainloader = DataLoader(train_data, args.batch_size, shuffle=False, num_workers=12)
    valloader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=12)
    num_iter_ts = len(trainloader)


    cnt = 0
    accmu_t = 0

    #gradient_accumulations = 16
    net.zero_grad()
    start_training = time.time()
    for epoch in range(epoch_start + 1, args.epochs):
        running_loss = 0.0
        running_val_loss = 0.0
        net.train()
        start_epoch = time.time()
        for i, sample_batched in enumerate(trainloader):
            #print("progress {}/{}\n".format(i, num_iter_ts))

            before_t = time.time()
            inputs, labels, label_name, size = sample_batched['images'], sample_batched['labels'], sample_batched[
                'label_name'], sample_batched['size']
            inputs.requires_grad_(True)
            labels.requires_grad_(False)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with autocast():
                prob_pred = net(inputs)
            #prob_pred = torch.nn.Sigmoid()(prob_pred)
            accmu_t += (time.time() - before_t)
            cnt += 1

            #prob_pred = (prob_pred - torch.min(prob_pred) + 1e-3) / (torch.max(prob_pred) - torch.min(prob_pred) + 1e-3)
            shape = labels.shape[3:]
            pred_list = []
            for t in range(args.seq_len):
                up_prob_pred = F.upsample(prob_pred[:, t, :, :, :], size=shape, mode='bilinear', align_corners=True)
                pred_list.append(up_prob_pred)
            prob_pred = torch.stack(pred_list, dim=1)
            with autocast():
                loss = loss_fn(prob_pred, labels)
            scaler.scale(loss).backward()
            running_loss += loss.item()

            #if (i + 1) % gradient_accumulations == 0:
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            #opt.step()
            opt.zero_grad()
            net.zero_grad()
        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch + 1, args.epochs, train_loss)))
        end_epoch = time.time()
        print("Epoch training time: {}s".format(end_epoch - start_epoch))

        with torch.no_grad():
            net.eval()
            for i, sample_batched in enumerate(valloader):
                inputs, labels, label_name, size = sample_batched['images'], sample_batched['labels'], sample_batched['label_name'], sample_batched['size']
                inputs.requires_grad_(False)
                labels.requires_grad_(False)
                inputs = inputs.to(device)
                labels = labels.to(device)

                prob_pred = net(inputs)
                #prob_pred = torch.nn.Sigmoid()(prob_pred)
                shape = labels.shape[3:]
                #prob_pred = (prob_pred - torch.min(prob_pred) + 1e-8) / (torch.max(prob_pred) - torch.min(prob_pred) + 1e-8)
                pred_list = []
                for t in range(args.seq_len):
                    up_prob_pred = F.upsample(prob_pred[:, t, :, :, :], size=shape, mode='bilinear', align_corners=True)
                    pred_list.append(up_prob_pred)
                prob_pred = torch.stack(pred_list, dim=1)
                loss = loss_fn(prob_pred, labels)

                running_val_loss += loss.item()
        val_loss = running_val_loss / len(valloader)
        val_losses.append(val_loss)
        print(print('Epoch [{}/{}], Val-loss: {:.7f}'.format(epoch + 1, args.epochs, val_loss)))
        #print("Current Learning rate: {}".format(scheduler.get_last_lr()))

        torch.save({
            'epoch': epoch,
            'net_state_dict': net.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_losses': val_losses,
            'train_losses': train_losses,
        }, 'trainseq.pt')
        print('Checkpoint saved')

    end_training = time.time()
    print("Complete training time: {}s".format(end_training - start_training))


if __name__ == '__main__':
    args = get_arguments()
    main(args)