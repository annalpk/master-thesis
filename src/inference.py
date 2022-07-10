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


# Custom includes
from model.videoseq import VideoSOD

# Dataloaders includes
from dataloaders import davis, visal
from dataloaders import image_transforms as trforms


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-epochs', type=int, default=1)

    ## Model settings
    parser.add_argument('-model_name', type=str, default='VideoSOD')
    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('-input_size', type=int, default=416)
    parser.add_argument('-output_stride', type=int, default=16)
    parser.add_argument('-seq_len', type=int, default=4)

    ## Visualization settings
    parser.add_argument('-load_path', type=str, default='path/to/weights')
    parser.add_argument('-save_dir', type=str, default='./results')

    parser.add_argument('-save_mask', type=bool, default=True)

    parser.add_argument('-test_dataset', type=str, default='DAVIS-Seq', choices=['DAVIS-Seq', 'DAVIS-valset', 'FBMS', 'ViSal', 'DUTOMRON', 'DUTS'])
    parser.add_argument('-data_dir', type=str, default='/Datasets/DAVIS-data/DAVIS')
    parser.add_argument('-test_fold'      , type=str  , default='/val')

    return parser.parse_args()


def softmax_2d(x):
    return torch.exp(x) / torch.sum(torch.sum(torch.exp(x), dim=-1, keepdim=True), dim=-2, keepdim=True)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = VideoSOD(nInputChannels=3, n_classes=1, os=16, img_backbone_type='resnet101', bidirectional=True, bias=False, device=device)
    net.to(device)

    loss_fn = nn.L1Loss()
    val_losses = []

    # load pre-trained Appearance network weights
    if args.load_path is None:
        pretrain_weights = torch.load("MGA_trained.pth")
        pretrain_keys = list(pretrain_weights.keys())
        # pretrain_keys = [key for key in pretrain_keys if not key.endswith('num_batches_tracked')]
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
    epoch_start = 0
    # load pre-trained weights
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path)
        epoch_start = checkpoint['epoch']
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
    print(epoch_start)


    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    if args.test_dataset == 'DAVIS-Seq':
        test_data = davis.DAVIS(dataset='val', data_dir=args.data_dir, transform=composed_transforms_ts, seq_len=args.seq_len, return_size=True)
    elif args.train_dataset == 'ViSal':
        test_data = visal.ViSal(dataset='val', data_dir=args.data_dir, transform=composed_transforms_ts, return_size=True)

    save_dir = args.save_dir + args.test_fold + '-' + args.model_name + '-' + args.test_dataset + '/saliency_map/'
    testloader = DataLoader(test_data, args.batch_size, shuffle=False, num_workers=12)

    net.eval()

    val_losses = []
    val_start = time.time()

    with torch.no_grad():
        overall_loss = 0
        overall_speed = 0
        total_pixel = 0
        correct_pixel = 0
        sum_accuracy = 0
        for i, sample_batched in enumerate(testloader):
            inputs, labels, label_name, size = sample_batched['images'], sample_batched['labels'], sample_batched['label_name'], sample_batched['size']
            inputs.requires_grad_(False)
            labels.requires_grad_(False)
            inputs = inputs.to(device)
            labels = labels.to(device)

            batch_start = time.time()
            prob_pred = net(inputs)
            prob_pred = torch.nn.Sigmoid()(prob_pred)
            batch_speed = time.time() - batch_start
            fps = args.seq_len / batch_speed
            overall_speed += fps

            shape = labels.shape[3:]
            size_shape = (size[0][0, 0], size[1][0, 1])
            prob_pred = (prob_pred - torch.min(prob_pred) + 1e-8) / (torch.max(prob_pred) - torch.min(prob_pred) + 1e-8)
            pred_list = []
            for t in range(args.seq_len):
                up_prob_pred = F.upsample(prob_pred[:, t, :, :, :], size=shape, mode='bilinear', align_corners=True)
                pred_list.append(up_prob_pred)

            save_list = []
            for t in range(args.seq_len):
                up_prob_pred = F.upsample(prob_pred[:, t, :, :, :], size=size_shape, mode='bilinear', align_corners=True)
                save_list.append(up_prob_pred)
            prob_pred = torch.stack(pred_list, dim=1)
            save_pred = torch.stack(save_list, dim=1).cpu().numpy()
            loss = loss_fn(prob_pred, labels)
            overall_loss += loss

            pred = prob_pred.cpu().data.numpy()
            label_data = labels.cpu().data.numpy()
            pred[pred > 0.5] = 1
            pred[pred <= 0.5]= 0
            #inputs_data = inputs.cpu().data.numpy()

            # accuracy
            batch_total_pixel = label_data.size
            total_pixel += batch_total_pixel
            batch_correct_pixel = np.equal(pred, label_data).sum()
            correct_pixel += batch_correct_pixel
            accuracy = 100 * batch_correct_pixel / batch_total_pixel
            sum_accuracy += accuracy

            #save_pred[save_pred > 0.5] = 1
            #save_pred[save_pred <= 0.5] = 0
            sequence_data = save_pred[0]
            for t in range(args.seq_len):
                save_png = sequence_data[t][0]
                save_png = np.round(save_png * 255)
                save_png = save_png.astype(np.uint8)
                save_png = Image.fromarray(save_png)
                save_path = save_dir + "/" + label_name[t][0]
                if not os.path.exists(save_path[:save_path.rfind('/')]):
                    os.makedirs(save_path[:save_path.rfind('/')])
                save_png.save(save_path, format="png")

            if args.save_mask:
                for t in range(args.seq_len):
                    mask_png = label_data[0][t][0]



            # print out results
            print("Loss: {}".format(loss))
            val_losses.append(loss)
            print("Batch accuracy: {}".format(accuracy))
            print("Batch run time: {}".format(batch_speed))
            print("Batch speed: {} fps".format(args.seq_len / batch_speed))
        print("Overall speed: {} fps".format(overall_speed / len(testloader)))
        print("Overall loss: {}".format(overall_loss / len(testloader)))
        print("Overall accuracy: {}".format(sum_accuracy / len(testloader)))

    val_end = time.time()
    print("Complete Validation time: {}s".format(val_end - val_start))


if __name__ == '__main__':
    args = get_arguments()
    main(args)