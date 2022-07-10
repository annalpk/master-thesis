import argparse
import time

# PyTorch includes
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
import torch.nn as nn
#from torch.cuda.amp import GradScaler, autocast

# Custom includes
from model.videoseq import VideoSOD
from earlystopping import EarlyStopping

# Dataloaders includes
from dataloaders import davis, fbms, segtrack
from dataloaders import image_transforms as trforms


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-epochs', type=int, default=1)

    ## Model settings
    parser.add_argument('-model_name', type=str, default='VideoSOD')
    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('-input_size', type=int, default=416)
    parser.add_argument('-output_stride', type=int, default=16)
    parser.add_argument('-seq_len', type=int, default=4)

    ## Training settings
    parser.add_argument('-patience', type=int, default=25)

    ## Visualization settings
    parser.add_argument('-load_path', type=str, default=None)
    parser.add_argument('-save_dir', type=str, default='./results')

    parser.add_argument('-train_dataset', type=str, default='Concat', choices=['DAVIS-Seq', 'DAVIS-valset', 'FBMS', 'DUTOMRON', 'Concat'])
    parser.add_argument('-data_dir', type=str, default='./Datasets/DAVIS-data/DAVIS')
    parser.add_argument('-train_fold', type=str, default='/train')
    parser.add_argument('-save_weights', type=str, default='weights.pt')

    return parser.parse_args()


def softmax_2d(x):
    return torch.exp(x) / torch.sum(torch.sum(torch.exp(x), dim=-1, keepdim=True), dim=-2, keepdim=True)

def set_bn_eval(net):
    for module in net.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)

            module.track_running_stats = False
            module.eval()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = VideoSOD(nInputChannels=3, n_classes=1, os=16, img_backbone_type='resnet101', bidirectional=True, bias=False, device=device)
    net.to(device)
    torch.cuda.empty_cache()

    # set the appearance branch on not requiring grad, since we don't want to train it
    for name, p in net.named_parameters():
        p.requires_grad = False
        if "convLSTM" in name:
            p.requires_grad = True
        if "end_conv" in name:
            p.requires_grad = True
    print("froze appearance branch tensors")

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3, momentum=0.9,
                          weight_decay=0.0005)
    #scaler = GradScaler()

    epoch_start = -1
    train_losses = []
    val_losses = []
    best_score = None
    patience = args.patience
    pat_count = 0
    print("Patience: {}".format(patience))

    # load pre-trained Appearance network weights
    if args.load_path is None:
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
            elif "concat_conv" in key:
                key_ = key.replace("concat_conv", "last_conv")
                net.state_dict()[key].copy_(pretrain_weights[key_])
                print("Copied Last Conv for Concat Conv")
            else:
                print('missing key: ', key_)
        print('loaded pre-trained weights.')

    # load pre-trained weights
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path)
        epoch_start = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        best_score = checkpoint['best_score']
        #opt.load_state_dict(checkpoint['opt_state_dict'])
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

    print("Starting Epoch = {}".format(epoch_start + 1))

    for name, p in net.named_parameters():
        print(name, p.requires_grad)

    print("Current Best Score: {}".format(best_score))

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])


    if args.train_dataset == 'DAVIS-Seq':
        print("DAVIS dataset")
        train_data = davis.DAVIS(dataset='train', data_dir=args.data_dir, transform=composed_transforms_ts, seq_len=args.seq_len, return_size=True)
        val_data = davis.DAVIS(dataset='val', data_dir=args.data_dir, transform=composed_transforms_ts, seq_len=args.seq_len, return_size=True)
    elif args.train_dataset == 'FBMS':
        print("FBMS dataset")
        train_data = fbms.FBMS(dataset='train', data_dir=args.data_dir, transform=composed_transforms_ts, seq_len=args.seq_len, return_size=True)
    elif args.train_dataset == 'Concat':
        print("DAVIS + FBMS + SegTrackV2 dataset")
        davis_data = davis.DAVIS(dataset='train', data_dir=args.data_dir, transform=composed_transforms_ts, seq_len=args.seq_len, return_size=True)
        val_data = davis.DAVIS(dataset='val', data_dir=args.data_dir, transform=composed_transforms_ts, seq_len=args.seq_len, return_size=True)
        fbms_data = fbms.FBMS(dataset='train', data_dir='./Datasets/FBMS_Trainingset', transform=composed_transforms_ts, seq_len=args.seq_len, return_size=True)
        segtrack_data = segtrack.SegTrack(dataset='train', data_dir='./Datasets/SegTrackv2', transform=composed_transforms_ts, seq_len=args.seq_len, return_size=True)
        train_data = ConcatDataset([davis_data,fbms_data,segtrack_data])


    #save_dir = args.save_dir + args.train_fold + '-' + args.model_name + '-' + args.train_dataset + '/saliency_map/'
    trainloader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=12)
    valloader = DataLoader(val_data, args.batch_size, shuffle=True, num_workers=12)
    num_iter_ts = len(trainloader)


    cnt = 0
    accmu_t = 0
    #gradient_accumulations = 16

    net.zero_grad()
    start_training = time.time()
    for epoch in range(epoch_start + 1, args.epochs):
        running_loss = 0.0
        running_val_loss = 0.0
        batch_loss = 0.0
        net.train()
        set_bn_eval(net)
        opt.zero_grad()
        start_epoch = time.time()
        for i, sample_batched in enumerate(trainloader):
            before_t = time.time()
            inputs, labels, label_name, size = sample_batched['images'], sample_batched['labels'], sample_batched[
                'label_name'], sample_batched['size']
            inputs.requires_grad_(True)
            labels.requires_grad_(False)
            inputs = inputs.to(device)
            labels = labels.to(device)

            #with autocast():
            #    prob_pred = net(inputs)
            prob_pred = net(inputs)
            accmu_t += (time.time() - before_t)
            cnt += 1

            #prob_pred = (prob_pred - torch.min(prob_pred) + 1e-3) / (torch.max(prob_pred) - torch.min(prob_pred) + 1e-3)
            shape = labels.shape[3:]
            pred_list = []
            for t in range(args.seq_len):
                up_prob_pred = F.upsample(prob_pred[:, t, :, :, :], size=shape, mode='bilinear', align_corners=True)
                pred_list.append(up_prob_pred)
            prob_pred = torch.stack(pred_list, dim=1)
            #with autocast():
            #    loss = loss_fn(prob_pred, labels) / gradient_accumulations
            loss = loss_fn(prob_pred, labels)
            #scaler.scale(loss).backward()
            loss.backward()
            batch_loss += loss.item()

            #if ((i + 1) % gradient_accumulations == 0) or (i +1 == len(trainloader)):
            running_loss += batch_loss
            batch_loss = 0.0
                #scaler.step(opt)
                #scaler.update()
            opt.step()
            opt.zero_grad()
            net.zero_grad()

        #train_loss = running_loss / (len(trainloader) / gradient_accumulations)
        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch + 1, args.epochs, train_loss)))
        end_epoch = time.time()
        print("Epoch training time: {}s".format(end_epoch - start_epoch))

        val_batch_loss = 0.0
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
                    up_prob_pred = F.interpolate(prob_pred[:, t, :, :, :], size=shape, mode='bilinear', align_corners=True)
                    pred_list.append(up_prob_pred)
                prob_pred = torch.stack(pred_list, dim=1)
                #loss = (loss_fn(prob_pred, labels) / gradient_accumulations).item()
                loss = loss_fn(prob_pred, labels).item()
                val_batch_loss += loss

                #if ((i + 1) % gradient_accumulations == 0) or (i + 1 == len(valloader)):
                running_val_loss += val_batch_loss
                val_batch_loss = 0.0

        #val_loss = running_val_loss / (len(valloader) / gradient_accumulations)
        val_loss = running_val_loss / len(valloader)
        val_losses.append(val_loss)
        print(print('Epoch [{}/{}], Val-loss: {:.7f}'.format(epoch + 1, args.epochs, val_loss)))


        if best_score is None:
            best_score = val_loss
            torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'val_losses': val_losses,
                'train_losses': train_losses,
                'best_score': best_score
            }, 'best_' + args.save_weights)
            print("Saved best Score")
        elif best_score >= val_loss:
            print("New best score")
            best_score = val_loss
            torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'val_losses': val_losses,
                'train_losses': train_losses,
                'best_score': best_score
            }, 'best_' + args.save_weights)
            #pat_count = 0
            print("Saved best Score")


        torch.save({
            'epoch': epoch,
            'net_state_dict': net.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'val_losses': val_losses,
            'train_losses': train_losses,
            'best_score': best_score
        }, args.save_weights)
        print('Checkpoint saved')

    end_training = time.time()
    print("Complete training time: {}s".format(end_training - start_training))


if __name__ == '__main__':
    args = get_arguments()
    main(args)