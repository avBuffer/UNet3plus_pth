import argparse
import logging
import os
import sys
import math
import numpy as np
import matplotlib.pylot as plt
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from torch import optim
from tqdm import tqdm
from utils.eval import eval_net
from unet import UNet
from unet import UNet2Plus
from unet import UNet3Plus, UNet3Plus_DeepSup, UNet3Plus_DeepSup_CGM
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = 'data/train/imgs/'
dir_mask = 'data/train/masks/'
dir_checkpoint = 'ckpts/'


def train_net(unet_type, net, device, epochs=5, batch_size=1, lr=0.1, val_percent=0.1, save_cp=True, img_scale=0.5):
    dataset = BasicDataset(unet_type, dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
                     UNet type:       {unet_type}
                     Epochs:          {epochs}
                     Batch size:      {batch_size}
                     Learning rate:   {lr}
                     Dataset size:    {len(dataset)}
                     Training size:   {n_train}
                     Validation size: {n_val}
                     Checkpoints:     {save_cp}
                     Device:          {device.type}
                     Images scaling:  {img_scale}''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05 #cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = global_step

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    lrs = []
    best_loss = 10000
    for epoch in range(epochs):
        cur_lr = optimizer.param_groups[0]['lr']
        print('\nEpoch=', (epoch + 1), ' lr=', cur_lr)
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                
                assert imgs.shape[1] == net.n_channels, f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                dataset_len = len(dataset);
                
                a1 = dataset_len // 10
                a2 = dataset_len / 10
                b1 = global_step % a1
                b2 = global_step % a2

                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device, n_val)
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        # update scheduler
        scheduler.step()
        lrs.append(cur_lr)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            
            if loss.item() < best_loss:
                torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}_loss_{str(loss.item())}.pth')
                best_loss = loss.item()
                logging.info(f'Checkpoint {epoch + 1} saved ! loss (batch) = ' + str(loss.item()))
    
    # plot lr scheduler
    plt.plot(lrs , '.-', label='LambdaLR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.savefig('LR.png', dpi=300)

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu_id', dest='gpu_id', metavar='G', type=int, default=0, help='Number of gpu')
    parser.add_argument('-u', '--unet_type', dest='unet_type', metavar='U', type=str, default='v3', help='UNet type is v1/v2/v3 (unet unet++ unet3+)')
    
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2, help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1, help='Learning rate', dest='lr')
    
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    gpu_id = args.gpu_id 
    unet_type = args.unet_type 

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    if unet_type == 'v2':
        net = UNet2Plus(n_channels=3, n_classes=1)
    elif unet_type == 'v3':
        net = UNet3Plus(n_channels=3, n_classes=1)
        #net = UNet3Plus_DeepSup(n_channels=3, n_classes=1)
        #net = UNet3Plus_DeepSup_CGM(n_channels=3, n_classes=1)
    else:
        net = UNet(n_channels=3, n_classes=1)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{'Bilinear' if net.bilinear else 'Dilated conv'} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    try:
        train_net(unet_type=unet_type, net=net, epochs=args.epochs, batch_size=args.batchsize, 
                  lr=args.lr, device=device, img_scale=args.scale, val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os.exit(0)
