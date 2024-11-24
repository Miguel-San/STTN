import os
import cv2
import time
import math
import glob
from tqdm import tqdm
import shutil
import importlib
import datetime
import numpy as np
from PIL import Image
from math import log10

from joblib import Parallel, delayed

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import torch.distributed as dist

from core.dataset import Dataset
from core.loss import AdversarialLoss
from core.vg_regularizer import VGRegularizer


class Trainer():
    def __init__(self, config, debug=False):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        if debug:
            self.config['trainer']['save_freq'] = 5
            self.config['trainer']['valid_freq'] = 5
            self.config['trainer']['iterations'] = 5

        # setup data set and data loader
        self.train_dataset = Dataset(config['data_loader'], split='train',  debug=debug)
        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'], 
                rank=config['global_rank'])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None), 
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)
        
        # Test dataset
        #TODO: CAMBIAR ESTO: test -> test_5050
        self.test_dataset = Dataset(config['data_loader'], split='test_5050', debug=debug)
        self.control_imgs_loader = DataLoader(
            Subset(self.test_dataset, indices=[0]),
            batch_size=1,
            shuffle=False,
            num_workers=self.train_args['num_workers'],
            sampler=None
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=True,
            num_workers=self.train_args['num_workers'],
            sampler=None
        )
        self.test_iterations = self.train_args.get("test_iterations", None)

        # set loss functions 
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()

        vg_coords_raw = self.config["losses"]["vg_coords"]
        self.vg_coords = []
        for ymin, ymax in vg_coords_raw:
            self.vg_coords += list(range(ymin, ymax+1))

        print("****************** DEBUG ***********************")
        print(self.vg_coords)
        print("************************************************")

        # setup models including generator and discriminator
        net = importlib.import_module('model.'+config['model'])
        self.netG = net.InpaintGenerator()
        self.netG = self.netG.to(self.config['device'])
        self.netD = net.Discriminator(
            in_channels=1, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
        self.netD = self.netD.to(self.config['device'])
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.load()

        if config['distributed']:
            self.netG = DDP(
                self.netG, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
            self.netD = DDP(
                self.netD, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            # self.dis_writer = SummaryWriter(
            #     os.path.join(config['save_dir'], 'dis'))
            # self.gen_writer = SummaryWriter(
            #     os.path.join(config['save_dir'], 'gen'))

            sum_writer_path = os.path.join(config['save_dir']+"/tb_log")
            self.tb_writer = SummaryWriter(sum_writer_path)

    # get current learning rate
    def get_lr(self):
        return self.optimG.param_groups[0]['lr']

     # learning rate scheduler, step
    def adjust_learning_rate(self):
        decay = 0.1**(min(self.iteration,
                          self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr

    # add summary
    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name]/100, self.iteration)
            self.summary[name] = 0

    # load netG and netD
    def load(self):
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None
        if latest_epoch is not None:
            gen_path = os.path.join(
                model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_path = os.path.join(
                model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(
                model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_path))
            data = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(data['netG'])
            data = torch.load(dis_path, map_location=self.config['device'])
            self.netD.load_state_dict(data['netD'])
            data = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']
        else:
            if self.config['global_rank'] == 0:
                print(
                    'Warnning: There is no trained model found. An initialized model will be used.')

    # save parameters every eval_epoch
    def save(self, it):
        if self.config['global_rank'] == 0:
            gen_path = os.path.join(
                self.config['save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
            dis_path = os.path.join(
                self.config['save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
            opt_path = os.path.join(
                self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
            print('\nsaving model to {} ...'.format(gen_path))
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                netD = self.netD.module
            else:
                netG = self.netG
                netD = self.netD
            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['save_dir'], 'latest.ckpt')))

    # train entry
    def train(self):
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
        
        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            self._train_epoch(pbar)
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    # process input and calculate loss every training epoch
    def _train_epoch(self, pbar):
        device = self.config['device']

        while self.iteration <= self.train_args['iterations']:
            print("************ EPOCH {} ************".format(self.epoch))
            for frames, masks in self.train_loader:
                self.adjust_learning_rate()
                self.iteration += 1

                frames, masks = frames.to(device), masks.to(device)
                b, t, c, h, w = frames.size()
                masked_frame = (frames * (1 - masks).float())
                pred_img = self.netG(masked_frame, masks)
                frames = frames.view(b*t, c, h, w)
                masks = masks.view(b*t, 1, h, w)
                comp_img = frames*(1.-masks) + masks*pred_img

                gen_loss = 0
                dis_loss = 0

                # discriminator adversarial loss
                real_vid_feat = self.netD(frames)
                fake_vid_feat = self.netD(comp_img.detach())
                dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)
                dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)
                dis_loss += (dis_real_loss + dis_fake_loss) / 2
                # self.add_summary(
                #     self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())
                # self.add_summary(
                #     self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
                self.optimD.zero_grad()
                dis_loss.backward()
                self.optimD.step()

                # generator adversarial loss
                gen_vid_feat = self.netD(comp_img)
                gan_loss = self.adversarial_loss(gen_vid_feat, True, False)
                gan_loss = gan_loss * self.config['losses']['adversarial_weight']
                gen_loss += gan_loss
                # self.add_summary(
                #     self.gen_writer, 'loss/gan_loss', gan_loss.item())

                # generator l1 loss
                hole_loss = self.l1_loss(pred_img*masks, frames*masks)
                hole_loss = hole_loss / torch.mean(masks) * self.config['losses']['hole_weight']
                gen_loss += hole_loss 
                # self.add_summary(
                #     self.gen_writer, 'loss/hole_loss', hole_loss.item())

                valid_loss = self.l1_loss(pred_img*(1-masks), frames*(1-masks))
                valid_loss = valid_loss / torch.mean(1-masks) * self.config['losses']['valid_weight']
                gen_loss += valid_loss 
                # self.add_summary(
                #     self.gen_writer, 'loss/valid_loss', valid_loss.item())

                # Visibility Graph Regularizer
                if self.config['losses']['vg_weight'] > 0:
                    vg_count = 0
                    for i in range(pred_img.shape[0]):                
                        for k in range(len(self.vg_coords)):
                            coord = self.vg_coords[k]
                            pred_series = pred_img[i, :, coord, :]
                            orig_series = frames.view(b*t, c, h, w)[i, :, coord, :]

                            vg_pred = VGRegularizer()
                            vg_pred.build(pred_series.flatten())

                            vg_orig = VGRegularizer()
                            vg_orig.build(orig_series.flatten())

                            vg_loss = self.l1_loss(vg_pred.G, vg_orig.G) * self.config['losses']['vg_weight']
                            vg_count += 1
                    gen_loss += vg_loss/vg_count
                else:
                    vg_loss = torch.tensor([0]).to(device)
                
                self.optimG.zero_grad()
                gen_loss.backward()
                self.optimG.step()

                # console logs
                if self.config['global_rank'] == 0:
                    pbar.update(1)
                    pbar.set_description((
                        f"d: {dis_loss.item():.3f}; g: {gan_loss.item():.3f}; "
                        f"hole: {hole_loss.item():.3f}; valid: {valid_loss.item():.3f}; "
                        f"vg: {vg_loss.item():.3f}")
                    )

                # Tensorboard logs
                if self.iteration % self.train_args["log_freq"] == 0:
                    loss_dict = {
                        'dis_loss': dis_loss.item(),
                        'gan_loss': gan_loss.item(),
                        'hole_loss': hole_loss.item(),
                        'valid_loss': valid_loss.item(),
                        "gen_loss": gen_loss.item(),
                        "vg_loss": vg_loss.item()
                    }

                    self.netD.eval()
                    self.netG.eval()

                    with torch.no_grad():
                        _testing = True

                        dis_loss_batch = []
                        gan_loss_batch = []
                        hole_loss_batch = []
                        valid_loss_batch = []
                        gen_loss_batch = []
                        vg_loss_batch = []

                        while _testing:
                            testing_iter = 0
                            for frames, masks in self.test_loader:
                                testing_iter += 1

                                gen_loss = 0
                                dis_loss = 0

                                frames, masks = frames.to(device), masks.to(device)
                                b, t, c, h, w = frames.size()
                                masked_frame = (frames * (1 - masks).float())
                                pred_img = self.netG(masked_frame, masks)
                                frames = frames.view(b*t, c, h, w)
                                masks = masks.view(b*t, 1, h, w)
                                comp_img = frames*(1.-masks) + masks*pred_img

                                real_vid_feat = self.netD(frames)
                                fake_vid_feat = self.netD(comp_img.detach())
                                dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)
                                dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)
                                dis_loss += (dis_real_loss + dis_fake_loss) / 2

                                gen_vid_feat = self.netD(comp_img)
                                gan_loss = self.adversarial_loss(gen_vid_feat, True, False)
                                gan_loss = gan_loss * self.config['losses']['adversarial_weight']
                                gen_loss += gan_loss

                                hole_loss = self.l1_loss(pred_img*masks, frames*masks)
                                hole_loss = hole_loss / torch.mean(masks) * self.config['losses']['hole_weight']
                                gen_loss += hole_loss

                                valid_loss = self.l1_loss(pred_img*(1-masks), frames*(1-masks))
                                valid_loss = valid_loss / torch.mean(1-masks) * self.config['losses']['valid_weight']
                                gen_loss += valid_loss 

                                # Visibility Graph Regularizer
                                if self.config['losses']['vg_weight'] > 0:
                                    vg_count = 0
                                    for i in range(pred_img.shape[0]):                
                                        for k in range(len(self.vg_coords)):
                                            coord = self.vg_coords[k]
                                            pred_series = pred_img[i, :, coord, :]
                                            orig_series = frames.view(b*t, c, h, w)[i, :, coord, :]

                                            vg_pred = VGRegularizer()
                                            vg_pred.build(pred_series.flatten())

                                            vg_orig = VGRegularizer()
                                            vg_orig.build(orig_series.flatten())

                                            vg_loss = self.l1_loss(vg_pred.G, vg_orig.G) * self.config['losses']['vg_weight']
                                            vg_count += 1
                                    gen_loss += vg_loss/vg_count
                                else:
                                    vg_loss = torch.tensor([0]).to(device)

                                dis_loss_batch.append(dis_loss.item())
                                gan_loss_batch.append(gan_loss.item())
                                hole_loss_batch.append(hole_loss.item())
                                valid_loss_batch.append(valid_loss.item())
                                gen_loss_batch.append(gen_loss.item())
                                vg_loss_batch.append(vg_loss.item())

                                if self.test_iterations is not None:
                                    if testing_iter > self.test_iterations:
                                        _testing = False
                                        break

                            _testing = False

                    loss_dict_val = {
                        'dis_loss': np.mean(dis_loss_batch),
                        'gan_loss': np.mean(gan_loss_batch),
                        'hole_loss': np.mean(hole_loss_batch),
                        'valid_loss': np.mean(valid_loss_batch),
                        "gen_loss": np.mean(gen_loss_batch),
                        "vg_loss": np.mean(vg_loss_batch)
                    }

                    for k in loss_dict.keys():
                        v_tr = loss_dict[k]
                        v_val = loss_dict_val[k]

                        self.tb_writer.add_scalars("Loss/"+k, {
                            "Training" : v_tr,
                            "Validation" : v_val
                        }, self.iteration)

                # Control images            
                if self.iteration % self.train_args['valid_freq'] == 0:
                    win_len = self.config['data_loader']['sample_length']
                    self.netG.eval()
                    with torch.no_grad():
                        for i, (frames, masks) in enumerate(self.control_imgs_loader):
                            video_name = self.test_dataset.video_names[0]
                            orig_frames = []
                            pred_frames = []
                            comp_frames = []

                            frames, masks = frames.to(device), masks.to(device)
                            b, t, c, h, w = frames.size()
                            masked_frame = (frames * (1 - masks).float())
                            pred_img = self.netG(masked_frame, masks)
                            frames = frames.view(b*t, c, h, w)
                            masks = masks.view(b*t, 1, h, w)
                            comp_img = frames*(1.-masks) + masks*pred_img
                            comp_img = comp_img.view(b, t, c, h, w)
                            comp_img = (comp_img + 1) / 2

                            orig_frames.append(((frames.view(b,t,c,h,w)[0,...]+1)/2).cpu().numpy().astype(np.float32))
                            pred_frames.append(((pred_img.view(b,t,c,h,w)[0,...]+1)/2).cpu().numpy().astype(np.float32))
                            comp_frames.append(comp_img.view(b,t,c,h,w)[0,...].cpu().numpy().astype(np.float32))

                            # for f in range(video_len-10-win_len, video_len-10):
                            # for f in range(win_len//2+1, win_len//2 + 1 + win_len):
                            #     frames_win = frames[:, f:f+win_len]
                            #     masks_win = masks[:, f:f+win_len]
                            #     frames_win, masks_win = frames_win.to(device), masks_win.to(device)
                            #     b, t, c, h, w = frames_win.size()
                            #     masked_frame = (frames_win * (1 - masks_win).float())
                            #     pred_img = self.netG(masked_frame, masks_win)
                            #     frames_win = frames_win.view(b*t, c, h, w)
                            #     masks_win = masks_win.view(b*t, 1, h, w)
                            #     comp_img = frames_win*(1.-masks_win) + masks_win*pred_img
                            #     comp_img = comp_img.view(b, t, c, h, w)
                            #     comp_img = (comp_img + 1) / 2
                                
                            #     orig_frames.append(((frames_win.view(b,t,c,h,w)[0,win_len//2+1,...]+1)/2).cpu().numpy())
                            #     pred_frames.append(((pred_img.view(b,t,c,h,w)[0,win_len//2+1,...]+1)/2).cpu().numpy())
                            #     comp_frames.append(comp_img.view(b,t,c,h,w)[0,win_len//2+1,...].cpu().numpy())

                            grid = make_grid(torch.cat([
                                torch.tensor(np.array(orig_frames)[0]), 
                                torch.tensor(np.array(pred_frames)[0]), 
                                torch.tensor(np.array(comp_frames)[0])], dim=0), nrow=win_len)
                            save_dir = os.path.join(self.config['save_dir'], 'control_images')
                            os.makedirs(save_dir, exist_ok=True)
                            save_image(grid, os.path.join(save_dir, f'{video_name}_{self.iteration}.png'))
                    self.netG.train()
                            

                # saving models
                if self.iteration % self.train_args['save_freq'] == 0:
                    self.save(int(self.iteration))
                if self.iteration > self.train_args['iterations']:
                    break

