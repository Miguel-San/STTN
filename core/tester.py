import os
import importlib

from tqdm import tqdm

import numpy as np

import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader

from core.dataset import Dataset, TestDataset
from joblib import Parallel, delayed

class Tester():
    def __init__(self, config):
        self.config = config
        self.test_args = config["tester"]

        self.dataset = TestDataset(config["data_loader"], split=self.config["ds_name"], debug=False)
        self.test_loader = DataLoader(
            self.dataset,
            batch_size = 1,
            shuffle=False,
            num_workers=self.test_args["num_workers"],
            sampler=None,
        )

        net = importlib.import_module('model.' + config["model"])
        self.netG = net.InpaintGenerator().to(config["device"])

        state_dict = torch.load(config["save_dir"]+"/gen_"+str(config["ckpt"]).zfill(5)+".pth", map_location=config["device"])
        self.netG.load_state_dict(state_dict["netG"])
        print("loading model from: {}".format(config["ckpt"]))
        self.netG.eval()

    def test(self):
        win_len = self.config["data_loader"]["sample_length"]
        comp_frames = {v_name:[] for v_name in self.dataset.video_names}
        video_len = {}

        print("Predicting...")
        self.netG.eval()
        for i, (frames, masks) in enumerate(self.test_loader):
            video_name = self.dataset.video_names[i]
            video_len[video_name] = frames.shape[1]
            for f in tqdm(range(video_len[video_name]-win_len), desc=video_name):  # Debería ser desde win_len//2+1 hasta video_len[video_name]-win_len//2         
                frames_win = frames[:, f:f+win_len] # Cambiar a f-win_len//2:f+win_len//2+1
                masks_win = masks[:, f:f+win_len]

                with torch.no_grad():
                    frames_win, masks_win = frames_win.to(self.config["device"]), masks_win.to(self.config["device"])
                    b, t, c, h, w = frames_win.size()
                    masked_frame = (frames_win * (1 - masks_win).float())
                    pred_img = self.netG(masked_frame, masks_win)
                    frames_win = frames_win.view(b*t, c, h, w)
                    masks_win = masks_win.view(b*t, 1, h, w)
                    comp_img = frames_win*(1.-masks_win) + masks_win*pred_img
                    comp_img = comp_img.view(b, t, c, h, w)

                    comp_img = (comp_img + 1) / 2

                    comp_frames[video_name].append(comp_img.cpu().numpy())
                    # comp_frames[video_name].append(pred_img.view(b,t,c,h,w).cpu().numpy())

        print("Promediating each frame for every window...")
        # Promediate each frame with corresponding generated frames in each window
        comp_mean_frames = {}
        for video_name, frames in comp_frames.items():
            print(len(comp_frames[video_name]), comp_frames[video_name][0].shape)

            comp_mean_frames[video_name] = np.empty((video_len[video_name], *comp_frames[video_name][0].shape[-int(win_len/2):])) # <--- Qué es esto?
            for i in tqdm(range(video_len[video_name]), desc=video_name): # Separar en tres loops: 0:win_len//2, win_len//2:video_len-win_len//2, video_len-win_len//2:video_len
                win_indices, frame_indices = TestDataset.get_win_idx_from_frame(i, win_len, video_len[video_name])
                frame_stack = np.empty((len(win_indices), *comp_frames[video_name][0].shape[-int(win_len/2):]))
                for j in range(len(win_indices)):
                    frame_stack[j] = comp_frames[video_name][win_indices[j]][0,frame_indices[j],...]
                comp_mean_frames[video_name][i] = np.mean(frame_stack, axis=0)

        print("Saving...")
        def save_frame(video_name, frame, i, config):
            img_path = f"{config['results_dir']}/{config['data_loader']['name']}/{video_name}/frame_{i}.png"
            frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255.0
            frame = frame.astype(np.uint8)
            img = Image.fromarray(np.squeeze(frame), mode="L")
            img.save(img_path)

        # Save the completed frames as separate images
        for video_name, frames in comp_mean_frames.items():
            os.makedirs(self.config["results_dir"]+"/"+self.config["data_loader"]["name"]+"/"+video_name, exist_ok=True)
            for i, frame in tqdm(enumerate(frames), desc=video_name, total=len(frames)):
                img_path = self.config["results_dir"]+"/"+self.config["data_loader"]["name"]+"/"+video_name+"/frame_"+str(i)+".png"

                frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255.0
                frame = frame.astype(np.uint8)

                img = Image.fromarray(np.squeeze(frame), mode="L")
                img.save(img_path)            

            # Parallel(n_jobs=-1, verbose=1)(
            #     delayed(save_frame)(video_name, frame, i, self.config) for i, frame in enumerate(frames)
            # )
