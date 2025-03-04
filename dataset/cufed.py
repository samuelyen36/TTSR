import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.fliplr(sample['LR']).copy()
            sample['HR'] = np.fliplr(sample['HR']).copy()
            sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
            sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        LR, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
        LR = LR.transpose((2,0,1))
        LR_sr = LR_sr.transpose((2,0,1))
        HR = HR.transpose((2,0,1))
        Ref = Ref.transpose((2,0,1))
        Ref_sr = Ref_sr.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float(),
                'input_filename': sample['input_filename'],
                'ref_filename': sample['ref_filename']}

class ToTensor_multi(object):
    def __call__(self, sample):
        LR, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
        LR = LR.transpose((2,0,1))
        LR_sr = LR_sr.transpose((2,0,1))
        HR = HR.transpose((2,0,1))
        for i in range(0, len(Ref)):
            Ref[i] = Ref[i].transpose((2,0,1))
            Ref_sr[i] = Ref_sr[i].transpose((2,0,1))
            Ref[i] = torch.from_numpy(Ref[i]).float()
            Ref_sr[i] = torch.from_numpy(Ref_sr[i]).float()
        #Ref = Ref.transpose((2,0,1))
        #Ref_sr = Ref_sr.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': Ref,
                'Ref_sr': Ref_sr,
                'input_filename': sample['input_filename'],
                'ref_filename': sample['ref_filename']}



class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()]) ):
        self.input_list = sorted([os.path.join(args.dataset_dir, 'train/input', name) for name in 
            os.listdir( os.path.join(args.dataset_dir, 'train/input') )])
        self.ref_list = sorted([os.path.join(args.dataset_dir, 'train/ref', name) for name in 
            os.listdir( os.path.join(args.dataset_dir, 'train/ref') )])
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.input_list[idx])
        h,w = HR.shape[:2]
        #HR = HR[:h//4*4, :w//4*4, :]

        ### LR and LR_sr
        LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref_sub = imread(self.ref_list[idx])
        h2, w2 = Ref_sub.shape[:2]
        Ref_sr_sub = np.array(Image.fromarray(Ref_sub).resize((w2//4, h2//4), Image.BICUBIC))       #downscale ref image
        Ref_sr_sub = np.array(Image.fromarray(Ref_sr_sub).resize((w2, h2), Image.BICUBIC))          #upscale the former downscaled-image
    
        ### complete ref and ref_sr to the same size, to use batch_size > 1
        Ref = np.zeros((160, 160, 3))
        Ref_sr = np.zeros((160, 160, 3))
        Ref[:h2, :w2, :] = Ref_sub
        Ref_sr[:h2, :w2, :] = Ref_sr_sub

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.
        

        """
        ### rgb range to[0,1]
        LR = LR / 256.
        LR_sr = LR_sr / 256.
        HR = HR / 256.
        Ref = Ref / 256.
        Ref_sr = Ref_sr / 256.
        """

        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr,
                  'input_filename': self.input_list[idx],
                  'ref_filename': self.ref_list[idx]}

        if self.transform:
            sample = self.transform(sample)
        return sample


class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
        self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', '*_0.png')))
        self.ref_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', 
            '*_' + ref_level + '.png')))
        self.transform = transform
        self.randompick = args.randompick

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.input_list[idx])
        h, w = HR.shape[:2]
        h, w = h//4*4, w//4*4
        HR = HR[:h, :w, :] ### crop to the multiple of 4

        ### LR and LR_sr
        LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

        ref_filename=''

        ### Ref and Ref_sr
        if self.randompick == False:
            Ref = imread(self.ref_list[idx])
            ref_filename = self.ref_list[idx]
        if self.randompick == True:
            _len = len(self.input_list)
            _idx = random.randrange(0, _len)
            Ref = imread(self.ref_list[_idx])
            ref_filename = self.ref_list[_idx]
            #print("pick ref image rangomly: LR: {}, Ref:{}".format(self.input_list[idx],self.ref_list[_idx]))
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        Ref = Ref[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))


        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,                                             #also modify toTensor to add new keys
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr,
                  'input_filename': str(self.input_list[idx]),
                  'ref_filename': str(ref_filename)}

        if self.transform:
            sample = self.transform(sample)
        return sample

class TestSet_multiframe(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor_multi()])):
        self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', '*_0.png')))
        self.transform = transform
        self.multi_frame_count = args.eval_multiframe_count
        print("in dataset of dataloader, {} frames are used".format(self.multi_frame_count))

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.input_list[idx])
        h, w = HR.shape[:2]
        h, w = h//4*4, w//4*4
        HR = HR[:h, :w, :] ### crop to the multiple of 4

        ### LR and LR_sr
        LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

        ref_filename=''
        Ref_list=[]
        Ref_SR_list=[]
        ref_filename_list=[]
        ### Ref and Ref_sr
        for i in range(1,self.multi_frame_count+1):
            #Ref_list.append()
            #print(self.input_list[idx][:-5]+str(i)+".png")
            Ref = imread(self.input_list[idx][:-5]+str(i)+".png")
            ref_filename_list.append(self.input_list[idx][:-5]+str(i)+".png")

            h2, w2 = Ref.shape[:2]
            h2, w2 = h2//4*4, w2//4*4
            Ref = Ref[:h2, :w2, :]
            Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
            Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))
            Ref = Ref.astype(np.float32)
            Ref_sr = Ref_sr.astype(np.float32)
            Ref = Ref / 127.5 - 1.
            Ref_sr = Ref_sr / 127.5 - 1.
            Ref_list.append(Ref)
            Ref_SR_list.append(Ref_sr)

        #Ref = imread(self.ref_list[idx])
        #ref_filename = self.ref_list[idx]
        #print("pick ref image rangomly: LR: {}, Ref:{}".format(self.input_list[idx],self.ref_list[_idx]))


        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.

        sample = {'LR': LR,                                             #also modify toTensor to add new keys
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref_list, 
                  'Ref_sr': Ref_SR_list,
                  'input_filename': str(self.input_list[idx]),
                  'ref_filename': ref_filename_list}

        if self.transform:
            sample = self.transform(sample)
        return sample