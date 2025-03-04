from cv2 import PSNR
from utils import calc_psnr_and_ssim
from model import Vgg19

import os
import numpy as np
from imageio import imread, imsave
from PIL import Image

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils


class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model                  #TTSR.TTSR(args).to(device)
        #self.rcan = make_cleaning_net().to("cuda")      #create RCAN
        #self.rcan.load_state_dict(torch.load("Gxy_112941.pth"))
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = nn.DataParallel(self.vgg19, list(range(self.args.num_gpu)))

        self.params = [
            {"params": filter(lambda p: p.requires_grad, self.model.MainNet.parameters() if 
             args.num_gpu==1 else self.model.module.MainNet.parameters()),
             "lr": args.lr_rate
            },
            {"params": filter(lambda p: p.requires_grad, self.model.LTE.parameters() if 
             args.num_gpu==1 else self.model.module.LTE.parameters()), 
             "lr": args.lr_rate_lte
            }
        ]
        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            #model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items()}
            model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            if type(sample_batched[key]) != list:
                sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0, is_init=False):
        self.model.train()
        if (not is_init):
            self.scheduler.step()
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer.param_groups[0]['lr']))

        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()

            sample_batched = self.prepare(sample_batched)
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            #with torch.no_grad():       #no need to update rcan
            #    lr_RCAN = self.rcan(sample_batched['LR_sr'].cuda())
            hr = sample_batched['HR']
            ref = sample_batched['Ref']
            ref_sr = sample_batched['Ref_sr']
            sr, S, T_lv3, T_lv2, T_lv1 = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)       #T are used to calculate TPL loss
            #sr, S, T_lv3, T_lv2, T_lv1 = self.model(lr=lr, lrsr=lr_RCAN, ref=ref, refsr=ref)       #use RCAN to refine LR and use original ref image

            ### calc loss
            is_print = ((i_batch + 1) % self.args.print_every == 0) ### flag of print

            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
            loss = rec_loss
            if (is_print):
                self.logger.info( ('init ' if is_init else '') + 'epoch: ' + str(current_epoch) + 
                    '\t batch: ' + str(i_batch+1) )
                self.logger.info( 'rec_loss: %.10f' %(rec_loss.item()) )

            if (not is_init):       #using not only reconstruction loss
                if ('per_loss' in self.loss_all):
                    sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                    with torch.no_grad():
                        hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                    per_loss = self.args.per_w * self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
                    loss += per_loss
                    if (is_print):
                        self.logger.info( 'per_loss: %.10f' %(per_loss.item()) )
                if ('tpl_loss' in self.loss_all):
                    sr_lv1, sr_lv2, sr_lv3 = self.model(sr=sr)
                    tpl_loss = self.args.tpl_w * self.loss_all['tpl_loss'](sr_lv3, sr_lv2, sr_lv1, 
                        S, T_lv3, T_lv2, T_lv1)
                    loss += tpl_loss
                    if (is_print):
                        self.logger.info( 'tpl_loss: %.10f' %(tpl_loss.item()) )
                if ('adv_loss' in self.loss_all):
                    adv_loss = self.args.adv_w * self.loss_all['adv_loss'](sr, hr)
                    loss += adv_loss
                    if (is_print):
                        self.logger.info( 'adv_loss: %.10f' %(adv_loss.item()) )

            loss.backward()
            self.optimizer.step()
            

        if ((not is_init) and current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip('/')+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)

    def see_sim_of_different_level(self, sim_dict):
        for _, sample_batched in enumerate(self.dataloader['test'][str(1)]):
            input_filename = sample_batched['input_filename']
            sim_cat = sim_dict[(str(input_filename)+"_"+str(1))]
            for i in range(2,6):    #2~5levels
                current_sim = sim_dict[(str(input_filename)+"_"+str(i))]
                sim_cat = torch.cat((sim_cat,current_sim), dim=0)     #concate along channel
            _val, _idx = torch.max(input=sim_cat, dim=0)
            stat = [0] * 5
            _idx_list = _idx.tolist()
            for i in range(len(_idx_list)):
                for j in range(len(_idx_list[i])):
                    stat[_idx_list[i][j]] = stat[_idx_list[i][j]]+1
            percent_list = [round(number/sum(stat),2) for number in stat]
            #print("similarity index: {}\t\tpercent: {}".format(stat, percent_list))
            print("similarity index: {:5}{:5}{:5}{:5}{:5}\tpercent: {:<6} {:<6} {:<6} {:<6} {:<6} ".format(*stat, *percent_list))
            #print("filename: {}\tsim_map_shape: {}\tmax_shape: {}\targmax_shape: {}".format(input_filename, sim_cat.shape, _val.shape, _idx.shape))


    def evaluate(self, current_epoch=0):
        PSNR_list=[]
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')

        if (self.args.dataset == 'CUFED'):
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0
            
                for i_batch, sample_batched in enumerate(self.dataloader['test'][str(self.args.eval_ref)]): #batch_size=1
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR']
                    lr_sr = sample_batched['LR_sr']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']
                    ref_sr = sample_batched['Ref_sr']
                    input_filename = sample_batched['input_filename']
                    ref_filename = sample_batched['ref_filename']

                    sr, _, _, _, _ = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
                    if (self.args.eval_save_results):
                        sr_save = (sr+1.) * 127.5
                        sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'.png'), sr_save)
                    
                    ### calculate psnr and ssim
                    _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())
                    PSNR_list.append((_psnr, input_filename[0], ref_filename[0]))
                    psnr += _psnr
                    ssim += _ssim
                
                """
                sim_map_dict = {}
                for i in range(1,6):
                    print("evaluating on ref level: {}".format(i))
                    for i_batch, sample_batched in enumerate(self.dataloader['test'][str(i)]): #batch_size=1
                        cnt += 1
                        sample_batched = self.prepare(sample_batched)
                        lr = sample_batched['LR']
                        lr_sr = sample_batched['LR_sr']
                        hr = sample_batched['HR']
                        ref = sample_batched['Ref']
                        ref_sr = sample_batched['Ref_sr']
                        input_filename = sample_batched['input_filename']
                        ref_filename = sample_batched['ref_filename']
                        #print("LRSR: {}\tREF: {}\tREFSR: {}".format(lr_sr.shape, ref.shape, ref_sr.shape))
                        sr, similarity_map, _, _, _ = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)      #return sr, S, T_lv3, T_lv2, T_lv1
                        if (self.args.eval_save_results):
                            sr_save = (sr+1.) * 127.5
                            sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                            imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'ref{}'.format(str(i))+'.png'), sr_save)
                        
                        ### calculate psnr and ssim
                        _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())
                        PSNR_list.append((_psnr, input_filename[0], ref_filename[0]))
                        psnr += _psnr
                        ssim += _ssim
                        sim_map_dict[(str(input_filename)+"_"+str(i))] = torch.squeeze(similarity_map,0)

                self.see_sim_of_different_level(sim_map_dict)
                """
                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                self.logger.info('Ref  PSNR (now): %.3f \t SSIM (now): %.4f' %(psnr_ave, ssim_ave))
                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)' 
                    %(self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))

        self.logger.info('Evaluation over.')
        PSNR_list = sorted(PSNR_list, key=lambda tup: tup[0])
        #for (_p, inname, refname) in PSNR_list:
            #print("psnr: {}, input_file: {}, ref img: {}".format(_p, inname, refname))

    def test(self):
        self.logger.info('Test process...')
        self.logger.info('lr path:     %s' %(self.args.lr_path))
        self.logger.info('ref path:    %s' %(self.args.ref_path))

        ### LR and LR_sr
        LR = imread(self.args.lr_path)
        h1, w1 = LR.shape[:2]
        LR_sr = np.array(Image.fromarray(LR).resize((w1*4, h1*4), Image.BICUBIC))
        
        ### Ref and Ref_sr
        Ref = imread(self.args.ref_path)
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        Ref = Ref[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        ### to tensor
        LR_t = torch.from_numpy(LR.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        LR_sr_t = torch.from_numpy(LR_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_t = torch.from_numpy(Ref.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_sr_t = torch.from_numpy(Ref_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            sr, _, _, _, _ = self.model(lr=LR_t, lrsr=LR_sr_t, ref=Ref_t, refsr=Ref_sr_t)
            sr_save = (sr+1.) * 127.5
            sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            save_path = os.path.join(self.args.save_dir, 'save_results', os.path.basename(self.args.lr_path))
            imsave(save_path, sr_save)
            self.logger.info('output path: %s' %(save_path))

        self.logger.info('Test over.')

    def evaluate_multiframe(self, current_epoch=0):
        PSNR_list=[]
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')

        if (self.args.dataset == 'CUFED'):
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0
            
                for i_batch, sample_batched in enumerate(self.dataloader['multi']): #batch_size=1
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR']
                    lr_sr = sample_batched['LR_sr']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']     #now, it's a list
                    ref_sr = sample_batched['Ref_sr']
                    input_filename = sample_batched['input_filename']
                    ref_filename = sample_batched['ref_filename']
                    

                    for i in range(0, len(ref)):
                        ref[i] = ref[i].to(self.device)
                        ref_sr[i] = ref_sr[i].to(self.device)
                    sr, _, _, _, _ = self.model.multiframe_test(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
                    
                    if (self.args.eval_save_results):
                        sr_save = (sr+1.) * 127.5
                        sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'.png'), sr_save)
                    
                    ### calculate psnr and ssim
                    _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())
                    PSNR_list.append((_psnr, input_filename[0], ref_filename[0]))
                    psnr += _psnr
                    ssim += _ssim
                
                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                self.logger.info('Ref  PSNR (now): %.3f \t SSIM (now): %.4f' %(psnr_ave, ssim_ave))
                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)' 
                    %(self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))

        self.logger.info('Evaluation over.')
        PSNR_list = sorted(PSNR_list, key=lambda tup: tup[0])
        #for (_p, inname, refname) in PSNR_list:
            #print("psnr: {}, input_file: {}, ref img: {}".format(_p, inname, refname))
