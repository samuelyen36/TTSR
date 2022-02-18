from model import MainNet, LTE, SearchTransfer

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTSR(nn.Module):
    def __init__(self, args):
        super(TTSR, self).__init__()
        self.args = args
        self.num_res_blocks = list( map(int, args.num_res_blocks.split('+')) )
        self.MainNet = MainNet.MainNet(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, 
            res_scale=args.res_scale)
        self.LTE      = LTE.LTE(requires_grad=True)
        self.LTE_copy = LTE.LTE(requires_grad=False) ### used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer.SearchTransfer()
        self.SearchTransfer_multiframe = SearchTransfer.SearchTransfer_multiframe()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        lrsr_lv1, lrsr_lv2, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)     #in past, only get feautres from level 3
        refsr_lv1, refsr_lv2, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)  #in past, only get features from level 3
        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)

        if self.args.featurelevel == 1:
            S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(self.maxpool(self.maxpool(lrsr_lv1)), self.maxpool(self.maxpool(refsr_lv1)), ref_lv1, ref_lv2, ref_lv3)
        elif self.args.featurelevel == 2:
            S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(self.maxpool(lrsr_lv2), self.maxpool(refsr_lv2), ref_lv1, ref_lv2, ref_lv3)
        else:       #feature level == 3
            S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)    #in past, only this
        #lv3: torch.Size([B, 256, 40, 40])      lv2: torch.Size([B, 128, 80, 80])      lv1: torch.Size([B, 64, 160, 160])
        #S: torch.Size([B, 1, 40, 40])

        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)

        return sr, S, T_lv3, T_lv2, T_lv1


    def multiframe_test(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None, sim_map=None, T_feat=None):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        refsr_lv3_list = []
        ref_lv1 = []
        ref_lv2 = []
        ref_lv3 = []

        lrsr_lv1, lrsr_lv2, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)     #in past, only get feautres from level 3
        for _frame in refsr:                #gather all frame from ref list(which is 5 in CUFED5)
            _, _, tmp = self.LTE((_frame.detach() + 1.) / 2.)
            refsr_lv3_list.append(tmp)

        for _frame in ref:
            lv1, lv2, lv3 = self.LTE((_frame.detach() + 1.) / 2.)
            ref_lv1.append(lv1)
            ref_lv2.append(lv2)
            ref_lv3.append(lv3)

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer_multiframe(lrsr_lv3, refsr_lv3_list, ref_lv1, ref_lv2, ref_lv3)    #in past, only this
        #lv3: torch.Size([B, 256, 40, 40])      lv2: torch.Size([B, 128, 80, 80])      lv1: torch.Size([B, 64, 160, 160])
        #S: torch.Size([B, 1, 40, 40])

        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)

        return sr, S, T_lv3, T_lv2, T_lv1
