import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

    def bis(self, input, dim, index):       #last dimension of input is the number of 1D feature vectors
        # batch index select
        # input: [N, ?, ?, ...], N is batchsize
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)  #gather doesn't change the dimension of input, https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        ### search
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)    #https://blog.csdn.net/qq_34914551/article/details/102940368
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)   #default stride is one
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]
        #print(refsr_lv3_unfold.shape)
        #print(lrsr_lv3_unfold.shape)

        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N, Hr*Wr, H*W], Performs a batch matrix-matrix product of matrices on two 3D vectors
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1) #[N, H*W]

        ### transfer
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)           #diffferent levels of reference feature
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)
        #Lv3: torch.Size([18, 2304, 1600])       LV2: torch.Size([18, 4608, 1600])       LV1: torch.Size([18, 9216, 1600]) (shape)

        #print("Lv3: {}\tLV2: {}\tLV1: {}\n".format(ref_lv3_unfold.shape, ref_lv2_unfold.shape, ref_lv1_unfold.shape))

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)  #smallest feature map, deepest
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)  #largest feature map, most shallow

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)
        #Lv3: torch.Size([18, 256, 40, 40])      LV2: torch.Size([18, 128, 80, 80])      LV1: torch.Size([18, 64, 160, 160])

        S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

        return S, T_lv3, T_lv2, T_lv1