import torch
import torch.nn as nn

class AdaIN(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, y):		#reserve content of x, transfer to distribution of y
		eps = 0	    #eps is used to add on std to avoid divide by zero
		mean_x = torch.mean(x, dim=[1,2])
		mean_y = torch.mean(y, dim=[1,2])

		std_x = torch.std(x, dim=[1,2])
		std_y = torch.std(y, dim=[1,2])

		#mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
		#mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

		#std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
		#std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps

		std_x = std_x + eps
		std_y = std_y + eps

		#print("meanx: {} meany: {} \tstd_x: {} std_y: {}".format(mean_x, mean_y, std_x, std_y))
		#print("ori: {}".format(x))
		out = (x - mean_x)/ std_x * std_y + mean_y
		#print("after: {}".format(out))

		return out
