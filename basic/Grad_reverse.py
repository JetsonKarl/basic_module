import torch

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

def printgradnorm(self, grad_input, grad_output):
    #print('Inside ' + self.__class__.__name__ + ' backward')
    #print('Inside class:' + self.__class__.__name__)
    #print('')
    #print('grad_input: ', type(grad_input))
    #print('grad_input[0]: ', type(grad_input[0]))
    #print('grad_output: ', type(grad_output))
    #print('grad_output[0]: ', type(grad_output[0]))
    #print('')
    #print('grad_input size:', grad_input[0].size())
    #print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].data.norm())




class FCDiscriminator_img(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator_img, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
		self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)

		return x