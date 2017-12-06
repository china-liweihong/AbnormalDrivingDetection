--
--  resnet16.lua
--
--  Copyright (c) 2017, Mingyuan Luo
--

require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')

model = nn.Sequential()
model:add(cudnn.SpatialConvolution(3, 10, 11, 11, 2, 2, 0, 0, 1))
model:add(cudnn.SpatialBatchNormalization(10))
model:add(cudnn.ReLU())

model:add(nn.ConcatTable()
  :add(nn.Sequential()
    :add(cudnn.SpatialConvolution(10, 64, 3, 3, 2, 2, 1, 1, 1))
    :add(cudnn.SpatialBatchNormalization(64))
    :add(cudnn.ReLU())
    :add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)))
  :add(nn.Sequential()
    :add(cudnn.SpatialConvolution(10, 64, 1, 1, 2, 2, 0, 0, 1))))
model:add(nn.CAddTable())
model:add(cudnn.SpatialBatchNormalization(64))
model:add(cudnn.ReLU())

model:add(nn.ConcatTable()
  :add(nn.Sequential()
    :add(cudnn.SpatialConvolution(64, 64, 3, 3, 2, 2, 1, 1, 1))
    :add(cudnn.SpatialBatchNormalization(64))
    :add(cudnn.ReLU())
    :add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)))
  :add(nn.Sequential()
    :add(cudnn.SpatialConvolution(64, 128, 1, 1, 2, 2, 0, 0, 1))))
model:add(nn.CAddTable())
model:add(cudnn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU())

model:add(nn.ConcatTable()
  :add(nn.Sequential()
    :add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.SpatialBatchNormalization(128))
    :add(cudnn.ReLU())
    :add(cudnn.SpatialConvolution(128, 128, 3, 3, 2, 2, 1, 1, 1)))
  :add(nn.Sequential()
    :add(cudnn.SpatialConvolution(128, 128, 1, 1, 2, 2, 0, 0, 1))))
model:add(nn.CAddTable())
model:add(cudnn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU())

model:add(nn.ConcatTable()
  :add(nn.Sequential()
    :add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.SpatialBatchNormalization(256))
    :add(cudnn.ReLU())
    :add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)))
  :add(nn.Sequential()
    :add(cudnn.SpatialConvolution(128, 256, 1, 1, 1, 1, 0, 0, 1))))
model:add(nn.CAddTable())
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())

model:add(nn.ConcatTable()
  :add(nn.Sequential()
    :add(cudnn.SpatialConvolution(256, 256, 3, 3, 2, 2, 1, 1, 1))
    :add(cudnn.SpatialBatchNormalization(256))
    :add(cudnn.ReLU())
    :add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)))
  :add(nn.Sequential()
    :add(cudnn.SpatialConvolution(256, 256, 1, 1, 2, 2, 0, 0, 1))))
model:add(nn.CAddTable())
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())

model:add(nn.ConcatTable()
  :add(nn.Sequential()
    :add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.SpatialBatchNormalization(256))
    :add(cudnn.ReLU())
    :add(cudnn.SpatialConvolution(256, 256, 3, 3, 2, 2, 1, 1, 1)))
  :add(nn.Sequential()
    :add(cudnn.SpatialConvolution(256, 256, 1, 1, 2, 2, 0, 0, 1))))
model:add(nn.CAddTable())
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())

model:add(nn.Reshape(256*8*10))
model:add(nn.Linear(256*8*10, 100))
model:add(cudnn.ReLU())
model:add(nn.Linear(100, 100))
model:add(cudnn.ReLU())
model:add(nn.Linear(100, 10))
model:add(nn.LogSoftMax())

cudnn.convert(model, cudnn)
model = model:cuda()

model = require('../weight-init')(model, 'xavier')

--[[
input = torch.randn(2, 3, 480, 640):cuda()
output = model:forward(input)
print(#output)
]]

return model:clearState()
