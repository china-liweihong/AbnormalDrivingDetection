--
--  wide12.lua
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
model:add(cudnn.SpatialConvolution(3, 32, 11, 11, 2, 2, 0, 0, 1))
model:add(cudnn.SpatialBatchNormalization(32))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(32, 128, 3, 3, 1, 1, 1, 1, 1))
model:add(cudnn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1))
model:add(cudnn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1))
model:add(cudnn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

model:add(cudnn.SpatialConvolution(256, 512, 3, 3, 2, 2, 1, 1, 1))
model:add(cudnn.SpatialBatchNormalization(512))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, 1, 1, 1))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

model:add(nn.Reshape(256*7*10))
model:add(nn.Linear(256*7*10, 112))
model:add(cudnn.ReLU())
model:add(nn.Linear(112, 112))
model:add(cudnn.ReLU())
model:add(nn.Linear(112, 10))
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
