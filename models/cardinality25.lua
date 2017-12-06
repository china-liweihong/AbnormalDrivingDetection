--
--  cardinality25.lua
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
model:add(cudnn.SpatialConvolution(10, 64, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(64))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

model:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(64))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(64))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(64))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(64))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 2))
model:add(cudnn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0))

model:add(nn.Reshape(256*7*9))
model:add(nn.Linear(256*7*9, 136))
model:add(cudnn.ReLU())
model:add(nn.Linear(136, 136))
model:add(cudnn.ReLU())
model:add(nn.Linear(136, 10))
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
