--
--  densenet88.lua
--
--  Copyright (c) 2017, Mingyuan Luo
--

require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')

local growthRate = 32
local stages = {4, 10, 16, 10}
local reduction = 0.5
local nChannels = 2 * growthRate

function addLayer(model, nChannels)
  local net = nn.Sequential()
  net:add(cudnn.SpatialBatchNormalization(nChannels))
  net:add(cudnn.ReLU())   
  net:add(cudnn.SpatialConvolution(nChannels, 4 * growthRate, 1, 1, 1, 1, 0, 0))
  net:add(cudnn.SpatialBatchNormalization(4 * growthRate))
  net:add(cudnn.SpatialConvolution(4 * growthRate, growthRate, 3, 3, 1, 1, 1, 1))
  params = params + nChannels * 4 * growthRate + 4 * growthRate * growthRate * 9
  
  return model
    :add(nn.Concat(2)
      :add(nn.Identity())
      :add(net))  
end

function addTransition(model, nChannels, nOutChannels)
  model:add(cudnn.SpatialBatchNormalization(nChannels))
  model:add(cudnn.ReLU())      
  model:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
  params = params + nChannels * nOutChannels
  model:add(cudnn.SpatialAveragePooling(2, 2)) 
end

function addDenseBlock(model, nChannels, N)
  for i = 1, N do 
    addLayer(model, nChannels)
    nChannels = nChannels + growthRate
  end
  return nChannels
end

params = 0

model = nn.Sequential()
model:add(cudnn.SpatialConvolution(3, nChannels, 7,7, 2,2, 3,3))
params = params + 3 * nChannels * 7 * 7
model:add(cudnn.SpatialBatchNormalization(nChannels))
model:add(cudnn.ReLU())
model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))

nChannels = addDenseBlock(model, nChannels, stages[1])
addTransition(model, nChannels, math.floor(nChannels*reduction))
nChannels = math.floor(nChannels*reduction)

nChannels = addDenseBlock(model, nChannels, stages[2])
addTransition(model, nChannels, math.floor(nChannels*reduction))
nChannels = math.floor(nChannels*reduction)

nChannels = addDenseBlock(model, nChannels, stages[3])
addTransition(model, nChannels, math.floor(nChannels*reduction))
nChannels = math.floor(nChannels*reduction)

nChannels = addDenseBlock(model, nChannels, stages[4])
addTransition(model, nChannels, 230)

model:add(nn.Reshape(230*7*10))
model:add(nn.Linear(230*7*10, 112))
model:add(cudnn.ReLU())
model:add(nn.Linear(112, 112))
model:add(cudnn.ReLU())
model:add(nn.Linear(112, 10))
model:add(nn.LogSoftMax())


cudnn.convert(model, cudnn)
model = model:cuda()

model = require('../weight-init')(model, 'xavier')

--[[
input = torch.randn(1, 3, 480, 640):cuda()
output = model:forward(input)
print(#output)
]]

return model:clearState()
