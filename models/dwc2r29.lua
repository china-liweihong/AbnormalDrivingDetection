--
--  dwc2r29.lua
--
--  Copyright (c) 2017, Mingyuan Luo
--

require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require '../layers/AddRes'

torch.setdefaulttensortype('torch.FloatTensor')

local stages = {2, 2, 2, 2}
local reduction = 0.8
local channels = { 64, 64, 64, 56 }

function addLayer(model, nChannels, layerType, inChannels, outChannels, nOutChannels)
  local stride = 1
  local tran = nn.Identity()
  local oChannels = nChannels
  if layerType == 'down' then
    stride = 2
    oChannels = nOutChannels
    tran = nn.Sequential()
    tran:add(cudnn.SpatialBatchNormalization(inChannels))
    tran:add(cudnn.ReLU(true))
    tran:add(cudnn.SpatialConvolution(inChannels, outChannels, 1, 1, stride, stride, 0, 0, 1))
    params = params + inChannels * outChannels
  end
  
  local net = nn.Sequential()
  net:add(cudnn.SpatialBatchNormalization(nChannels))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(nChannels, 4 * nChannels, 1, 1, 1, 1, 0, 0, 1))
  net:add(cudnn.SpatialBatchNormalization(4 * nChannels))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(4 * nChannels, 4 * nChannels, 3, 3, stride, stride, 1, 1, 2))
  net:add(cudnn.SpatialBatchNormalization(4 * nChannels))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(4 * nChannels, oChannels, 1, 1, 1, 1, 0, 0, 1))
  params = params + nChannels * 4 * nChannels + 4 * nChannels * 4 * nChannels * 9 / 2 + 4 * nChannels * oChannels
  
  return model
    :add(nn.ParallelTable()
      :add(tran)
      :add(net))
    :add(nn.AddRes())
end

function addDenseBlock(model, nChannels, N)
  for i = 1, N do 
    addLayer(model, nChannels)
  end
end

params = 0

model = nn.Sequential()
model:add(cudnn.SpatialConvolution(3, channels[1], 7, 7, 2, 2, 3, 3))
params = params + 3 * channels[1] * 7 * 7
model:add(cudnn.SpatialBatchNormalization(channels[1]))
model:add(cudnn.ReLU(true))
model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))

model:add(nn.ConcatTable()
  :add(nn.Identity())
  :add(nn.Identity()))

chanCount = channels[1]
for i = 1, 3 do
  addDenseBlock(model, channels[i], stages[i])
  chanCount = chanCount + stages[i] * channels[i]
  addLayer(model, channels[i], 'down', chanCount, math.floor(chanCount / channels[1 + i] * reduction) * channels[1 + i], channels[1 + i])
  chanCount = math.floor(chanCount / channels[1 + i] * reduction) * channels[1 + i] + channels[1 + i]
end
addDenseBlock(model, channels[4], stages[4])
chanCount = chanCount + stages[4] * channels[4]
model:add(nn.JoinTable(2))
chanCount = chanCount + channels[4]
model:add(cudnn.SpatialBatchNormalization(chanCount))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(chanCount, 230, 1, 1, 1, 1, 0, 0))
model:add(cudnn.SpatialAveragePooling(2, 2))

model:add(nn.Reshape(230*7*10))
model:add(nn.Linear(230*7*10, 128))
model:add(cudnn.ReLU())
model:add(nn.Linear(128, 128))
model:add(cudnn.ReLU())
model:add(nn.Linear(128, 10))
model:add(nn.LogSoftMax())

cudnn.convert(model, cudnn)
model = model:cuda()

model = require('../weight-init')(model, 'xavier')

--[[
print(model)
print(params)
input = torch.randn(1, 3, 480, 640):cuda()
output = model:forward(input)
print(#output)
]]

return model:clearState()
