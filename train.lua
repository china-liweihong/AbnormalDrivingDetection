--
--  train.lua
--
--  Copyright (c) 2017, Mingyuan Luo
--

require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'optim'
require 'paths'
require 'layers/AddDense'
require 'layers/AddRes'
local datasets = require 'datasets'
local opts = require 'opts'

cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')

local opt = opts.trainParse(arg)
trainName = opt.model
modelFile = './results/' .. trainName .. '/' .. trainName .. '_0.t7'
if not paths.dirp('./results/' .. trainName) then
  paths.mkdir('./results/' .. trainName)
end
if not paths.filep(modelFile) then
  model = require('./models/' .. trainName)
  torch.save(modelFile, model)
end

if not paths.filep('./datasets/split/c0_index.t7') then
  datasets.init()
end
if not paths.filep('./datasets/train_label.t7') then
  datasets.initTrain()
end
trainset = {
    batchSize = opt.batchSize,
    length = 11209,
    data = torch.load('./datasets/train_data_norm.t7'),
    label = torch.load('./datasets/train_label.t7')
}

function trainset:size() 
    return math.ceil(self.length / self.batchSize)
end

shuffle = torch.randperm(trainset.length)
batchData = torch.Tensor(trainset.batchSize, 3, 480, 640):cuda()
batchLabel = torch.Tensor(trainset.batchSize):cuda()

getBatch = function(dataset, i)
    local t = batchIndex or 0
    local size = math.min(t + trainset.batchSize, trainset.length) - t
    if(size ~= batchData:size(1)) then
        batchData = torch.Tensor(size, 3, 480, 640):cuda();
        batchLabel = torch.Tensor(size):cuda();
    end
    for k = 1, size do
        batchData[{ k, {}, {}, {} }] = dataset.data[{ shuffle[t + k], {}, {}, {} }]
        batchLabel[k] = dataset.label[shuffle[t + k]]
    end
    batchIndex = t + size
    return {batchData, batchLabel}
end

setmetatable(trainset, 
    {__index = function(t, i) 
        return getBatch(t, i)
    end}
);

model = model or torch.load(modelFile)
cudnn.convert(model, cudnn)
model = model:cuda()
model:training()

criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

trainer = nn.StochasticGradient(model, criterion)
trainer.maxIteration = opt.nEpochs
trainer.learningRate = opt.LR
trainer.learningRateDecay = opt.weightDecay
trainer.hookIteration = function(self, iteration, currentError)
    batchIndex = 0
    torch.save('./results/' .. trainName .. '/' .. trainName .. '_' .. iteration .. '.t7', model:clearState())
end
print(trainName .. ' training start ...')
io.output('./results/' .. trainName .. '.log')
trainer:train(trainset)
