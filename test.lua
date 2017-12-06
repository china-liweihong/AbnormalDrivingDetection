--
--  test.lua
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

local opt = opts.testParse(arg)
testName = opt.model
testIndex = opt.index
testEnd = testIndex + opt.more - 1

if not paths.filep('./datasets/split/c0_index.t7') then
  datasets.init()
end
if not paths.filep('./datasets/test_label.t7') then
  datasets.initTest()
end
testset = {
    batchSize = opt.batchSize,
    length = 11215,
    data = torch.load('./datasets/test_data_norm.t7'),
    label = torch.load('./datasets/test_label.t7')
}

function testset:size()
    return math.ceil(self.length / self.batchSize)
end

batchData = torch.Tensor(testset.batchSize, 3, 480, 640):cuda()
batchLabel = torch.Tensor(testset.batchSize):cuda()

getBatch = function(dataset, i)
    local t = batchIndex or 0
    local size = math.min(t + testset.batchSize, testset.length) - t
    if(size ~= batchData:size(1)) then
        batchData = torch.Tensor(size, 3, 480, 640):cuda();
        batchLabel = torch.Tensor(size):cuda();
    end
    for k = 1, size do
        batchData[{ k, {}, {}, {} }] = dataset.data[{ t + k, {}, {}, {} }]
        batchLabel[k] = dataset.label[t + k]
    end
    batchIndex = t + size
    return {batchData, batchLabel}
end

setmetatable(testset, 
    {__index = function(t, i) 
        return getBatch(t, i)
    end}
);

if not paths.dirp('./results/' .. testName .. '_test') then
  paths.mkdir('./results/' .. testName .. '_test')
end
print(testName .. ' testing start ...')
for testI = testIndex, testEnd do
    modelFile = './results/' .. testName .. '/' .. testName .. '_' .. testI .. '.t7'
    model = torch.load(modelFile)
    cudnn.convert(model, cudnn)
    model = model:cuda()
    model:evaluate()

    batchIndex = nil
    classCorrect = nil
    for i = 1, testset:size() do
        local data = testset[i]
        local prediction = model:forward(data[1])
        local _, indices = torch.sort(prediction, true)
        if classCorrect == nil then
            classCorrect = torch.Tensor((#indices)[2]):fill(0)
        end
        for j = 1, data[2]:size(1) do
            local index = -1
            for k = 1, (#indices)[2] do
                if data[2][j] == indices[j][k] then
                    index = k
                    break
                end
            end
            classCorrect[{ {index, (#indices)[2]} }]:add(1)
        end
    end

    io.output('./results/' .. testName .. '_test/' .. testName .. '_' .. testI .. '_test.log')
    for i = 1, (#classCorrect)[1] do
        print(i .. ' ' .. (classCorrect[i] / testset.length))
    end
end
