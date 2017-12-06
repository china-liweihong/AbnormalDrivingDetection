--
--  datasets.lua
--
--  Copyright (c) 2017, Mingyuan Luo
--

require 'paths'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

datasets = {}

local dataDir = './datasets/train/'
local dirImgCount = { 2489, 2267, 2317, 2346, 2326, 2312, 2325, 2002, 1911, 2129 }

local loadData = function(label)
  print('Load data c' .. label .. ' ...')
  
  local splitIndex = torch.randperm(dirImgCount[1 + label])
  local trainCount = torch.floor(dirImgCount[1 + label] / 2)
  local testCount = dirImgCount[1 + label] - trainCount
  local data = { torch.Tensor(trainCount, 3, 480, 640), torch.Tensor(testCount, 3, 480, 640) }
  
  local dir = dataDir .. 'c' .. label .. '/'
  local dataIndex = 0
  for index = 0, 102200 do
    local imgFile = dir .. 'img_' .. index .. '.jpg'
    local f = io.open(imgFile)
    if(f ~= nil) then
      io.close(f)
      dataIndex = 1 + dataIndex
      local arrayIndex = splitIndex[dataIndex]
      if arrayIndex <= trainCount then
        data[1][{ {arrayIndex}, {}, {}, {} }] = image.load(imgFile)
      else
        data[2][{ {arrayIndex - trainCount}, {}, {}, {} }] = image.load(imgFile)
      end
    end
  end
  if dataIndex ~= dirImgCount[1 + label] then
    print('label c' ..  label .. ' number has error!')
  end
  
  if not paths.dirp('./datasets/split') then
    paths.mkdir('./datasets/split')
  end
  
  torch.save('./datasets/split/c' .. label .. '_index.t7', splitIndex) 
  torch.save('./datasets/split/c' .. label .. '_train.t7', data[1])
  torch.save('./datasets/split/c' .. label .. '_test.t7', data[2])
end

local saveTrain = function()
  print('Gets train data ...')
  
  local trainData = { torch.Tensor(11209, 3, 480, 640), torch.Tensor(11209) }
  
  local trainIndex = 1
  for label = 0, 9 do
    local data = torch.load('./datasets/split/c' .. label .. '_train.t7')
    local size = (#data)[1]
    trainData[1][{ {trainIndex, trainIndex + size - 1}, {}, {}, {} }] = data
    trainData[2][{ {trainIndex, trainIndex + size - 1} }] = 1 + label
    trainIndex = trainIndex + size
  end
  if trainIndex ~= 11210 then
    print('train data size error!')
  end
  
  torch.save('./datasets/train_data.t7', trainData[1])
  torch.save('./datasets/train_label.t7', trainData[2])
end

local saveTest = function()
  print('Gets test data ...')
  
  local testData = { torch.Tensor(11215, 3, 480, 640), torch.Tensor(11215) }
  
  local testIndex = 1
  for label = 0, 9 do
    local data = torch.load('./datasets/split/c' .. label .. '_test.t7')
    local size = (#data)[1]
    testData[1][{ {testIndex, testIndex + size - 1}, {}, {}, {} }] = data
    testData[2][{ {testIndex, testIndex + size - 1} }] = 1 + label
    testIndex = testIndex + size
  end
  if testIndex ~= 11216 then
    print('test data size error!')
  end
  
  torch.save('./datasets/test_data.t7', testData[1])
  torch.save('./datasets/test_label.t7', testData[2])

  print('Save test data completed.')
end

local normTrain = function()
  print('Norm train data ...')
  
  local data = torch.load('./datasets/train_data.t7')
  local ms = { torch.Tensor(3), torch.Tensor(3) }
  ms[1][1] = data[{ {}, 1, {}, {} }]:mean()
  ms[1][2] = data[{ {}, 2, {}, {} }]:mean()
  ms[1][3] = data[{ {}, 3, {}, {} }]:mean()
  ms[2][1] = data[{ {}, 1, {}, {} }]:std()
  ms[2][2] = data[{ {}, 2, {}, {} }]:std()
  ms[2][3] = data[{ {}, 3, {}, {} }]:std()
  
  data[{ {}, 1, {}, {} }] = data[{ {}, 1, {}, {} }]:csub(ms[1][1]) / ms[2][1]
  data[{ {}, 2, {}, {} }] = data[{ {}, 2, {}, {} }]:csub(ms[1][2]) / ms[2][2]
  data[{ {}, 3, {}, {} }] = data[{ {}, 3, {}, {} }]:csub(ms[1][3]) / ms[2][3]
  
  torch.save('./datasets/ms.t7', ms)
  torch.save('./datasets/train_data_norm.t7', data)
  
  print('Norm train data completed.')
end

local normTest = function()
  print('Norm test data ...')
  
  local data = torch.load('./datasets/test_data.t7')
  local ms = torch.load('./datasets/ms.t7')
  
  data[{ {}, 1, {}, {} }] = data[{ {}, 1, {}, {} }]:csub(ms[1][1]) / ms[2][1]
  data[{ {}, 2, {}, {} }] = data[{ {}, 2, {}, {} }]:csub(ms[1][2]) / ms[2][2]
  data[{ {}, 3, {}, {} }] = data[{ {}, 3, {}, {} }]:csub(ms[1][3]) / ms[2][3]
  
  torch.save('./datasets/test_data_norm.t7', data)
  
  print('Norm test data completed.')
end

function datasets.init()
  for i = 0, 9 do
    loadData(i)
  end
end

function datasets.initTrain()
  saveTrain()
  normTrain()
end

function datasets.initTest()
  saveTest()
  normTest()
end

return datasets
