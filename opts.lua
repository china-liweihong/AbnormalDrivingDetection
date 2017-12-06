--
--  opts.lua
--
--  Copyright (c) 2017, Mingyuan Luo
--

local M = { }

function M.trainParse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Abnormal Driving Detection Training Script')
  cmd:text('See https://github.com/lmy0217/AbnormalDrivingDetection/blob/master/TRAINING.md for examples')
  cmd:text()
  cmd:text('Options:')
  ------------ General options --------------------
  cmd:option('-model',     'vgg16',      'name of training model')
  ------------- Training options --------------------
  cmd:option('-nEpochs',      10,     'Number of total epochs to run')
  cmd:option('-batchSize',     4,    'mini-batch size (1 = pure stochastic)')
  ---------- Optimization options ----------------------
  cmd:option('-LR',          1e-4,  'initial learning rate')
  cmd:option('-weightDecay',    1e-7,  'weight decay')
  cmd:text()

  local opt = cmd:parse(arg or {})

  if not paths.filep('./models/' .. opt.model .. '.lua') then
    cmd:error('error: missing model file: `./models/' .. opt.model .. '.lua`')
  end

  if not paths.filep('./datasets/train_label.t7') and not paths.dirp('./datasets/train') then
    cmd:error('error: missing data directory')
  end

  return opt
end

function M.testParse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Abnormal Driving Detection Testing Script')
  cmd:text('See https://github.com/lmy0217/AbnormalDrivingDetection/blob/master/TESTING.md for examples')
  cmd:text()
  cmd:text('Options:')
  ------------ General options --------------------
  cmd:option('-model',     'vgg16',      'name of testing model')
  ------------- Testing options --------------------
  cmd:option('-index',      0,     'Index of epoch to test')
  cmd:option('-more',      1,     'Number of epochs to test')
  cmd:option('-batchSize',     12,    'mini-batch size (1 = pure stochastic)')
  cmd:text()

  local opt = cmd:parse(arg or {})

  for i = opt.index, opt.index + opt.more - 1 do
    if not paths.filep('./results/' .. opt.model .. '/' .. opt.model .. '_' .. i .. '.t7') then
     cmd:error('error: missing `./results/' .. opt.model .. '/' .. opt.model .. '_' .. i .. '.t7`')
    end
  end

  if not paths.filep('./datasets/test_label.t7') and not paths.dirp('./datasets/train') then
    cmd:error('error: missing data directory')
  end

  return opt
end

return M
