--
--  AddRes.lua
--
--  Copyright (c) 2017, Mingyuan Luo
--

local AddRes, parent = torch.class('nn.AddRes', 'nn.Module')

function AddRes:__init()
    parent.__init(self)
    self.gradInput = {}
    self.output = {}
end

function AddRes:updateOutput(input)
  self.output[1] = torch.cat(input[1], input[2], 2)
  self.output[2] = input[2]:clone()
  for i = 1, (#input[2])[2], (#input[1])[2] do
    self.output[2]:add(self.output[1]:narrow(2, i, (#input[2])[2]))
  end
  return self.output
end

function AddRes:updateGradInput(input, gradOutput)
  for i = 1, #input do
    self.gradInput[i] = self.gradInput[i] or input[i].new()
    self.gradInput[i]:resizeAs(input[i])
  end
  self.gradInput[1]:copy(gradOutput[1]:narrow(2, 1, (#input[1])[2]))
  self.gradInput[2]:copy(gradOutput[1]:narrow(2, 1 + (#input[1])[2], (#input[2])[2]))
  self.gradInput[2]:add(gradOutput[2])
  for i = 1, (#input[2])[2], (#input[1])[2] do
    self.gradInput[1]:narrow(2, i, (#input[2])[2]):add(gradOutput[2])
  end
  return self.gradInput
end