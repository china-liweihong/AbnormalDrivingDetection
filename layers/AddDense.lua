--
--  AddDense.lua
--
--  Copyright (c) 2017, Mingyuan Luo
--

local AddDense, parent = torch.class('nn.AddDense', 'nn.Module')

function AddDense:__init(D)
  parent.__init(self)
  self.D = D
end

function AddDense:updateOutput(input)
  self.output = self.output or input.new()
  self.output:resizeAs(input)
  for i = 1, self.D, (#input)[2] - self.D do
    self.output:narrow(2, 1 + (#input)[2] - self.D, self.D):add(self.output:narrow(2, i, self.D))
  end
  return self.output
end

function AddDense:updateGradInput(input, gradOutput)
  self.gradInput = self.gradInput or input.new()
  self.gradInput:resizeAs(input)
  for i = 1, self.D, (#input)[2] - self.D do
    self.gradInput:narrow(2, i, self.D):add(gradOutput:narrow(2, 1 + (#input)[2] - self.D, self.D))
  end
  return self.gradInput
end