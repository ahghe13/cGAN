-- Evaluation parameters
require 'torch'

require 'libs/merge_lists'


batchSize = torch.Tensor(5,1)
for i=1,batchSize:size()[1] do
  batchSize[i] = i+10
end

nz = torch.Tensor(5,1)
for i=1,nz:size()[1] do
  nz[i] = i+10
end

params = merge(batchSize,nz)

for i=1,params:size()[1] do
	os.execute('BATCH_SIZE=' .. params[i][1] .. ' NZ=' .. params[i][2] .. ' qlua GAN.lua')
	print(i*100/params:size()[1] .. '%')
end

