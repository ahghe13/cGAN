require 'torch'
require 'nn'

require 'libs/table_handling'
require 'libs/tif_handler'



function MSE_ideal_output(netG, ideal, opt)
	local dim = #ideal
	local target = torch.Tensor(opt.ne, dim[1], dim[2], dim[3])
	for i=1, opt.ne do; target[i] = ideal; end

	local noise = torch.rand(opt.ne, opt.nz, 1, 1)
	local output = netG:forward(noise)

	local criterion = nn.MSECriterion()
	local err = criterion(target, output)

	local file = io.open('Evaluation/MSE_ideal_output.csv', 'a')
	file:write(err .. ","  .. opt.ne .. "," .. opt.epochs .. "," .. opt.batchSize .. ","
		   .. opt.nz .. "," .. opt.ngf .. "," .. opt.ndf .. "," .. opt.nc .. "\n")
	file:close()
end

function Simple(netD, testSet, cs)
	local testData = cloneTable(testSet)
	local paths = PopCol(testData, 1)
	local target_output = torch.cat(torch.Tensor(#paths):fill(1), torch.Tensor(testData), 2)
	local imgs = LoadImgs(paths, cs)
	local output = netD:forward(imgs)
	return target_output, output
end

