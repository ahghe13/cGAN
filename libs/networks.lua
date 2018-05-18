require 'torch'
require 'nn'

include('weights_init.lua')


function mini_networks(opt)
	--######################### GENERATOR ########################--

	local netG = nn.Sequential()
	-- Input consists of noise vector, z, and class parameter(s)
	netG:add(nn.SpatialFullConvolution(opt.nz + table.getn(opt.classes), opt.ngf, 4, 4, 2, 2, 1, 1))
	netG:add(nn.SpatialBatchNormalization(opt.ngf)):add(nn.ReLU(true))
	-- State size: (ngf) x 2 x 2
	netG:add(nn.SpatialFullConvolution(opt.ngf, table.getn(opt.cs), 4, 4, 2, 2, 1, 1))
	netG:add(nn.Tanh())
	-- state size: (cs) x 4 x 4
	netG:apply(weights_init)

	print('HERE')
	print(#netG:get(1).weight)



	--####################### DESCRIMINATOR #######################--

	local netD = nn.Sequential()
	-- input is (cs) x 4 x 4
	netD:add(nn.SpatialConvolution(table.getn(opt.cs), opt.ndf, 4, 4, 2, 2, 1, 1))
	netD:add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf) x 2 x 2
	netD:add(nn.SpatialConvolution(opt.ndf, table.getn(opt.classes) + 1, 4, 4, 2, 2, 1, 1))
	netD:add(nn.Sigmoid())
	-- state size: (classes + 1) x 1 x 1
	netD:add(nn.View(table.getn(opt.classes) + 1):setNumInputDims(3))
	-- state size: (classes + 1)
	netD:apply(weights_init)

	return netG, netD

end

function networks(opt)
	--######################### GENERATOR ########################--

	local netG = nn.Sequential()
	-- Input consists of noise vector, z, and class parameter(s)
	netG:add(nn.SpatialFullConvolution(opt.nz + table.getn(opt.classes), opt.ngf * 8, 4, 4))
	netG:add(nn.SpatialBatchNormalization(opt.ngf * 8)):add(nn.ReLU(true))
	-- state size: (ngf*8) x 4 x 4
	netG:add(nn.SpatialFullConvolution(opt.ngf * 8, opt.ngf * 4, 4, 4, 2, 2, 1, 1))
	netG:add(nn.SpatialBatchNormalization(opt.ngf * 4)):add(nn.ReLU(true))
	-- state size: (ngf*4) x 8 x 8
	netG:add(nn.SpatialFullConvolution(opt.ngf * 4, opt.ngf * 2, 4, 4, 2, 2, 1, 1))
	netG:add(nn.SpatialBatchNormalization(opt.ngf * 2)):add(nn.ReLU(true))
	-- state size: (ngf*2) x 16 x 16
	netG:add(nn.SpatialFullConvolution(opt.ngf * 2, opt.ngf, 4, 4, 2, 2, 1, 1))
	netG:add(nn.SpatialBatchNormalization(opt.ngf)):add(nn.ReLU(true))
	-- state size: (ngf) x 32 x 32
	netG:add(nn.SpatialFullConvolution(opt.ngf, table.getn(opt.cs), 4, 4, 2, 2, 1, 1))
	netG:add(nn.Tanh())
	-- state size: (cs) x 64 x 64

	netG:apply(weights_init)


	--####################### DESCRIMINATOR #######################--

	local netD = nn.Sequential()
	-- input is (cs) x 64 x 64
	netD:add(nn.SpatialConvolution(table.getn(opt.cs), opt.ndf, 4, 4, 2, 2, 1, 1))
	netD:add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf) x 32 x 32
	netD:add(nn.SpatialConvolution(opt.ndf, opt.ndf * 2, 4, 4, 2, 2, 1, 1))
	netD:add(nn.SpatialBatchNormalization(opt.ndf * 2)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf * 2) x 16 x 16
	netD:add(nn.SpatialConvolution(opt.ndf * 2, opt.ndf * 4, 4, 4, 2, 2, 1, 1))
	netD:add(nn.SpatialBatchNormalization(opt.ndf * 4)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf * 4) x 8 x 8
	netD:add(nn.SpatialConvolution(opt.ndf * 4, opt.ndf * 8, 4, 4, 2, 2, 1, 1))
	netD:add(nn.SpatialBatchNormalization(opt.ndf * 8)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf * 8) x 4 x 4
	netD:add(nn.SpatialConvolution(opt.ndf * 8, table.getn(opt.classes) + 1, 4, 4))
	netD:add(nn.Sigmoid())
	-- state size: (classes + 1) x 1 x 1
	netD:add(nn.View(table.getn(opt.classes) + 1):setNumInputDims(3))
	-- state size: (classes +1)

	netD:apply(weights_init)

	return netG, netD

end
