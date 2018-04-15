--######################## GAN TYPE ########################--

net_type = 'full_size'			-- Choose between 'mini_size' or 'full_size'


--################# OPTIONS FOR MINI_GAN ###################--

opt_mini_size = {
	net_name = 'mini_cGAN',
	data_info = '',
	trainP = 80, testP = 10, validP = 10,

	epochs = 10,				-- Training Epochs
	batchSize = 12,			-- Batch Size
	imDim = 4,				-- Dimension of the image
	cs = {0,1,2},			-- Channel select
	nz = 100,				-- Number of noise elements passed to generator
	ngf = 4,				-- Number of generator filters in the first layer
	ndf = 4,				-- Number of descriminator filters in the first layer
	ne = 50,				-- Number of evaluation iterations

	noise_type = 'gaussian',	-- Choose between 'gaussian', 'uniform_zero2one', or 'uniform_minusone2one'
	save_nets = 1,				-- save every nth network; 0=disable
	save_nets_path = '/media/ag/F81AFF0A1AFEC4A2/Master Thesis/Networks/Networks/',
	display = 1,
	gpu = 1
}


--#################### OPTIONS FOR GAN #####################--

opt_full_size = {
	net_name = 'cGAN',
	data_info = '/home/ahghe13/Campanula_cropped/data_info.csv',
	trainP = 80, testP = 10, validP = 10,

	epochs = 50,			-- Training Epochs
	batchSize = 12,			-- Batch Size
	imDim = 64,				-- Dimension of the image
	cs = {163},				-- Channel select
	nz = 100,				-- Number of noise elements passed to generator
	ngf = 64,				-- Number of generator filters in the first layer
	ndf = 64,				-- Number of descriminator filters in the first layer
	ne = 50,				-- Number of evaluation iterations

	noise_type = 'gaussian',	-- Choose between 'gaussian', 'uniform_zero2one', or 'uniform_minusone2one'
	save_nets = 1,			-- save every nth network; 0=disable
	save_nets_path = '/home/ahghe13/Networks/',
	display = 1,
	gpu = 1
}
