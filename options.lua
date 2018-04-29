--######################## GAN TYPE ########################--

net_type = 'full_size'			-- Choose between 'mini_size' or 'full_size'


--################# OPTIONS FOR MINI_GAN ###################--

opt_mini_size = {
	net_name = 'mini_cGAN',
	data_info = '/home/ag/Dropbox/Uni/Master thesis/Software/Data/Mini_cGAN1/data_info.csv',
	trainP = 80, testP = 10, validP = 10,

	epochs = 30,				-- Training Epochs
	batchSize = 12,			-- Batch Size
	imDim = 4,				-- Dimension of the image
	cs = {0,1,2},			-- Channel select
	nz = 100,				-- Number of noise elements passed to generator
	ngf = 4,				-- Number of generator filters in the first layer
	ndf = 4,				-- Number of descriminator filters in the first layer

	noise_type = 'gaussian',	-- Choose between 'gaussian', 'uniform_zero2one', or 'uniform_minusone2one'
	save_nets = 1,				-- save every nth network; 0=disable
	save_nets_path = '/media/ag/F81AFF0A1AFEC4A2/Master Thesis/Networks/Networks/',
	display = 1,
	gpu = 0
}


--#################### OPTIONS FOR GAN #####################--

opt_full_size = {
	net_name = 'cGAN',
--	data_info = '/scratch/sdubats/ahghe13/data/Campanula_cropped/data_info.csv',
	data_info = '/media/ag/F81AFF0A1AFEC4A2/Master Thesis/Data/Campanula_cropped/data_info_no_classes.csv',
	trainP = 80, testP = 10, validP = 10,

	epochs = 20,			-- Training Epochs
	batchSize = 12,			-- Batch Size
	imDim = 64,				-- Dimension of the image
	cs = {163},				-- Channel select. For all channels, write {'all'}
	nz = 100,				-- Number of noise elements passed to generator
	ngf = 64,				-- Number of generator filters in the first layer
	ndf = 64,				-- Number of descriminator filters in the first layer

	noise_type = 'gaussian',	-- Choose between 'gaussian', 'uniform_zero2one', or 'uniform_minusone2one'
	save_nets = 1,			-- save every nth network; 0=disable
--	save_nets_path = '/scratch/sdubats/ahghe13/Networks/',
	save_nets_path = '/media/ag/F81AFF0A1AFEC4A2/Master Thesis/Networks/Networks/',
	display = 1,
	gpu = 0
}
