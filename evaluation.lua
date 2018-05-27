require 'torch'
require 'nn'
require 'image'

include('libs/data.lua')
include('libs/table_handling.lua')
include('libs/tif_handling.lua')
include('libs/generate.lua')
include('libs/image_normalization.lua')
include('libs/paths_handling.lua')
include('libs/image_distance.lua')
include('libs/tensor_handling.lua')



function MSE(netD, data, tar)
	local out = netD:forward(data)
	local out = Remove_col(out, 1)
	local tar = Remove_col(tar, 1)

	local mse = nn.MSECriterion()
	if data:type() == 'torch.cudaTensor' then; mse:cuda(); end
	return mse:forward(tar, out)
end

function Generate_data(netG, batch_size, number_of_classes)
	local noise_c, class = generate_noise_c(netG:get(1).nInputPlane, number_of_classes, batch_size)
   	local classes = Cat_vector(class, 0)
	if netG:get(1).weight:type() == 'torch.CudaTensor' then; noise_c, class = noise_c:cuda(), class:cuda(); end
	local generated_imgs = netG:forward(noise_c)
	return generated_imgs, classes
end

function Generate_first_row(names_of_classes)
	local output = {'image_gt', 'image_p'}
	for i=1,table.getn(opt.classes) do
		table.insert(output, names_of_classes[i] .. '_gt')
		table.insert(output, names_of_classes[i] .. '_p')
	end
	return {output}
end

--################### CHOOSE EVALUATION METHOD ####################--

methods = {
	mse = 0,									-- Mini cGAN and Full cGAN
	generate_images = 0,						-- All types of GAN
	generate_images_cGAN = 1,					-- Mini cGAN and Full cGAN
	transfer_function_analysis_fake = 0,		-- Mini cGAN and Full cGAN
	transfer_function_analysis_real = 0,		-- Mini cGAN and Full cGAN
	kullback_leibler_distance = 0,				-- All types of GAN ()
	deviation_from_ideal = 0					-- Mini GAN only	
}


--#################### SORT NETS INTO TABLE #######################--

--nets_dir_path = arg[1] or '/home/ag/Desktop/Networks05'
nets_dir_path = arg[1] or '/media/ag/F81AFF0A1AFEC4A2/Master Thesis/Networks/Networks_Mini_sGAN'
--nets_dir_path = arg[1] or '/scratch/sdubats/ahghe13/Networks01_nonCuda'

nets = getNets(nets_dir_path)
torch.manualSeed(10)


--################# LOAD OPT, TEST, AND VALID #####################--

opt = torch.load(nets_dir_path .. '/opt.t7')
test_tab = CSV2Table(nets_dir_path .. '/test.csv'); table.remove(test_tab, 1)
test = Data:create(test_tab, opt.batchSize, opt.cs)
valid_tab = CSV2Table(nets_dir_path .. '/valid.csv'); table.remove(valid_tab, 1)
valid = Data:create(valid_tab, opt.batchSize, opt.cs)

test_data, test_target = test:getData(); test_target = Cat_vector(test_target, 1)
valid_data, valid_target = valid:getData(); valid_target = Cat_vector(valid_target, 1)

if opt.gpu > 0 then
	require 'cunn'
	test_data, test_target = test_data:cuda(), test_target:cuda()
	valid_data, valid_target = valid_data:cuda(), valid_target:cuda()
end

print(opt)

--######### APPLY EVALUATION METHODS FOR DESCRIMINATOR ############--

evaluation_path = nets_dir_path .. '/Evaluation/'
paths.mkdir(evaluation_path)

Table2CSV({{'Epoch', 'Generator_path', 'Descriminator_path'}}, evaluation_path .. '/Networks.csv')
Table2CSV(nets, evaluation_path .. '/Networks.csv', 'a')


--                              MSE                                --

if methods.mse == 1 then
	io.write('Computing MSE... '):flush()
	tableMSE = {{'Epoch', 'MSE (test)', 'MSE (valid)'}}

	for i=1,table.getn(nets) do
		local netD = torch.load(nets[i][3])
		local testMSE = MSE(netD, test_data, test_target)
		local validMSE = MSE(netD, valid_data, valid_target)
		table.insert(tableMSE, {nets[i][1], testMSE, validMSE})
	end
	Table2CSV(tableMSE, evaluation_path .. '/Mean_square_error.csv')

	print('Done!')
end

--                         GENERATE IMAGES                         --

if methods.generate_images == 1 then
	local row, col = 5, 10

	io.write('Generating images... '):flush()
	local gen_path = evaluation_path .. '/Generated_images/'
	paths.mkdir(gen_path)

	for i=1,table.getn(nets) do
		local netG = torch.load(nets[i][2])

		local im = generate(netG, row, col, table.getn(opt.classes))

		if opt.net_name == 'mini_cGAN' then
			im = image.scale(norm_zero2one(im), 2000,1000, 'simple')
			image.save(gen_path .. File_name(nets[i][2]):sub(1,-4) .. '.png', im)
		else 
			save_tif(gen_path .. File_name(nets[i][2]):sub(1,-4) .. '.tif', im)
		end
	end

	local test_sample = test:getData(row*col)
	test_sample = arrange(test_sample, row, col)
	if opt.net_name == 'mini_cGAN' then
		test_sample = image.scale(norm_zero2one(test_sample), 2000,1000, 'simple')
		image.save(gen_path .. 'real_test' .. '.png', test_sample)
	else 
		save_tif(gen_path .. 'real_test' .. '.tif', test_sample)
	end

	print('Done!')
end


--                       GENERATE IMAGES cGAN                       --

if methods.generate_images_cGAN == 1 then
	local row, col = 5, 10

	io.write('Generating images with classes... '):flush()
	local gen_path = evaluation_path .. '/Generated_images_with_classes/'
	paths.mkdir(gen_path)

	local test_sample, test_classes = test:getData(row*col)
	test_sample = arrange(test_sample, row, col)
	if opt.net_name == 'mini_cGAN' then
		test_sample = image.scale(norm_zero2one(test_sample), 2000,1000, 'simple')
		image.save(gen_path .. 'real_test' .. '.png', test_sample)
	else 
		save_tif(gen_path .. 'real_test' .. '.tif', test_sample)
	end

	Table2CSV(Tensor2Table_beta(test_classes), gen_path .. 'classes.csv')

	for i=1,table.getn(nets) do
		local netG = torch.load(nets[i][2])

		local im, classes = generate(netG, row, col, table.getn(opt.classes), test_classes)

		if opt.net_name == 'mini_cGAN' then
			im = image.scale(norm_zero2one(im), col*200,row*200, 'simple')
			image.save(gen_path .. File_name(nets[i][2]):sub(1,-4) .. '.png', im)
		else 
			save_tif(gen_path .. File_name(nets[i][2]):sub(1,-4) .. '.tif', im)
		end
	end

	print('Done!')
end

--             TRANSFER FUNCTION ANALYSIS - FAKE IMAGES            --

if methods.transfer_function_analysis_fake == 1 then

	io.write('Transfer function analysis with fake image... '):flush()
	local trans_path = evaluation_path .. '/Transfer_function_Fake/'
	paths.mkdir(trans_path)

	local batch_size = 100

	for i=1,table.getn(nets) do
		netG = torch.load(nets[i][2])
		netD = torch.load(nets[i][3])
		local data_batch, class = Generate_data(netG, batch_size, table.getn(opt.classes))
		local outputD = netD:forward(data_batch)

		local result = merge_tensors(class, outputD)
		local result_table = Tensor2Table(result)

		local output_file = trans_path .. 'Epoch' .. i .. '.csv'
		Table2CSV(Generate_first_row(opt.classes), output_file, 'w')
		Table2CSV(result_table, output_file, 'a')
	end

	print('Done!')
end

--             TRANSFER FUNCTION ANALYSIS - REAL IMAGES            --

if methods.transfer_function_analysis_real == 1 then

	io.write('Transfer function analysis with real images... '):flush()
	-- Make directory
	local trans_path = evaluation_path .. '/Transfer_function_Real/'
	paths.mkdir(trans_path)

	for i=1,table.getn(nets) do
		netD = torch.load(nets[i][3])

		local outputD = netD:forward(test_data)

		local result = merge_tensors(test_target, outputD)
		local result_table = Tensor2Table(result)

		local output_file = trans_path .. 'Epoch' .. i .. '.csv'
		Table2CSV(Generate_first_row(opt.classes), output_file, 'w')
		Table2CSV(result_table, output_file, 'a')
	end

	print('Done!')
end


--                     KULLBACK-LEIBLER DISTANCE                   --

if methods.kullback_leibler_distance == 1 then
	local sample_size = 150

	io.write('Computing Kullback-Leibler distance... '):flush()
	local kl_path = evaluation_path .. '/Kullback-Leibler/'
	paths.mkdir(kl_path)

	local test_distances = images_distance(test_data, sample_size)
	test_hist = torch.histc(test_distances)

	kl_crit = nn.DistKLDivCriterion()
	kl_distances = {{'Epoch', 'KL Distance'}}

	for i=1,table.getn(nets) do
		local netG = torch.load(nets[i][2])
		local fake_data = Generate_data(netG, sample_size, table.getn(opt.classes))
		fake_data = norm_minusone2one_beta(fake_data)
		local fake_data_distances = images_distance(fake_data, sample_size)
		local fake_data_hist = torch.histc(fake_data_distances)
		local distance = kl_crit:forward(torch.log(fake_data_hist:add(1)), test_hist)
		table.insert(kl_distances, {nets[i][1], distance})

		Histogram2CSV(fake_data_hist, kl_path .. 'epoch' .. i ..  '_iid_histogram.csv')
	end

	Histogram2CSV(test_hist, kl_path .. 'test_data_iid_histogram.csv')

	Table2CSV(kl_distances, kl_path .. 'Kullback-Leibler_distance.csv', 'w')

	print('Done!')
end


--                     DEVIATION FROM IDEAL                   --

if methods.deviation_from_ideal == 1 then
	local sample_size = 10000
	local ideal_path = '/media/ag/F81AFF0A1AFEC4A2/Master Thesis/Data/ideal.tif'

	io.write('Computing deviation from ideal... '):flush()
	local ideal = load_tif(ideal_path, 'all', 'minusone2one')
	dev = {{'Epoch', 'Mean', 'Standard Deviation', 'Mean (train)', 'Standard Deviation (train)'}}

	local train = CSV2Table(nets_dir_path .. '/train.csv'); table.remove(train, 1)
	local train_data, train_target = Load_Data(train, opt.cs)
	local linear_dist = Linear_Images_distance(train_data, ideal)
	local m_t, s_t = torch.mean(linear_dist), torch.std(linear_dist)


	for i=1,table.getn(nets) do
		local netG = torch.load(nets[i][2])
		local fake_data = Generate_data(netG, sample_size, table.getn(opt.classes))
		fake_data = norm_minusone2one_beta(fake_data)
		local linear_dist = Linear_Images_distance(fake_data, ideal)
		table.insert(dev, {nets[i][1], torch.mean(linear_dist), torch.std(linear_dist),m_t, s_t})
	end

	local output_file = evaluation_path .. 'Deviation_from_ideal.csv'
	Table2CSV(dev, output_file, 'w')

	print('Done!')
end
