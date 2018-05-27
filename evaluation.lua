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
include('libs/misc.lua')



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

function merge_tensors(t1, t2)
	local length = t1:size(1)
	local width = t1:size(2)
	local t = torch.Tensor(length, width*2)
	for i=1,width do
		t:select(2, i*2-1):copy(t1:select(2,i))
		t:select(2, i*2):copy(t2:select(2,i))
	end
	return t
end

function Pick_Sample(tab, sample_size)
	sample = cloneTable(tab)
	sample = Shuffle(sample)
	while table.getn(sample) > sample_size do
		table.remove(sample)
	end
	return sample
end

function Load_Real_Images(dataSet, row, col)
	local real_imgs_table = Pick_Sample(dataSet, row*col)
	local real_imgs = LoadImgs(PopCol(real_imgs_table, 1), opt.cs, 'minusone2one')
	local imgs = arrange(real_imgs, row, col)
	return imgs, real_imgs_table
end

function Histogram2CSV(hist, path)
	local test_hist_tab = torch.Tensor(hist:size(1), 1)
	test_hist_tab:copy(hist)
	test_hist_tab = Tensor2Table(test_hist_tab)

	Table2CSV({{'Histogram'}}, path, 'w')
	Table2CSV(test_hist_tab, path, 'a')
end

--################### CHOOSE EVALUATION METHOD ####################--

methods = {
	mse = 1,									-- Mini cGAN and Full cGAN
	generate_images = 1,						-- All types of GAN
	transfer_function_analysis_fake = 1,		-- Mini cGAN and Full cGAN
	transfer_function_analysis_real = 1,		-- Mini cGAN and Full cGAN
	kullback_leibler_distance = 1,				-- All types of GAN ()
	deviation_from_ideal = 0					-- Mini GAN only	
}


--#################### SORT NETS INTO TABLE #######################--

--nets_dir_path = arg[1] or '/home/ag/Desktop/Networks05'
nets_dir_path = arg[1] or '/media/ag/F81AFF0A1AFEC4A2/Master Thesis/Networks/Networks_Mini_sGAN'
--nets_dir_path = arg[1] or '/scratch/sdubats/ahghe13/Networks01_nonCuda'

nets_paths = List_Files_in_Dir(nets_dir_path, '.t7')
Exclude_paths(nets_paths, 'epoch')
nets = Nets2Table(nets_paths)
torch.manualSeed(10)


--################# LOAD OPT, TEST, AND VALID #####################--

opt = torch.load(nets_dir_path .. '/opt.t7')
test = CSV2Table(nets_dir_path .. '/test.csv'); table.remove(test, 1)
test = Data:create(test, opt.batchSize, opt.cs)
valid = CSV2Table(nets_dir_path .. '/valid.csv'); table.remove(valid, 1)
valid = Data:create(valid, opt.batchSize, opt.cs)

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

		local c = torch.Tensor(classes_test)
		local im, classes = generate(netG, row, col, table.getn(opt.classes), c)

		if opt.net_name == 'mini_cGAN' then
			im = image.scale(norm_zero2one(im), 2000,1000, 'simple')
			image.save(gen_path .. File_name(nets[i][2]):sub(1,-4) .. '.png', im)
		else 
			save_tif(gen_path .. File_name(nets[i][2]):sub(1,-4) .. '.tif', im)
		end
	end

	if opt.net_name == 'mini_cGAN' then
		local im_test = image.scale(norm_zero2one(test_data), 2000,1000, 'simple')
		local im_valid = image.scale(norm_zero2one(valid_data), 2000,1000, 'simple')
		image.save(gen_path .. 'real_test' .. '.png', im_test)
		image.save(gen_path .. 'real_valid' .. '.png', im_valid)
	else 
		save_tif(gen_path .. 'real_test' .. '.tif', im_test)
		save_tif(gen_path .. 'real_valid' .. '.tif', im_valid)
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
