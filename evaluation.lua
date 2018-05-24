require 'torch'
require 'nn'
require 'image'

include('libs/table_handling.lua')
include('libs/tif_handling.lua')
include('libs/generate.lua')
include('libs/image_normalization.lua')
include('libs/paths_handling.lua')
include('libs/image_distance.lua')

function Remove_col(tensor, idx)
	local indices = torch.LongTensor(tensor:size(2)-1)
	local ii = 1
	for i=1,tensor:size(2) do
		if i ~= idx then; indices[ii] = i; ii = ii + 1; end
	end
	return tensor:index(2, indices)
end

function Nets2Table(netspaths)
	local nets = cloneTable(netspaths)
	local sorted_nets = {}
	local e = 9999
	local prev_e = 0
	local netG_path = ''
	local netD_path = ''

	for i=1,table.getn(nets)/2 do
		for j=1,table.getn(nets) do
			if get_epoch(nets[j]) <= e and get_epoch(nets[j]) > prev_e then
				e = get_epoch(nets[j])
				if get_net_type(nets[j]) == 'netG' then; netG_path = nets[j]; end
				if get_net_type(nets[j]) == 'netD' then; netD_path = nets[j]; end
			end
		end
		table.insert(sorted_nets, {e, netG_path, netD_path})
		prev_e = e
		e = 9999
	end
	return sorted_nets
end

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

function Load_Data(dataSet, cs, normalize)
	-- Loads images from table's first column
	-- and class-values from the other columns
	local normalize = normalize or 'minusone2one' 
	local data = cloneTable(dataSet)
	local paths = PopCol(data, 1)
	return LoadImgs(paths, cs, normalize), Load_Target(dataSet)
end

function Load_Target(dataSet)
	-- Takes table as input (w. paths and class-values),
	-- converts the class-values to a tensor,
	-- and adds a one-column in the beginning
	local class_values = cloneTable(dataSet)
	PopCol(class_values,1)
	if table.getn(class_values[1]) == 0 then; return torch.Tensor(table.getn(class_values)):fill(1); end
	class_values = torch.Tensor(class_values)
	class_values_and_one_vector = Cat_vector(class_values, 1)
    return class_values_and_one_vector
end

function Cat_vector(tensor, vector_value)
	local vector = torch.Tensor(tensor:size(1)):fill(vector_value)
	local tensor_and_vector = torch.cat(vector, tensor, 2)
	return tensor_and_vector
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
	mse = 0,									-- Mini cGAN and Full cGAN
	generate_images = 1,						-- All types of GAN
	transfer_function_analysis_fake = 0,		-- Mini cGAN and Full cGAN
	transfer_function_analysis_real = 0,		-- Mini cGAN and Full cGAN
	kullback_leibler_distance = 0,				-- All types of GAN ()
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
test = CSV2Table(nets_dir_path .. '/test.csv')
table.remove(test, 1)
valid = CSV2Table(nets_dir_path .. '/valid.csv')
table.remove(valid, 1)
print(opt)

test_data, test_target = Load_Data(test, opt.cs)
valid_data, valid_target = Load_Data(valid, opt.cs)

if opt.gpu > 0 then
	require 'cunn'
	test_data, test_target = test_data:cuda(), test_target:cuda()
	valid_data, valid_target = valid_data:cuda(), valid_target:cuda()
end


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
		netG = torch.load(nets[i][2])

		local im, classes = generate(netG, row, col, table.getn(opt.classes))
		if classes ~= nil then
			classes =  classes:reshape(classes:size(1), 1)
			classes = Tensor2Table(classes)
			Table2CSV(classes, gen_path .. 'classes_epoch' .. i .. '.csv')
		end

		if opt.net_name == 'mini_cGAN' then
			im = image.scale(norm_zero2one(im), 2000,1000, 'simple')
			image.save(gen_path .. File_name(nets[i][2]):sub(1,-4) .. '.png', im)
		else 
			save_tif(gen_path .. File_name(nets[i][2]):sub(1,-4) .. '.tif', im)
		end
	end

	local im_test, classes_test = Load_Real_Images(test, row, col)
	local im_valid, classes_valid = Load_Real_Images(valid, row, col)
	if opt.net_name == 'mini_cGAN' then
		im_test = image.scale(norm_zero2one(im_test), 2000,1000, 'simple')
		im_valid = image.scale(norm_zero2one(im_valid), 2000,1000, 'simple')
		image.save(gen_path .. 'real_test' .. '.png', im_test)
		image.save(gen_path .. 'real_valid' .. '.png', im_valid)
	else 
		save_tif(gen_path .. 'real_test' .. '.tif', im_test)
		save_tif(gen_path .. 'real_valid' .. '.tif', im_valid)
	end

	Table2CSV(classes_test, gen_path .. 'classes_test.csv')
	Table2CSV(classes_valid, gen_path .. 'classes_valid.csv')

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
