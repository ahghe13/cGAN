require 'torch'
require 'nn'
require 'image'

include('libs/list_files_in_dir.lua')
include('libs/table_handling.lua')
include('libs/csv_to_table.lua')
include('libs/tif_handler.lua')
include('libs/generate.lua')
include('libs/image_normalization.lua')
include('libs/paths_handling.lua')


function Forward_netD(netD, dataSet, cs)
	local netData = cloneTable(dataSet)
	local paths = PopCol(netData, 1)
	local target_output = torch.cat(torch.Tensor(#paths):fill(1), torch.Tensor(netData), 2)
print('loading images...')
	local imgs = LoadImgs(paths, cs, 'minusone2one')
print('images loaded. Now, forwarding them in netD')
	local output = netD:forward(dataSet)
	return target_output, output
end

function Remove_col(tensor, idx)
	local indices = torch.LongTensor(tensor:size(2)-1)
	local ii = 1
	for i=1,tensor:size(2) do
		if i ~= idx then; indices[ii] = i; ii = ii + 1; end
	end
	return tensor:index(2, indices)
end

function Nets2Table(netspaths)
	nets = cloneTable(netspaths)
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
print('Forwarding data')
	local out = netD:forward(data)
	local out = Remove_col(out, 1)
	local tar = Remove_col(tar, 1)

	local mse = nn.MSECriterion()
	return mse:forward(tar, out)
end

function Generate_data(netG, batch_size, number_of_classes)
	local noise_c, class = generate_noise_c(netG:get(1).nInputPlane, number_of_classes, batch_size)
	local generated_imgs = netG:forward(noise_c)
	local classes = Cat_vector(class, 0)
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
	-- and adds a one to the first column
	local class_values = cloneTable(dataSet)
	PopCol(class_values,1)
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

--################### CHOOSE EVALUATION METHOD ####################--

methods = {
	mse = 0,
	generate_images = 1,
	transfer_function_analysis_fake = 0,
	transfer_function_analysis_real = 0

}


--#################### SORT NETS INTO TABLE #######################--

nets_dir_path = '/home/ag/Desktop/Networks05'
--nets_dir_path = '/media/ag/F81AFF0A1AFEC4A2/Master Thesis/Networks/Networks'
--nets_dir_path = '/scratch/sdubats/ahghe13/Networks01_nonCuda'

nets_paths = List_Files_in_Dir(nets_dir_path, '.t7')
Exclude_paths(nets_paths, 'epoch')
evaluation = Nets2Table(nets_paths)


--################# LOAD OPT, TEST, AND VALID #####################--

opt = torch.load(nets_dir_path .. '/opt.t7')
test = CSV2Table(nets_dir_path .. '/test.csv')
table.remove(test, 1)
valid = CSV2Table(nets_dir_path .. '/valid.csv')
table.remove(valid, 1)
print(opt)


--######### APPLY EVALUATION METHODS FOR DESCRIMINATOR ############--

parameters = {'Epoch', 'Generator_path', 'Descriminator_path'}

--                              MSE                                --

if methods.mse == 1 then
	print('Computing MSE')
--	io.write('Computing MSE... '):flush()
	table.insert(parameters, 'MSE (test)')
	table.insert(parameters, 'MSE (valid)')
	local test_data = Load_Data(test, opt.cs)
	local test_target = Load_Target(test)
	local valid_data = Load_Data(valid, opt.cs)
	local valid_target = Load_Target(valid)

	for i=1,table.getn(evaluation) do
print('Loading network ' .. i .. ': ' .. evaluation[i][3])
		netD = torch.load(evaluation[i][3])
print('Computing MSE using test set')
		table.insert(evaluation[i], MSE(netD, test_data, test_target))
print('Computing MSE using valid set')
		table.insert(evaluation[i], MSE(netD, valid_data, valid_target))
	end
	print('Done!')
end

--                         GENERATE IMAGES                         --

if methods.generate_images == 1 then
	io.write('Generating images... '):flush()
	gen_path = nets_dir_path .. '/Generated_images/'
	paths.mkdir(gen_path)

	local row, col = 5, 10
	generator_batch_size = row*col

	for i=1,table.getn(evaluation) do
		netG = torch.load(evaluation[i][2])

		im = generate(netG, row, col, table.getn(opt.classes))
		if opt.net_name == 'mini_cGAN' then
			im = image.scale(im, 2000,1000, 'simple')
			image.save(gen_path .. File_name(evaluation[i][2]):sub(1,-4) .. '.jpg', im)
		else 
			save_tif(gen_path .. File_name(evaluation[i][2]):sub(1,-4) .. '.tif', im)
		end
	end

	local real_imgs_table = cloneTable(valid)
	while table.getn(real_imgs_table) > generator_batch_size do
		table.remove(real_imgs_table)
	end
	local real_imgs = LoadImgs(PopCol(real_imgs_table, 1), opt.cs, 'minusone2one')
	local im = arrange(real_imgs, row, col)
	if opt.net_name == 'mini_cGAN' then
		im = image.scale(im, 2000,1000, 'simple')
		image.save(gen_path .. 'real' .. '.jpg', im)
	else 
		save_tif(gen_path .. 'real' .. '.tif', im)
	end


	print('Done!')
end

--             TRANSFER FUNCTION ANALYSIS - FAKE IMAGES            --

if methods.transfer_function_analysis_fake == 1 then

	io.write('Transfer function analysis with fake image... '):flush()
	trans_path = nets_dir_path .. '/Transfer_function_Fake/'
	paths.mkdir(trans_path)

	local batch_size = 100

	for i=1,table.getn(evaluation) do
		netG = torch.load(evaluation[i][2])
		netD = torch.load(evaluation[i][3])

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
	trans_path = nets_dir_path .. '/Transfer_function_Real/'
	paths.mkdir(trans_path)

	local data_batch, class = Load_Data(test, opt.cs)

	for i=1,table.getn(evaluation) do
		netD = torch.load(evaluation[i][3])

		local outputD = netD:forward(data_batch)

		local result = merge_tensors(class, outputD)
		local result_table = Tensor2Table(result)

		local output_file = trans_path .. 'Epoch' .. i .. '.csv'
		Table2CSV(Generate_first_row(opt.classes), output_file, 'w')
		Table2CSV(result_table, output_file, 'a')
	end

	print('Done!')
end

--######################### SAVE RESULTS ##########################--

Table2CSV({parameters}, nets_dir_path .. '/evaluation.csv')
Table2CSV(evaluation, nets_dir_path .. '/evaluation.csv', 'a')

--[[

Evaluation methods
------------------


Generator
	- Generate images
	- Image distance?

Descriminator
	- MSE
	- 

]]--
