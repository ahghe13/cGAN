require 'torch'
require 'nn'
require 'image'

include('libs/list_files_in_dir.lua')
include('libs/table_handling.lua')
include('libs/csv_to_table.lua')
include('libs/tif_handler.lua')
include('libs/generate.lua')
include('libs/image_normalization.lua')

function File_name(path) -- extracts filename from full path
	return path:sub(path:match('.*()/')+1)
end

function Exclude_paths(paths, key)
	-- Excludes paths to files whose name does not include key
	bad_paths = {}
	for i=1, table.getn(paths) do
		if File_name(paths[i]):match(key) == nil then
			table.insert(bad_paths, i)
		end
	end

	local l = table.getn(bad_paths)
	for i=1,l do
		table.remove(paths, bad_paths[l-i+1])
	end
end

function get_epoch(path)
	local file = File_name(path)
	local epoch = file:match('h.*_')
	epoch = epoch:sub(2, epoch:len()-1)
	return tonumber(epoch)
end

function get_net_type(path)
	local file = File_name(path)
	local net_type = file:match('_.*')
	net_type = net_type:sub(2, net_type:len()-3)
	return net_type
end

function Forward_netD(netD, dataSet, cs)
	local netData = cloneTable(dataSet)
	local paths = PopCol(netData, 1)
	local target_output = torch.cat(torch.Tensor(#paths):fill(1), torch.Tensor(netData), 2)
	local imgs = LoadImgs(paths, cs, 'minusone2one')
	local output = netD:forward(imgs)
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

function MSE(netD, data, cs)
	local tar, out = Forward_netD(netD, data, cs)

	local out = Remove_col(out, 1)
	local tar = Remove_col(tar, 1)

	local mse = nn.MSECriterion()
	return mse:forward(tar, out)
end

function Generate_data(netG, batch, opt)
	local inputG = torch.Tensor(batch, opt.nz + table.getn(opt.classes), 1, 1)
	local noise = torch.Tensor(batch, opt.nz, 1, 1)
	local class = torch.Tensor(batch, table.getn(opt.classes))

	if opt.noise_type == 'gaussian' then; noise:normal()
	elseif opt.noise_type == 'uniform_zero2one' then; noise:uniform(0,1)
	elseif opt.noise_type == 'uniform_minusone2one' then; noise:uniform(-1,1); end
	class:select(2,1):uniform(0,1)
	class:select(2,2):fill(0)
	inputG = torch.cat(noise, class, 2)
	return netG:forward(inputG), class
end

function Load_Data(dataSet, cs, normalize)
	-- Loads data from table's first column
	local normalize = normalize or 'minusone2one' 
	local data = cloneTable(dataSet)
	local paths = PopCol(data, 1)
	return LoadImgs(paths, cs, normalize), data
end


--################### CHOOSE EVALUATION METHOD ####################--

methods = {
	mse = 0,
	generate_images = 1,
	transfer_function_analysis_fake = 1,
	transfer_function_analysis_real = 1

}

--#################### SORT NETS INTO TABLE #######################--

--nets_dir_path = '/media/ag/F81AFF0A1AFEC4A2/Master Thesis/Networks/Networks'
nets_dir_path = '/home/ahghe13/Networks'

nets_paths = List_Files_in_Dir(nets_dir_path, '.t7')
Exclude_paths(nets_paths, 'epoch')
evaluation = Nets2Table(nets_paths)


--################# LOAD OPT, TEST, AND VALID #####################--

opt = torch.load(nets_dir_path .. '/opt.t7')
test = CSV2Table(nets_dir_path .. '/test.csv')
table.remove(test, 1)
valid = CSV2Table(nets_dir_path .. '/valid.csv')
table.remove(valid, 1)
if opt.gpu > 0 then; require 'cunn'; end

--######### APPLY EVALUATION METHODS FOR DESCRIMINATOR ############--

parameters = {'Epoch', 'Generator_path', 'Descriminator_path'}

--                              MSE                                --

if methods.mse == 1 then
	io.write('Computing MSE... '):flush()
	table.insert(parameters, 'MSE (test)')
	table.insert(parameters, 'MSE (valid)')
	for i=1,table.getn(evaluation) do
		netD = torch.load(evaluation[i][3])
		table.insert(evaluation[i], MSE(netD, test, opt.cs))
		table.insert(evaluation[i], MSE(netD, valid, opt.cs))
	end
	print('Done!')
end
--                         GENERATE IMAGES                         --

if methods.generate_images == 1 then
	io.write('Generating images... '):flush()
	gen_path = nets_dir_path .. '/Generated_images/'
	paths.mkdir(gen_path)

	for i=1,table.getn(evaluation) do
		netG = torch.load(evaluation[i][2])
		im = generate(netG, 5, 10, table.getn(opt.classes))
		if opt.net_name == 'mini_cGAN' then
			im = image.scale(im, 2000,1000, 'simple')
			image.save(gen_path .. File_name(evaluation[i][2]):sub(1,-4) .. '.jpg', im)
		else 
			save_tif(gen_path .. File_name(evaluation[i][2]):sub(1,-4) .. '.tif', im)
		end
	end
	print('Done!')
end

--             TRANSFER FUNCTION ANALYSIS - FAKE IMAGES            --

if methods.transfer_function_analysis_fake == 1 then 
	io.write('Transfer function analysis with fake image... '):flush()
	trans_path = nets_dir_path .. '/Transfer_function_Fake/'
	paths.mkdir(trans_path)

	local batch = 100

	for i=1,table.getn(evaluation) do
		local output = {'image_gt', 'image_p'}
		for i=1,table.getn(opt.classes) do
			table.insert(output, opt.classes[i] .. '_gt')
			table.insert(output, opt.classes[i] .. '_p')
		end
		output = {output}
		netG = torch.load(evaluation[i][2])
		netD = torch.load(evaluation[i][3])
		local data_batch, class = Generate_data(netG, batch, opt)
		local outputD = netD:forward(data_batch)

		for i=1,batch do
			local output_tmp = {0, outputD[i][1]}
			for j=1,class:size(2) do
				table.insert(output_tmp, class[i][j])
				table.insert(output_tmp, outputD[i][j+1])
			end
			table.insert(output, output_tmp)
		end

		local output_file = trans_path .. 'Epoch' .. i .. '.csv'
		Table2CSV(output, output_file, 'w')
	end

	print('Done!')
end

--             TRANSFER FUNCTION ANALYSIS - REAL IMAGES            --

if methods.transfer_function_analysis_real == 1 then

	io.write('Transfer function analysis with real images... '):flush()
	trans_path = nets_dir_path .. '/Transfer_function_Real/'
	paths.mkdir(trans_path)
	local data_batch, class = Load_Data(test, opt.cs)

	for i=1,table.getn(evaluation) do
		local output = {'image_gt', 'image_p'}
		for i=1,table.getn(opt.classes) do
			table.insert(output, opt.classes[i] .. '_gt')
			table.insert(output, opt.classes[i] .. '_p')
		end
		output = {output}
		netG = torch.load(evaluation[i][2])
		netD = torch.load(evaluation[i][3])
		local outputD = netD:forward(data_batch)

		for i=1,data_batch:size(1) do
			local output_tmp = {0, outputD[i][1]}
			for j=1,table.getn(class[1]) do
				table.insert(output_tmp, class[i][j])
				table.insert(output_tmp, outputD[i][j+1])
			end
			table.insert(output, output_tmp)
		end

		local output_file = trans_path .. 'Epoch' .. i .. '.csv'
		Table2CSV(output, output_file, 'w')
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
