require 'torch'
require 'nn'

include('libs/data.lua')
include('libs/paths_handling.lua')
include('libs/table_handling.lua')
include('libs/tensor_handling.lua')

source_path = '/media/ag/F81AFF0A1AFEC4A2/Master Thesis/SFFS/'			--Empty folder to save results
main = '/home/ag/Dropbox/Uni/Master thesis/Software/cGAN/cGAN/main.lua'	

--F = {}; for i=1,224 do; F[i] = i-1; end;		-- Full
--F = {10,11,12,13,14,15,16,17,101,113,223}		-- CBFS
--F = {111,113,115,117,119,121,123,125,127,129}
F = {0,1,2}
print(F)
evaluation_parameter = 'brightness'

function sGAN_Performance(sgan_path)
	local nets = getNets(sgan_path)

	local opt = torch.load(sgan_path .. '/opt.t7')
	local test = CSV2Table(sgan_path .. '/test.csv'); table.remove(test, 1)
	test = Data:create(test, opt.batchSize, opt.cs)
	local valid = CSV2Table(sgan_path .. '/valid.csv'); table.remove(valid, 1)
	valid = Data:create(valid, opt.batchSize, opt.cs)
	local test_data, test_target = test:getData(); test_target = Cat_vector(test_target, 1)
	local valid_data, valid_target = valid:getData(); valid_target = Cat_vector(valid_target, 1)
	local ep = Find_Element(opt.classes, evaluation_parameter)+1

	local criterion = nn.MSECriterion()
	local performance = torch.Tensor(table.getn(nets), 2)

	if opt.gpu > 0 then
		require 'cunn'
		test_data, test_target = test_data:cuda(), test_target:cuda()
		valid_data, valid_target = valid_data:cuda(), valid_target:cuda()
		criterion, performance = criterion:cuda(), performance:cuda()
	end


	for i=1,table.getn(nets) do
		local netD = torch.load(nets[i][3])

		local outputD = netD:forward(test_data)
		performance[i][1] = criterion:forward(outputD:select(2,ep), test_target:select(2,ep))
		local outputD = netD:forward(valid_data)
		performance[i][2] = criterion:forward(outputD:select(2,ep), valid_target:select(2,ep))
	end

	local v, i = torch.min(performance,1); i = i[1][1]

	Table2CSV({{'MSE (Test)', 'MSE (Valid)'}}, sgan_path .. 'mse_' .. ep .. '.csv')
	Table2CSV(Tensor2Table_beta(performance), sgan_path .. 'mse_' .. ep .. '.csv', 'a')

	return performance[i][2]
end

function Train_sGAN(sgan_path, features)
	paths.mkdir(sgan_path)
	arg = {sgan_path, features}
	dofile(main)
	return sgan_path
end

function Commutative(tab, a)
	output = cloneTable(tab)
	table.insert(output, a)
	return output
end

function Table2String(tab)
	local str = ""
	for i=1,table.getn(tab) do
		str = str .. tab[i] .. " "
	end
	return str
end

function Find_Element(tab, element)
	for i=1,table.getn(tab) do
		if tab[i] == element then
			return i
		end
	end
	return 0
end

Table2CSV({{'Key', evaluation_parameter}}, source_path .. 'SFFS.csv')

-- ###################### SFFS ###################### --

S = {}
p_best = 99999

for i=1,table.getn(F) do
	R = {}
	for j=1,table.getn(F) do
		if Find_Element(S, F[j]) == 0 then
			local key = i .. '-' .. Table2String(S) .. F[j]
			local sgan = Train_sGAN(source_path .. key .. '/', Commutative(S, F[j]))
			local p = sGAN_Performance(sgan)
			table.insert(R, {p, F[j]})
			Table2CSV({{key, p}}, source_path .. 'SFFS.csv', 'a')
		end
	end
	local M, I = torch.min(torch.Tensor(R), 1); I = I[1][1]
	if R[I][1] < p_best then
		p_best = R[I][1]
		table.insert(S, R[I][2])
	else
		break;
	end
end

print(S)

