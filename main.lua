require 'torch'
require 'nn'
require 'image'

include('options.lua')
include('libs/networks.lua')
include('libs/train.lua')
include('libs/evaluation.lua')
include('libs/generate.lua')
include('libs/csv_to_table.lua')
include('libs/table_handling.lua')

--##################### OPTIONS SETUP ######################--

if net_type == 'mini_size' then; opt = opt_mini_size; end
if net_type == 'full_size' then; opt = opt_full_size; end

opt.batchSize = tonumber(os.getenv('BATCH_SIZE')) or opt.batchSize
opt.nz = tonumber(os.getenv('NZ')) or opt.nz
opt.display = tonumber(os.getenv('DISP')) or opt.display
opt.save_nets = tonumber(os.getenv('SAVE_NETS')) or opt.save_nets

if opt.display == 0 then; opt.display = false; else; opt.display = true; end

torch.manualSeed(79)

--#################### DATA PREPARATION #####################--
-- Convert csv to table; save classes in opt.classes
dataTable = CSV2Table(opt.data_info); opt.classes = cloneTable(dataTable[1])

-- Save classes to train, test, and valid files
Table2CSV({opt.classes}, opt.save_nets_path .. 'train.csv')
Table2CSV({opt.classes}, opt.save_nets_path .. 'test.csv')
Table2CSV({opt.classes}, opt.save_nets_path .. 'valid.csv')

-- Remove first row to only include data; Remove first col of classes, which is file path
table.remove(dataTable, 1); table.remove(opt.classes, 1)

CatLine(opt.data_info:match(".*/"), dataTable, 'all', 1)

dataTable = shuffle(dataTable)
dataP = SplitTable(dataTable, {opt.trainP, opt.testP, opt.validP})
train = dataP[1]; test = dataP[2]; valid = dataP[3]

Table2CSV(train, opt.save_nets_path .. 'train.csv', 'a')
Table2CSV(test, opt.save_nets_path .. 'test.csv', 'a')
Table2CSV(valid, opt.save_nets_path .. 'valid.csv', 'a')

print(opt)
torch.save(opt.save_nets_path .. '/opt.t7', opt)

--########### DECLARE GENERATOR & DESCRIMINATOR ##############--

if net_type == 'mini_size' then; netG, netD = mini_networks(opt)
elseif net_type == 'full_size' then; netG, netD = networks(opt); end


--######################### TRAINING #########################--

Train(netG, netD, train, opt, opt.epochs)

--######################## EVALUATION ########################--
--[[

c = nn.MSECriterion()
best_perf = 10000
e = 0
for i=1,50 do
	perf = c:forward(Simple(torch.load('epoch' .. i .. '_netD.t7'), test, opt.cs))
	print('Performance of Epoch ' .. i .. ': ' .. perf)
	if perf < best_perf then; best_perf = perf; e = i; end
end


input = torch.cat(torch.randn(9,100,1,1), torch.Tensor{
	{1,0},{3.5/2-1,0},{3/2-1,0},{2.5/2-1,0},{2/2-1,0},{1.5/2-1,0},{1/2-1,0},{0.5/2-1,0},{-1,0}
	}, 2)
image.display(image.scale(arrange(netG:forward(input),3,3), 500,500, 'simple'))

input = torch.cat(torch.randn(9,100,1,1), torch.Tensor{
	{1},{3.5/2-1},{3/2-1},{2.5/2-1},{2/2-1},{1.5/2-1},{1/2-1},{0.5/2-1},{-1}
	}, 2)
image.display(image.scale(arrange(netG:forward(input),3,3), 500,500, 'simple'))


a = torch.Tensor(9,3,4,4)
a[1] = image_generator(ideal, 0.1, 4)
a[2] = image_generator(ideal, 0.1, 3.5)
a[3] = image_generator(ideal, 0.1, 3)
a[4] = image_generator(ideal, 0.1, 2.5)
a[5] = image_generator(ideal, 0.1, 2)
a[6] = image_generator(ideal, 0.1, 1.5)
a[7] = image_generator(ideal, 0.1, 1)
a[8] = image_generator(ideal, 0.1, 0.5)
a[9] = image_generator(ideal, 0.1, 0)
image.display(image.scale(arrange(a,3,3), 500,500, 'simple'))
--]]

