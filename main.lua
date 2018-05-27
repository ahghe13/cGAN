require 'torch'
require 'nn'
require 'image'

include('options.lua')
include('libs/networks.lua')
include('libs/train.lua')
include('libs/table_handling.lua')

--##################### OPTIONS SETUP ######################--

if net_size == 'mini_size' then; opt = opt_mini_size; end
if net_size == 'full_size' then; opt = opt_full_size; end

--opt.batchSize = tonumber(os.getenv('BATCH_SIZE')) or opt.batchSize
--opt.nz = tonumber(os.getenv('NZ')) or opt.nz
--opt.display = tonumber(os.getenv('DISP')) or opt.display
--opt.save_nets = tonumber(os.getenv('SAVE_NETS')) or opt.save_nets
opt.save_nets_path = arg[1] or opt.save_nets_path
opt.cs = arg[2] or opt.cs

if opt.display == 0 then; opt.display = false; else; opt.display = true; end

torch.manualSeed(79)

--#################### DATA PREPARATION #####################--
-- Convert csv to table; save classes in opt.classes
dataTable = CSV2Table(opt.data_info); opt.classes = cloneTable(dataTable[1])

-- Save classes to train, test, and valid files
Table2CSV({opt.classes}, opt.save_nets_path .. '/train.csv')
Table2CSV({opt.classes}, opt.save_nets_path .. '/test.csv')
Table2CSV({opt.classes}, opt.save_nets_path .. '/valid.csv')

-- Remove first row to only include data; Remove first col of classes, which is file path
table.remove(dataTable, 1); table.remove(opt.classes, 1)

CatLine(opt.data_info:match(".*/"), dataTable, 'all', 1)

dataTable = Shuffle(dataTable)
dataP = SplitTable(dataTable, {opt.trainP, opt.testP, opt.validP})
train = dataP[1]; test = dataP[2]; valid = dataP[3]

Table2CSV(train, opt.save_nets_path .. '/train.csv', 'a')
Table2CSV(test, opt.save_nets_path .. '/test.csv', 'a')
Table2CSV(valid, opt.save_nets_path .. '/valid.csv', 'a')

if opt.cs[1] == 'all' then; local im; im, opt.cs = load_tif(train[1][1]); end

print(opt)
torch.save(opt.save_nets_path .. '/opt.t7', opt)

--########### DECLARE GENERATOR & DESCRIMINATOR ##############--

if net_size == 'mini_size' then; netG, netD = mini_networks(opt)
elseif net_size == 'full_size' then; netG, netD = networks(opt); end


--######################### TRAINING #########################--

Train(netG, netD, train, opt, opt.epochs)

--######################## EVALUATION ########################--

--arg = {opt.save_nets_path}; dofile('evaluation.lua');

--######################## CONVERSION ########################--

--[[ Converts nets if cuda was used
if opt.gpu > 0 then
	nonCuda_path = opt.save_nets_path .. '/nonCuda/'
	paths.mkdir(nonCuda_path)
	arg = {'fromCuda', opt.save_nets_path, nonCuda_path}
	dofile('convert.lua')
end
]]--

--arg = {opt.save_nets_path}; dofile('evaluation.lua')
