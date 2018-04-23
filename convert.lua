require 'torch'
require 'nn'
require 'cunn'

include('libs/list_files_in_dir.lua')
include('libs/paths_handling.lua')

convert = 'fromCuda' -- choose between 'toCuda' or 'fromCuda'
path = '/home/ahghe13/Networks'
output_path = '/home/ahghe13/Networks_nonCuda'

-- Modify opt
opt = torch.load(path .. '/opt.t7')
if convert == 'toCuda' then; opt.gpu = 1; end
if convert == 'fromCuda' then; opt.gpu = 0; end
torch.save(output_path .. '/opt.t7', opt)
print('opt converted. \n')

-- Copy train, test, and valid
os.execute('cp ' .. path .. '/train.csv ' .. output_path .. '/train.csv')
os.execute('cp ' .. path .. '/test.csv ' .. output_path .. '/test.csv')
os.execute('cp ' .. path .. '/valid.csv ' .. output_path .. '/valid.csv')
print('train, test, and valid has been copied. \n')

-- Convert the nets
print('Converting networks:')
nets_paths = List_Files_in_Dir(nets_dir_path, '.t7')
Exclude_paths(nets_paths, 'epoch')

nets_total = table.getn(nets_paths)

for i=1,nets_total do
	net = torch.load(nets_paths[i])
	if convert == 'toCuda' then; net:cuda(); end
	if convert == 'fromCuda' then; net:double(); end
	torch.save(output_path .. '/' .. File_name(nets_paths[i]), net)
	print(i .. ' out of ' .. nets_total .. ' networks converted.')
end
