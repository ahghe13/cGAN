-- package.path = package.path .. ";/home/ag/Dropbox/Uni/Master thesis/Master thesis/Software/cGAN/cGAN_v10/?.lua"

require 'torch'
require 'image'

require 'libs/image_normalization'
require 'libs/tif_handler'

	ideal = torch.Tensor(3,4,4)

	A = torch.Tensor{
		{1,0.75,0.25,0},
		{0.75,0.25,0,0},
		{0.25,0,0,0},
		{0,0,0,0}
	}
	B = torch.Tensor{
		{0,0.25,0.75,1},
		{0,0,0.25,0.75},
		{0,0,0,0.25},
		{0,0,0,0}
	}
	C = torch.Tensor{
		{0,0,0,0},
		{0.25,0,0,0},
		{0.75,0.25,0,0},
		{1,0.75,0.25,0}
	}
	D = torch.Tensor{
		{0,0,0,0},
		{0,0,0,0.25},
		{0,0,0.25,0.75},
		{0,0.25,0.75,0.5}
	}

	ideal[1] = B + D/2
	ideal[2] = A + D/2
	ideal[3] = C

function image_generator(ideal, std_dev, brightness)
	local output = torch.Tensor(ideal:size(1),ideal:size(2),ideal:size(3))
	for c=1,ideal:size(1) do
		for i=1,ideal:size(2) do
			for j=1,ideal:size(3) do
				if (i > 1 and i < 4 and j > 1 and j < 4) then
					output[c][i][j] = brightness * ideal[c][i][j]
				else
					output[c][i][j] = torch.normal(ideal[c][i][j], std_dev)
				end
			end
		end
	end
	return output:clone()
end

function generate_images(path, amount, std_dev)
	file = io.open(path .. 'data_info.csv', 'w')
	file:write('file,brightness\n')
	for i=1,amount do
		n = torch.uniform(0,4)
		img = image_generator(ideal, std_dev, n)
	--	image.display(image.scale(img, 250,250, 'simple'))
		save_tif(path .. i .. '.tif', img)
		file:write(i .. '.tif,' .. n .. "\n")
	end
	file:close()
end

