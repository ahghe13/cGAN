require 'torch'
require 'nn'
require 'image'

function generate_noise_c(noise_c_dim, number_of_classes, batch_size)
	local noise = torch.randn(batch_size, noise_c_dim-number_of_classes, 1, 1)
	local class = torch.Tensor(batch_size, number_of_classes, 1, 1):uniform(0,1)
--class[1] = 0; class[2] = 0.1; class[3] = 0.2; class[4] = 0.3; class[5] = 0.4; class[6] = 0.5;
--class[7] = 0.6; class[8] = 0.7; class[9] = 0.8; class[10] = 0.9; class[11] = 1;
class[1] = 0.016; class[2] = 0.131; class[3] = 0.250; class[4] = 0.490; class[5] = 0.539; class[6] = 0.629;
class[7] = 0.741; class[8] = 0.982; class[9] = 0.0; class[10] = 0.0; class[11] = 0;
	if number_of_classes == 0 then; return noise; end;
	local noise_c = torch.cat(noise, class, 2)
	return noise_c, class, noise
end

function generate(netG, nrows, ncols, number_of_classes)
	local nimages = nrows*ncols
	local noise_c, classes = generate_noise_c(netG:get(1).nInputPlane, number_of_classes, nimages)
	if netG:get(1).weight:type() == 'torch.CudaTensor' then; noise_c = noise_c:cuda(); end
	local imgs = netG:forward(noise_c)
	return arrange(imgs, nrows, ncols), classes
end

function arrange(images, r, c, counter)
	if counter == nil then; counter = 1; end

	if r == 1 and c == 1 then; return images[counter]; end
	
	if r > 1 then
		return torch.cat(arrange(images, r-1, c, counter+c), arrange(images, 1, c, counter), 2)
	end

	if c > 1 then
		return torch.cat(arrange(images, r, c-1, counter+1), images[counter], 3)
	end

end
