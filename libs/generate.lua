require 'torch'
require 'nn'
require 'image'

function generate(netG, nrows, ncols, number_of_classes)
	nimages = nrows*ncols
	noise_dim = netG:get(1).nInputPlane
	noise = torch.randn(nimages, noise_dim, 1, 1)
	for i=1,number_of_classes do; noise:select(2, noise_dim-i+1):uniform(0,1); end

	imgs = netG:forward(noise)
	return arrange(imgs, nrows, ncols)
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
