require 'torch'
require 'nn'
require 'image'

function generate_noise_c(noise_c_dim, number_of_classes, batch_size, class)
	local noise = torch.randn(batch_size, noise_c_dim-number_of_classes, 1, 1)
	local class = class or torch.Tensor(batch_size, number_of_classes, 1, 1):uniform(0,1)
--class[1] = 0; class[2] = 0.1; class[3] = 0.2; class[4] = 0.3; class[5] = 0.4; class[6] = 0.5;
--class[7] = 0.6; class[8] = 0.7; class[9] = 0.8; class[10] = 0.9; class[11] = 1;
--class[1] = 0.016; class[2] = 0.131; class[3] = 0.250; class[4] = 0.490; class[5] = 0.539; class[6] = 0.629;
--class[7] = 0.741; class[8] = 0.982; class[9] = 0.0; class[10] = 0.0; class[11] = 0;
	if number_of_classes == 0 then; return noise; end;
	local noise_c = torch.cat(noise, class, 2)
	return noise_c, class, noise
end

function generate(netG, nrows, ncols, number_of_classes, class)
	local nimages = nrows*ncols
	local noise_c, classes = generate_noise_c(netG:get(1).nInputPlane, number_of_classes, nimages, class)
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


function generate_sGAN(test, valid, row, col, nets, gen_path)
	local im_test, classes_test = Load_Real_Images(test, row, col)
	local im_valid, classes_valid = Load_Real_Images(valid, row, col)

	for i=1,table.getn(nets) do
		local netG = torch.load(nets[i][2])

		local im, classes = generate(netG, row, col, table.getn(opt.classes), torch.Tensor(classes_test))
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
end

function generate_GAN(test, valid, row, col, nets, gen_path)
	local im_test, classes_test = Load_Real_Images(test, row, col)
	local im_valid, classes_valid = Load_Real_Images(valid, row, col)

	for i=1,table.getn(nets) do
		local netG = torch.load(nets[i][2])

		local im, classes = generate(netG, row, col, table.getn(opt.classes), torch.Tensor(classes_test))
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
end