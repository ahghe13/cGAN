require 'torch'
require 'nn'
require 'image'

function norm_zero2one(img)
	local im = img:clone()
	im:add(-torch.min(im)):div(torch.max(im))
	return im
end

function norm_minusone2one(img)
	local im = img:clone()
	im:add(-torch.min(im)):div(torch.max(im)/2):add(-1)
	return im
end

function norm_zero2one_beta(img)
	local im = torch.Tensor(img:size())

	if img:size():size() > 3 then
		for i=1,img:size(1) do
			im[i]:copy(norm_zero2one(img[i]))
		end
	else
		im = norm_zero2one(img)
	end

	return im
end

function norm_minusone2one_beta(img)
	local im = torch.Tensor(img:size())

	if img:size():size() > 3 then
		for i=1,img:size(1) do
			im[i]:copy(norm_minusone2one(img[i]))
		end
	else
		im = norm_minusone2one(img)
	end

	return im
end