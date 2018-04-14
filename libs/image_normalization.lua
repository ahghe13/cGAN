require 'torch'
require 'nn'
require 'image'

function norm_zero2one(img)
	im = img:clone()
	im:add(-torch.min(im)):div(torch.max(im))
	return im
end

function norm_minusone2one(img)
	im = img:clone()
	im:add(-torch.min(im)):div(torch.max(im)/2):add(-1)
	return im
end
