require 'torch'
require 'nn'
require 'image'

include('image_normalization.lua')

function factorial(n)
    if (n == 0) then
        return 1
    else
        return n * factorial(n - 1)
    end
end

function images_distance(imgs, n)
	-- Takes tensor of images as input.
	-- Outputs the pairwise mse error
	local mse = nn.MSECriterion()
	local n = n or imgs:size(1)
	local imgs_distances = torch.Tensor(factorial(n)/(2*factorial(n-2)))

	local index = 1
	for i=1,n-1 do
		for j=i+1,n do
			local distance = mse:forward(imgs[i], imgs[j])
			imgs_distances[index] = distance
			index = index + 1
		end
	end
	return imgs_distances
end


function Linear_Images_distance(imgs, ideal)
	-- Takes tensor of images as input.
	-- Outputs the pairwise mse error
	local imgs_distances = torch.Tensor(imgs:size(1), imgs:size(2), imgs:size(3), imgs:size(4))
	local imgs = imgs:clone() --norm_zero2one_multiple(imgs)

	for i=1,imgs:size(1) do
		imgs_distances[i] = imgs[i]:cdiv(ideal)
	end
	return imgs_distances
end
