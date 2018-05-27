require 'torch'

function Remove_col(tensor, idx)
	-- Removes column of Tensor
	local indices = torch.LongTensor(tensor:size(2)-1)
	local ii = 1
	for i=1,tensor:size(2) do
		if i ~= idx then; indices[ii] = i; ii = ii + 1; end
	end
	return tensor:index(2, indices)
end

function Cat_vector(tensor, vector_value)
	local vector = torch.Tensor(tensor:size(1)):fill(vector_value)
	local tensor_and_vector = torch.cat(vector, tensor, 2)
	return tensor_and_vector
end

function merge_tensors(t1, t2)
	local length = t1:size(1)
	local width = t1:size(2)
	local t = torch.Tensor(length, width*2)
	for i=1,width do
		t:select(2, i*2-1):copy(t1:select(2,i))
		t:select(2, i*2):copy(t2:select(2,i))
	end
	return t
end

function Histogram2CSV(hist, path)
	local test_hist_tab = torch.Tensor(hist:size(1), 1)
	test_hist_tab:copy(hist)
	test_hist_tab = Tensor2Table(test_hist_tab)

	Table2CSV({{'Histogram'}}, path, 'w')
	Table2CSV(test_hist_tab, path, 'a')
end
