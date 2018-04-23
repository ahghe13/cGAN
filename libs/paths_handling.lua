function File_name(path) -- extracts filename from full path
	return path:sub(path:match('.*()/')+1)
end

function Exclude_paths(paths, key)
	-- Excludes paths to files whose name does not include key
	bad_paths = {}
	for i=1, table.getn(paths) do
		if File_name(paths[i]):match(key) == nil then
			table.insert(bad_paths, i)
		end
	end

	local l = table.getn(bad_paths)
	for i=1,l do
		table.remove(paths, bad_paths[l-i+1])
	end
end

function get_epoch(path)
	local file = File_name(path)
	local epoch = file:match('h.*_')
	epoch = epoch:sub(2, epoch:len()-1)
	return tonumber(epoch)
end

function get_net_type(path)
	local file = File_name(path)
	local net_type = file:match('_.*')
	net_type = net_type:sub(2, net_type:len()-3)
	return net_type
end