require 'image'

function List_Files_in_Dirs(dirs_path, ext)
	local files = {}
	for i=1,#dirs_path do
		Merge_Tables(files, List_Files_in_Dir(dirs_path[i], ext))
	end
	return files

end

function List_Files_in_Dir(dir_path, ext)
	local files = {}
	for file in paths.files(dir_path) do
	   if file:find(ext .. '$') then
	      table.insert(files, paths.concat(dir_path, file))
	   end
	end
	return files
end

function Merge_Tables(t1, t2)
   for i,j in ipairs(t2) do
      table.insert(t1, j)
   end
end