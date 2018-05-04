function Join_Tables(tab1, tab2)
   for i,j in ipairs(tab2) do
      table.insert(tab1, j)
   end
end

function PopCol(tab, col)
	-- Pops column from 2D table. Returns the popped column
	local popped_col = {}
	for i=1,#tab do
		table.insert(popped_col, table.remove(tab[i], col))
	end
	return popped_col
end

function Shuffle(tab)
	-- Shuffles the first dimension of table
	local shuffledTable = cloneTable(tab)
	for i = 1,#tab do
		local rand = math.random(#tab)
		shuffledTable[i], shuffledTable[rand] = shuffledTable[rand], shuffledTable[i]
	end
	return shuffledTable
end

function cloneTable(orig)
  local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[cloneTable(orig_key)] = cloneTable(orig_value)
        end
        setmetatable(copy, cloneTable(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function SplitTable(tab, portions)
	local myTab = cloneTable(tab)

	local portionsSum = 0
	for i=1,#portions do; portionsSum = portionsSum + portions[i]; end

	local p = {}
	local dataSize = #tab
	for i = 1,#portions do
		myTab, p[i] = table.splice(myTab, 1, math.floor(portions[i]*dataSize/portionsSum))
	end
	return p
end

function Tensor2Table(tensor)
	local tab = {}
	for i=1,tensor:size(1) do
		tab[i] = {}
		for j=1,tensor:size(2) do
			tab[i][j] = tensor[i][j]
		end
	end
	return tab
end

function CSV2Table(path)
local csvFile = {}
  local file = assert(io.open(path, "r"))

   for line in file:lines() do
      cells = line:split(',')

      for i=1,#cells do
         local cell = cells[i]
         cells[i] = tonumber(cell) or cell
      end

      table.insert(csvFile, cells)
   end

   file:close()
   return csvFile
end

function string:split(sSeparator, nMax, bRegexp)
    if sSeparator == '' then
        sSeparator = ','
    end

    if nMax and nMax < 1 then
        nMax = nil
    end

    local aRecord = {}

    if self:len() > 0 then
        local bPlain = not bRegexp
        nMax = nMax or -1

        local nField, nStart = 1, 1
        local nFirst,nLast = self:find(sSeparator, nStart, bPlain)
        while nFirst and nMax ~= 0 do
            aRecord[nField] = self:sub(nStart, nFirst-1)
            nField = nField+1
            nStart = nLast+1
            nFirst,nLast = self:find(sSeparator, nStart, bPlain)
            nMax = nMax-1
        end
        aRecord[nField] = self:sub(nStart)
    end

    return aRecord
end

function CatLine(line, tab, row, col)
   if row == 'all' then
      for i=1,#tab do 
         tab[i][col] = line .. tab[i][col]
      end
   end

   return tab
end


function Table2CSV(tab, file_name, mode)
  local mode = mode or 'w'
  file = io.open(file_name, mode)
  for i=1,#tab do
    local s = ''
    for j=1,#tab[i] do
      s = s .. tab[i][j] .. ','
    end
    s = s:sub(1, -2)
    file:write(s .. '\n')
  end
  file:close()
end
