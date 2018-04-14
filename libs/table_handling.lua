function PopCol(tab, col)
	-- Pops column from 2D table. Returns the popped column
	local c = {}
	for i=1,#tab do
		table.insert(c, table.remove(tab[i], col))
	end
	return c
end

function shuffle(tab)
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



--[[
a = {{1,2,5,1,2},
	 {2,4,1,5,3},
	 {1,2,4,2,6},
	 {1,2,1,2,3},}
]]--