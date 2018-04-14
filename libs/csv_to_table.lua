
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


function Table2CSV(tab, file_name)
  file = io.open(file_name, 'w')
  for i=1,#tab do
    local s = ''
    for j=1,#tab[i] do
      s = s .. tab[i][j] .. ','
    end
    file:write(s .. '\n')
  end
  file:close()

end
