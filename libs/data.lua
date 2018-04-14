-- package.path = package.path .. ";/home/ag/Dropbox/Uni/Master thesis/Software/cGAN/cGAN_v10/?.lua"
require 'torch'
require 'image'

require 'libs/list_files_in_dir'
require 'libs/image_normalization'
require 'libs/tif_handler'
require 'libs/table_handling'

Data = {}
Data.__index = Data

function Data:create(dataTable, batchSize, channels)
   local myData = {}
   setmetatable(myData,Data)
   myData.dataTable = dataTable
   myData.classes = #dataTable[1] - 1
   myData.dataSize = table.getn(myData.dataTable)-1
   myData.batchSize = batchSize
   myData.channels = channels
   local imDim = #load_tif(myData.dataTable[1][1], myData.channels)
   myData.batch = torch.Tensor(myData.batchSize, imDim[1], imDim[2], imDim[3])
   myData.index = 1
   return myData
end

function Data:getClasses()
   return self.classes
end

function Data:getBatchSize()
   return self.batchSize
end

function Data:getDataSize()
   return self.dataSize
end

function Data:getTotalBatches()
   return math.floor(self:getDataSize()/self:getBatchSize())
end

function Data:shuffle()
	self.dataTable = shuffle(self.dataTable)
	self.index = 1
end

function Data:getBatch()
   local class_values = torch.Tensor(self:getBatchSize(), self:getClasses())
	for i=1, self:getBatchSize() do
		self.batch[i] = load_tif(self.dataTable[self.index][1], self.channels)
      self.batch[i] = norm_minusone2one(self.batch[i])
      for j=1,self:getClasses() do
         class_values[i][j] = self.dataTable[self.index][j+1]
      end
      self.index = self.index + 1
	end
	return self.batch, class_values
end
