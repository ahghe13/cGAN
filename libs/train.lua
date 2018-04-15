require 'torch'
require 'nn'
require 'optim'

include('data.lua')

function round(num, numDecimalPlaces)
  if numDecimalPlaces and numDecimalPlaces>0 then
    local mult = 10^numDecimalPlaces
    return math.floor(num * mult + 0.5) / mult
  end
  return math.floor(num + 0.5)
end


function Train(G, D, trainData, opt, e)

   data = Data:create(trainData, opt.batchSize, opt.cs)

   local fake_label = 0
   local real_label = 1
   local inputD = torch.Tensor(opt.batchSize, #opt.cs, opt.imDim, opt.imDim)

   local inputG = torch.Tensor(opt.batchSize, opt.nz + data:getClasses(), 1, 1)
   local noise = torch.Tensor(opt.batchSize, opt.nz, 1, 1)
   local class = torch.Tensor(opt.batchSize, data:getClasses())

   local labels = torch.Tensor(opt.batchSize, data:getClasses() + 1)
   local label = torch.Tensor(opt.batchSize, 1)

   local errD, errG

   local parametersD, gradParametersD = D:getParameters()
   local parametersG, gradParametersG = G:getParameters()
   local criterion = nn.BCECriterion()

   local epoch_tm = torch.Timer()
   local itr_tm = torch.Timer()
   local total_tm = torch.Timer()

   if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   inputD = inputD:cuda(); inputG = inputG:cuda();  noise = noise:cuda();  labels = labels:cuda(); label = label:cuda(); class = class:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(G, cudnn)
      cudnn.convert(D, cudnn)
   end
   D:cuda();           G:cuda();           criterion:cuda()
   print('GPU activated!')
   end

   local fDx = function(x)
      gradParametersD:zero()

      -- train with real
      local real, class_tmp = data:getBatch()
      class:copy(class_tmp)
      inputD:copy(real)
      label:fill(real_label)
      labels = torch.cat(label, class, 2)

      local output = D:forward(inputD)
      local errD_real = criterion:forward(output, labels)
      local df_do = criterion:backward(output, labels)
      D:backward(inputD, df_do)

      -- train with fake
      if opt.noise_type == 'gaussian' then; noise:normal()
      elseif opt.noise_type == 'uniform_zero2one' then; noise:uniform(0,1)
      elseif opt.noise_type == 'uniform_minusone2one' then; noise:uniform(-1,1); end
      class:uniform(-1,1)
      inputG = torch.cat(noise, class, 2)

      local fake = G:forward(inputG)
      inputD:copy(fake)
      label:fill(fake_label)
      labels = torch.cat(label, class, 2)

      local output = D:forward(inputD)
      local errD_fake = criterion:forward(output, labels)
      local df_do = criterion:backward(output, labels)
      D:backward(inputD, df_do)

      errD = errD_real + errD_fake

      return errD, gradParametersD
   end

   -- create closure to evaluate f(X) and df/dX of generator
   local fGx = function(x)
      gradParametersG:zero()

      --[[ the three lines below were already executed in fDx, so save computation
      noise:uniform(-1, 1) -- regenerate random noise
      local fake = netG:forward(noise)
      input:copy(fake) ]]--
      label:fill(real_label) -- fake labels are real for generator cost
      labels = torch.cat(label, class, 2)

      local output = D.output -- netD:forward(input) was already executed in fDx, so save computation
      errG = criterion:forward(output, labels)
      local df_do = criterion:backward(output, labels)
      local df_dg = D:updateGradInput(inputD, df_do)

      G:backward(inputG, df_dg)
      return errG, gradParametersG
   end

   if epoch == nil then; epoch = 1; end

   local totalBatches = data:getTotalBatches()
   total_tm:reset()

   for i=1,e do
      epoch_tm:reset()
      if opt.display == true then; print("Epoch " .. epoch); end

      for j=1,totalBatches do
         itr_tm:reset()
         optim.adam(fDx, parametersD, optimStateD)
         optim.adam(fGx, parametersG, optimStateG)
         print("Epoch " .. epoch .. " (" .. j .. "/" .. totalBatches .. ")", 
            "Itr time: " .. round(itr_tm:time().real, 4),
            "Total time: " .. round(total_tm:time().real, 4))
      end

      data:shuffle()

      if opt.display == true then
         print("Epoch " .. epoch .. " successfully completed!")
         print("Descriminator Error: " .. errD, "Generator Error: " .. errG)
         print("Epoch time: " .. epoch_tm:time().real .. "\n")
      end

      if epoch % opt.save_nets == 0 then
         torch.save(opt.save_nets_path .. 'epoch' .. epoch .. '_netG.t7', netG:clearState())
         torch.save(opt.save_nets_path .. 'epoch' .. epoch .. '_netD.t7', netD:clearState())
      end


      epoch = epoch+1
   end

   if opt.display == true then; print(total_tm:time().real); end

end

