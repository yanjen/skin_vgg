require 'nn'

local cmd = torch.CmdLine()
cmd:option('-ratio', 0.5, 'the ratio for skinning vgg network')
cmd:option('-model', 'vgg.t7', 'The model you whsh to skin')
opt = cmd:parse(arg)

vgg = torch.load('vgg_normalised.t7')

for i=53,32,-1 do
	vgg:remove(i)
end

vgg_skin = nn.Sequential()

local PlaneSize = 3

function skin_vgg(model, ratio)
	for i=1,#model do
		local layer = model:get(i)
		if i == 1 then
			vgg_skin:add(model:get(1))
		end
		if  torch.type(layer):find('SpatialConvolution') and i ~= 1 then
			local nInputPlane, nOutputPlane = PlaneSize, layer.nOutputPlane
			local nOutputPlane = math.ceil(nOutputPlane*ratio)
			PlaneSize = nOutputPlane
			vgg_skin:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
            vgg_skin:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1))
            b_vgg = vgg:get(i).bias:storage()
            b_vgg_skin = vgg_skin:get(i).bias:storage()
            s_vgg = vgg:get(i).weight:storage()
            s_vgg_skin = vgg_skin:get(i).weight:storage()
            for j=1,nOutputPlane do
            	b_vgg_skin[j] = b_vgg[j]
            	for k=1,nInputPlane do
            		for l=1,9 do
            			s_vgg_skin[(j-1)*nInputPlane*9+(k-1)*9+l] = s_vgg[(j-1)*layer.nInputPlane*9+(k-1)*9+l]
            		end
            	end
            end
            vgg_skin:add(nn.ReLU())
        end
        if torch.type(layer):find('MaxPooling') then
            vgg_skin:add(nn.SpatialMaxPooling(2,2))
        end
	end
	for i=1,#model do
		vgg_skin:get(i).name = model:get(i).name
	end
end

skin_vgg(opt.model, opt.ratio)
torch.save('vgg_skin7.t7',vgg_skin)