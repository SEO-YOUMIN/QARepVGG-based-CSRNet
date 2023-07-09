import h5py
import scipy.io as sio
import PIL.Image as Image
import numpy as np
import os
import glob
import torchvision.transforms.functional as F
from image import *
# from model import CSRNet
from repvgg_csrnet import Repvgg_CSRNet
import torch
import time
from collections import OrderedDict
import math


from torchvision import datasets, transforms
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

from torchsummaryX import summary
####### estination & groundtruth store #######
f = open("47test.json",'w')

##############################################

data_path='/workspace/dataset/part_A_final/test_data/images/'
img_paths=glob.glob(os.path.join(data_path, '*.jpg'))

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

#   1. Load trained RepVGG CSRNet model.
# model = CSRNet(load_weights=True)
model = Repvgg_CSRNet(backbone_name='RepVGG-A0', backbone_file=None, deploy=False, pretrained=False)

if use_cuda:
    model = model.cuda()
    checkpoint = torch.load('/workspace/RepVGG/RepVGG/code_6/47model_best.pth.tar')
else:
    checkpoint = torch.load('/workspace/RepVGG/RepVGG/code_6/47model_best.pth.tar', map_location='cpu')
# model.load_state_dict(checkpoint['state_dict'])
##
state_dict = checkpoint['state_dict']
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if 'module' in k:
        # print(k)
        k = k.replace('module.', '')
    new_state_dict[k]=v

# print('new!!!')
# for k, v in new_state_dict.items():
#    print(k)
# print('end!!')

model.load_state_dict(new_state_dict)
##
#   2. Convert and check the equivalence
input_ex = torch.rand(1, 3, 784, 1024)
input_ex_cuda = input_ex.cuda()
model.eval()
# print(model)
y_train = model(input_ex_cuda)
for module in model.modules():
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()
y_deploy = model(input_ex_cuda)
print('before save, train output size is ', y_train.size())
print('before save, inference output size is ', y_deploy.size())
print('before save, train result is ', y_train.detach().cpu().sum().numpy())
print('before save, inference result is ', y_deploy.detach().cpu().sum().numpy())
print('=================== The diff is')
print(((y_deploy.cpu() - y_train.cpu()) ** 2).detach().cpu().sum().numpy())
f.write("before save, train output size is {}\n\n".format((y_train.size())))
f.write("before save, inference output size is {}\n\n".format((y_deploy.size())))
f.write("before save, train result is {}\n\n".format(y_train.detach().cpu().sum().numpy()))
f.write("before save, inference result is {}\n\n".format(y_deploy.detach().cpu().sum().numpy()))
f.write("before save, The diff is {}\n\n".format((((y_deploy.cpu() - y_train.cpu()) ** 2).detach().cpu().sum().numpy())))

#   3. Save the converted model
torch.save(model.state_dict(), '47CSRNet-RepVGG-deploy.pth')
del model

#   4. For inference, load the saved model. There is no need to load the ImageNet-pretrained weights again.
deploy_model = Repvgg_CSRNet(backbone_name='RepVGG-A0', backbone_file=None, deploy=True, pretrained=False)
deploy_model.eval()
deploy_model.load_state_dict(torch.load('47CSRNet-RepVGG-deploy.pth'))

# print(deploy_model)

#   5. Check again or do whatever you want
y_deploy = deploy_model(input_ex)
# print('y_deploy is ', y_deploy.is_cuda)
# print('y_train is ', y_train.is_cuda)
print('after save and load, train output size is ', y_train.size())
print('after save and load, inference output size is ', y_deploy.size())
print('after save and load, train result is ', y_train.detach().cpu().sum().numpy())
print('after save and load, inference result is ', y_deploy.detach().cpu().sum().numpy())
print('==================== The diff is')
print(((y_deploy.cpu() - y_train.cpu()) ** 2).detach().cpu().sum().numpy())
f.write("after save and load, train output size is {}\n\n".format((y_train.size())))
f.write("after save and load, inference output size is {}\n\n".format((y_deploy.size())))
f.write("after save and load, train result is {}\n\n".format(y_train.detach().cpu().sum().numpy()))
f.write("after save and load, inference result is {}\n\n".format(y_deploy.detach().cpu().sum().numpy()))
f.write("after save and load, The diff is {}\n\n".format((((y_deploy.cpu() - y_train.cpu()) ** 2).detach().cpu().sum().numpy())))

deploy_model = deploy_model.cuda()

mae = 0
mse = 0
for i in range(len(img_paths)):
    t1 = time.time()

    if use_cuda:
        img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    else:
        img = transform(Image.open(img_paths[i]).convert('RGB'))

    file_name = img_paths[i].replace('.jpg','.mat').replace('images','ground_truth')
    gt_file_name = os.path.join(os.path.dirname(file_name), 'GT_' + os.path.basename(file_name))

    gt_file = sio.loadmat(gt_file_name)
    groundtruth = gt_file['image_info'][0][0][0][0][1][0][0]

    ###################### to check model flops ########################
    # rand_x = torch.randn(1,3,666,1024)  #change resolution (1,3,w,h)
    # rand_x = rand_x.cuda()
    # summary(deploy_model, rand_x)
    ####################################################################

    ##################### to check model parameter #####################
    #model_param = filter(lambda p: p.requires_grad, model.parameters())
    #params = sum([np.prod(p.size()) for p in model_param])
    #print('model total param number :', params)
    ####################################################################
    output = deploy_model(img.unsqueeze(0))

    ##################### to check data shape ##########################
    #print("img.unsqueeze : ",img.unsqueeze(0).shape)
    #print("gt : ",groundtruth)
    #print("output : ",torch.sum(output))
    ####################################################################


    t2 = time.time()
    print('time: {}'.format(t2-t1))
    print('{} -- {} : {}'.format(img_paths[i], output.detach().cpu().sum().numpy(), groundtruth))
    ###### estination & groundtruth store ######
    f.write("{} \t image: {}\t  estination: {}\t  groundtruth: {}\n\n".format(i, img_paths[i], output.detach().cpu().sum().numpy(), groundtruth))
    ############################################
    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    mse += (abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth)))**2 
    print(i,mae, mse)
print('Test End!')
###################### to check model flops ########################
rand_x = torch.randn(1,3,480,640)  #change resolution (1,3,w,h)
rand_x = rand_x.cuda()
summary(deploy_model, rand_x)
#summary(deploy_model, input_size=(3, 784, 1024))
####################################################################
print("final mae: {},  final mse: {}".format((mae/len(img_paths)), math.sqrt(mse/len(img_paths))))
f.write("Final MAE: {},  Final MSE: {}".format((mae/len(img_paths)), math.sqrt(mse/len(img_paths))))
f.close()
