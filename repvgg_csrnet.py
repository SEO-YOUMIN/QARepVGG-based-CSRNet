import torch
from torch import nn
import torch.nn.functional as F
from repvgg import get_RepVGG_func_by_name
from utils import save_net, load_net


#   The PSPNet parts are from
#   https://github.com/hszhao/semseg
class Repvgg_CSRNet(nn.Module):
    def __init__(self, backbone_name, backbone_file, deploy, pretrained=True):
        super(Repvgg_CSRNet, self).__init__()
        self.seen = 0
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3

        layer3_last_channel = 0
        # stride 2 -> 1
        for n, m in self.layer3.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                # m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                m.stride = (1, 1)
                # print('change dilation, padding, stride of ', n)
                layer3_last_channel = m.out_channels
            elif 'rbr_1x1' in n and isinstance(m, nn.Conv2d):
                m.stride = (1, 1)
                # print('change stride of ', n)
        
        self.layer3_last = layer3_last_channel
        
        self.backend = make_layers([192, 96, 48, 48], in_channels=self.layer3_last, dilation=True)
        self.output_layer = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x, y=None):
        # frontend
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # backend
        x = self.backend(x)
        x = self.output_layer(x)

        return x

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



if  __name__ == '__main__':
    #   1.  Build the PSPNet with RepVGG backbone. Download the ImageNet-pretrained weight file and load it.
    model = Repvgg_CSRNet(backbone_name='RepVGG-A0', backbone_file='RepVGG-A0-train.pth', deploy=False, pretrained=True)

    #   2.  Train it
    # seg_train(model)

    #   3.  Convert and check the equivalence
    input = torch.rand(4, 3, 713, 713)
    model.eval()
    print(model)
    y_train = model(input)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    y_deploy = model(input)
    print('output is ', y_deploy.size())
    print('=================== The diff is')
    print(((y_deploy - y_train) ** 2).sum())

    #   4.  Save the converted model
    torch.save(model.state_dict(), 'CSRNet-RepVGG-A0-deploy.pth')
    del model   #   Or do whatever you want with it

    #   5.  For inference, load the saved model. There is no need to load the ImageNet-pretrained weights again.
    deploy_model = Repvgg_CSRNet(backbone_name='RepVGG-A0', backbone_file=None, deploy=True, pretrained=False)
    deploy_model.eval()
    deploy_model.load_state_dict(torch.load('CSRNet-RepVGG-A0-deploy.pth'))

    print(deploy_model)

    #   6.  Check again or do whatever you want
    y_deploy = deploy_model(input)
    print('=================== The diff is')
    print(((y_deploy - y_train) ** 2).sum())
