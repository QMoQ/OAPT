import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from oapt.archs import rnan_common as common
import torch.nn.functional as F
import torch.nn as nn
# import rnan_common as common

def check_image_size(x, pad_size):
    b, c, h, w = x.shape
    mod_pad_h = (pad_size - h % pad_size) % pad_size
    mod_pad_w = (pad_size - w % pad_size) % pad_size

    try:
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
    except BaseException:
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant")

    return x

### RNAN
### residual attention + downscale upscale + denoising
class _ResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(_ResGroup, self).__init__()
        modules_body = []
        modules_body.append(common.ResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res

### nonlocal residual attention + downscale upscale + denoising
class _NLResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(_NLResGroup, self).__init__()
        modules_body = []
        modules_body.append(common.NLResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res


@ARCH_REGISTRY.register()
class RNAN(nn.Module):
    def __init__(self, n_colors, n_resgroup, n_resblock, n_feats, reduction, res_scale, conv=common.default_conv):
        super(RNAN, self).__init__()
        
        kernel_size = 3
        scale = 1
        act = nn.ReLU(True)
        
        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]
        
        # define body module
        modules_body_nl_low = [
            _NLResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale)]
        modules_body = [
            _ResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale) \
            for _ in range(n_resgroup - 2)]
        modules_body_nl_high = [
            _NLResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            conv(n_feats, n_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body_nl_low = nn.Sequential(*modules_body_nl_low)
        self.body = nn.Sequential(*modules_body)
        self.body_nl_high = nn.Sequential(*modules_body_nl_high)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        b, c, h, w = x.shape
        x = check_image_size(x, 2)

        feats_shallow = self.head(x)

        res = self.body_nl_low(feats_shallow)
        res = self.body(res)
        res = self.body_nl_high(res)


        res_main = self.tail(res)

        res_clean = x + res_main

        res_clean = res_clean[:,:,:h,:w]
        return res_clean 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

if __name__=="__main__":
    model = RNAN(n_colors=1,
                n_resgroup=10,
                n_resblock=16,
                n_feats=64,
                reduction=16,
                res_scale=1)
    height = 160
    width = 160
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model,(1, height, width),as_strings=True,print_per_layer_stat=False)
    print('MACs:  ' + macs)
    print('Params: ' + params)