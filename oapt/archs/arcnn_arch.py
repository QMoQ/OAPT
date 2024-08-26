import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class ARCNN(nn.Module):
    def __init__(self, inch=1, ouch=1):
        super(ARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(inch, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.Conv2d(16, ouch, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x











if __name__=='__main__':
    model = ARCNN(inch=1,ouch=1)
    height = 160
    width = 160
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model,(1, height, width),as_strings=True,print_per_layer_stat=False)
    print('MACs:  ' + macs)
    print('Params: ' + params)
