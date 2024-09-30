import torch
import torch.nn as nn


class SFB(nn.Module):
    def __init__(self, dim=180, out = 180, kernel=3):
        super().__init__()
        self.res_branch1 = nn.Sequential(
                        nn.Conv2d(dim, dim, kernel, 1, kernel//2),
                        nn.LeakyReLU(),
                        nn.Conv2d(dim, dim, kernel, 1, kernel//2)
                        )
        self.res_branch2_head = nn.Sequential(
                        nn.Conv2d(dim, dim, kernel, 1, kernel//2),
                        nn.LeakyReLU()
                        )
        self.res_branch2_body = nn.Sequential(
                        nn.Conv2d(2*dim, 2*dim, kernel, 1, kernel//2),
                        nn.LeakyReLU()
                        )
        self.res_branch2_tail = nn.Conv2d(dim, dim, 1, 1)
        self.conv = nn.Conv2d(2*dim, out, 1, 1)
    def forward(self,x):
        _, _, H, W = x.shape
        x1 = x + self.res_branch1(x)
        x2 = self.res_branch2_head(x)
        y = torch.fft.rfft2(x2)
        y_imag = y.imag
        y_real = y.real
        y = torch.cat([y_real, y_imag], dim=1)
        y = self.res_branch2_body(y)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W))
        x2 = y + x2
        x2 = self.res_branch2_tail(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


if __name__ == "__main__":
    x = torch.zeros((1,16,40,40))
    blk = SFB(16)
    x1 = blk(x)
    print(x1.shape)