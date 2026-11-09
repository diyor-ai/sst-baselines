"""B2 whole-region models: small U-Net and 2-layer ConvLSTM. Both map 30 days of (anomaly, sin, cos) fields to 3 output fields (48x48)."""
import torch
from torch import nn

GROUPS = 8


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(GROUPS, cout), nn.ReLU(inplace=False),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(GROUPS, cout), nn.ReLU(inplace=False))


class UNet(nn.Module):
    """48 -> 24 -> 12 -> 6; channels C, 2C, 4C, 8C. Input (B, 30, 3, 48, 48) is stacked to 90 channels."""

    def __init__(self, c=16, n_in=90, n_out=3):
        super().__init__()
        self.e1, self.e2, self.e3, self.mid = block(n_in, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        self.pool = nn.MaxPool2d(2)
        self.u3, self.u2, self.u1 = (nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), nn.ConvTranspose2d(4 * c, 2 * c, 2, 2),
                                     nn.ConvTranspose2d(2 * c, c, 2, 2))
        self.d3, self.d2, self.d1 = block(8 * c, 4 * c), block(4 * c, 2 * c), block(2 * c, c)
        self.head = nn.Conv2d(c, n_out, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


class ConvLSTMLayer(nn.Module):
    """Standard ConvLSTM: gates = conv_x(x_t) + conv_h(h_{t-1}) (= one 3x3 conv on the concatenation); input conv done for all steps at once."""

    def __init__(self, cin, hid, k=3):
        super().__init__()
        self.hid = hid
        self.cx = nn.Conv2d(cin, 4 * hid, k, padding=k // 2)
        self.ch = nn.Conv2d(hid, 4 * hid, k, padding=k // 2, bias=False)
        with torch.no_grad():
            self.cx.bias.zero_()
            self.cx.bias[hid:2 * hid] = 1.0                  # forget-gate bias 1 (gate order i, f, g, o)

    def forward(self, x):                                     # x: (B, T, cin, H, W) -> (B, T, hid, H, W)
        B, T, _, H, W = x.shape
        gx = self.cx(x.reshape(B * T, *x.shape[2:])).reshape(B, T, 4 * self.hid, H, W)
        h = x.new_zeros(B, self.hid, H, W)
        c = x.new_zeros(B, self.hid, H, W)
        out = []
        for t in range(T):
            i, f, g, o = (gx[:, t] + self.ch(h)).chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            out.append(h)
        return torch.stack(out, dim=1)


class ConvLSTM(nn.Module):
    def __init__(self, c=16, n_in=3, n_out=3):
        super().__init__()
        self.l1, self.l2 = ConvLSTMLayer(n_in, c), ConvLSTMLayer(c, c)
        self.head = nn.Conv2d(c, n_out, 1)

    def forward(self, x):                                     # x: (B, 30, 3, 48, 48)
        return self.head(self.l2(self.l1(x))[:, -1])          # last hidden state of layer 2


class ConvLSTMED(nn.Module):
    """Encoder-forecaster ConvLSTM (compute-reduced revision): every 3rd input day (t-27..t, 10 steps) -> strided conv 48->24 (shared over steps)
    -> 2-layer ConvLSTM (C, 3x3) at 24x24 -> last hidden -> bilinear x2 + conv + GN + ReLU -> 1x1 conv -> 3 channels at 48x48."""
    STEP = 3

    def __init__(self, c=16, n_in=3, n_out=3):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(n_in, c, 3, stride=2, padding=1), nn.GroupNorm(GROUPS, c), nn.ReLU())
        self.l1, self.l2 = ConvLSTMLayer(c, c), ConvLSTMLayer(c, c)
        self.dec = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(GROUPS, c), nn.ReLU())
        self.head = nn.Conv2d(c, n_out, 1)

    def forward(self, x):                                     # x: (B, 30, 3, 48, 48)
        x = x[:, self.STEP - 1::self.STEP]                    # days t-27, t-24, ..., t  (index 29 is day t)
        B, T = x.shape[:2]
        z = self.enc(x.reshape(B * T, *x.shape[2:]))
        z = z.reshape(B, T, *z.shape[1:])
        h = self.l2(self.l1(z))[:, -1]
        h = nn.functional.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(self.dec(h))


def build(model, c, **kw):
    return {"unet": UNet, "convlstm": ConvLSTM, "convlstm_ed": ConvLSTMED}[model](c, **kw)
