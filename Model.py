import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm(nn.Module):
    '''
    Layer normalization
    '''

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        h = x.size(2)
        x = rearrange(x, 'b c h w -> b (h w) c', h=h)
        y = self.fn(self.norm(x), **kwargs)
        y = rearrange(y, 'b (h w) c -> b c h w', h=h)
        return y


class MHCA(nn.Module):
    '''
    The multi-head channel attention (MHCA)
    '''

    def __init__(self, dim=128, heads=4, dropout=0.):
        super().__init__()
        assert dim % heads == 0
        inner_dim = dim

        self.heads = heads
        self.scale = nn.Parameter(torch.ones(heads, 1, 1))

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = torch.nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q.transpose(-1, -2), k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(v, attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MLP(nn.Module):
    '''
    The Multilayer perceptron (MLP)
    '''

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class Transformer_Encoder(nn.Module):
    '''
    The Transformer Encoder for capturing long-range interdependencies between channels
    '''

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(1, depth + 1):
            self.layers.append(nn.ModuleList([
                LayerNorm(dim, MHCA(dim, heads=heads, dropout=dropout)),
                MLP(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x)
        return x


class DownSample(nn.Module):
    '''
    The 2*2 average pooling for downsampling
    '''

    def __init__(self):
        super().__init__()
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    '''
    The 2*2 transposed convolution for upsampling
    '''

    def __init__(self, dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim, dim, kernel_size=2, padding=0, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class Uformer(nn.Module):
    '''
    The U-shape Transformer for nonlinear feature representation to learn more accurate coefficients
    '''

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.level_1 = Transformer_Encoder(dim, depth, heads, mlp_dim, dropout)
        self.level_2 = Transformer_Encoder(dim, depth, heads, mlp_dim, dropout)
        self.level_3 = Transformer_Encoder(dim, depth, heads, mlp_dim, dropout)
        self.down_2 = DownSample()
        self.down_1 = DownSample()
        self.up_1 = UpSample(dim)
        self.up_2 = UpSample(dim)

    def interpolation_x_y(self, x, y):
        ## The interpolation is applied if the spatial sizes are unmatched
        if (x.shape[2] != y.shape[2]) or (x.shape[3] != y.shape[3]):
            y = F.interpolate(y, size=(x.shape[2], x.shape[3]), mode='bicubic', align_corners=False)
        ## residual connection
        return x + y

    def forward(self, x):
        # level 1
        x1 = self.level_1(x)
        # level 2
        y1 = self.down_1(x)
        x2 = self.level_2(y1)
        # level 3
        y2 = self.down_2(y1)
        x3 = self.level_3(y2)
        # level 2 (up)
        z = self.up_1(x3)
        z = self.interpolation_x_y(x2, z)
        # level 1 (up)
        z = self.up_2(z)
        z = self.interpolation_x_y(x1, z)
        # global residual connection
        return z + x


class MSM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.con1x1 = nn.Conv2d(dim, 1, 1, 1)
        self.con3x3 = nn.Conv2d(dim, 1, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, X):
        X1 = self.con1x1(X)
        X2 = self.con3x3(X)

        return X2 + X1


class SRN(torch.nn.Module):
    '''
    The proposed deep subspace representation network separates the target and background components and unmixes them into endmembers and abundances.
    '''

    def __init__(self, c, k1, k2=1, dim=128, heads=4):
        super().__init__()
        self.k1 = k1  # The number of atoms in background subspace
        self.k2 = k2  # The number of atoms in target subspace (Usually 1)
        self.firstConv = nn.Sequential(
            nn.Conv2d(c, dim, 1, 1),
            nn.LeakyReLU()
        )  # The first convolutional layer to extract shallow features
        self.uformer = Uformer(dim, 1, heads, dim // 2, dropout=0.)
        # Uformer is used to enhance nonlinear feature representation and extract more accurate abundances
        self.adjust = nn.Sequential(
            nn.Conv2d(dim, self.k1 + 1, 1, 1)
        )
        # Transforming the enhanced features into the desired abundances
        self.Ab = nn.Parameter(torch.randn(c, self.k1, 1, 1))
        # The adaptively learnable background subspace Ab
        self.msm = MSM(c)
        # The multi-scale mapping transforms the target components into the final detection map

    def forward(self, X, t_atom):
        ## The learning of abundances
        Y = self.firstConv(X)
        S = self.uformer(Y)
        S = self.adjust(S)

        ## Here we keep the background subspace Ab non-negative using the ReLU activation
        Ab = torch.relu(self.Ab)
        ## The softmax is applied to realize the non-negative and sum-to-one constraints of abundanceas
        S = torch.softmax(S, dim=1)
        ## The joint subspaces (endmembers)
        A = torch.cat([Ab, t_atom], dim=1)
        ## Reconstruct the HSI using the endmembers and abundances
        out = F.conv2d(S, A)

        ## Separate the background and target abundances
        Sb = S[:, :self.k1, :, :]
        St = S[:, self.k1:, :, :]

        ## Synthesize the target components using the target endmember and abundances
        target = F.conv2d(St, t_atom)

        ##map the target component into the detection map
        detection_map = self.msm(target)
        detection_map = torch.squeeze(detection_map, dim=1)

        return out, detection_map, S
