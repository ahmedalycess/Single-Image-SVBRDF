import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataset import MaterialDataset


class InstanceNormWithStats(nn.Module):
    """
    Instance normalization that also returns per-channel means.
    """
    def __init__(self, num_features, eps=1e-5):
        super(InstanceNormWithStats, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.randn(num_features))
        self.offset = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        mean = x.mean(dim=(2, 3), keepdim=True)         # [B, C, 1, 1]
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm * self.scale.view(1, -1, 1, 1) + self.offset.view(1, -1, 1, 1)
        
        # Extract channel-wise mean before normalization (the original code uses these means)
        input_mean = x.mean(dim=(2, 3), keepdim=False)  # [B, C]
        
        return x_norm, input_mean

class FullyConnectedGlobal(nn.Module):
    """
    Fully connected layer for updating the global feature vector.
    """
    def __init__(self, input_dim, output_dim):
        super(FullyConnectedGlobal, self).__init__()
        # Weight init similar to TF code: normal init with std dev scaled by sqrt(1/input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(1.0/input_dim)*0.01)
        nn.init.zeros_(self.fc.bias)
        
        self.act = nn.SELU()
        
    def forward(self, x):
        return self.act(self.fc(x))

class ConvBlock(nn.Module):
    """
    Encoder block: optional LeakyReLU -> Conv -> InstanceNorm -> Global track update.
    """
    def __init__(self, in_channels, out_channels, use_leaky_relu=True, stride=2):
        super(ConvBlock, self).__init__()
        self.use_leaky_relu = use_leaky_relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True)
        nn.init.normal_(self.conv.weight, 0.0, 0.02)
        nn.init.zeros_(self.conv.bias)
        
        self.inorm = InstanceNormWithStats(out_channels)

    def forward(self, x):
        # Activation before conv if use_leaky_relu
        if self.use_leaky_relu:
            x = F.leaky_relu(x, 0.2)
        x = self.conv(x)
        x_norm, input_mean = self.inorm(x)
        return x_norm, input_mean

class DeconvBlock(nn.Module):
    """
    Decoder block: LeakyReLU -> (concat skip) -> Deconv -> InstanceNorm -> Global track update -> optional dropout
    Uses a ConvTranspose2d for upsampling.
    """
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(DeconvBlock, self).__init__()
        self.use_dropout = use_dropout
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True)
        nn.init.normal_(self.deconv.weight, 0.0, 0.02)
        nn.init.zeros_(self.deconv.bias)
        
        self.inorm = InstanceNormWithStats(out_channels)
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

    def forward(self, x, skip):
        x = F.leaky_relu(x, 0.2)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.deconv(x)
        x_norm, input_mean = self.inorm(x)
        x_norm = self.dropout(x_norm)
        return x_norm, input_mean

class SVBRDFNetwork(nn.Module):
    """
    Full network:
    - 8 encoder layers
    - 8 decoder layers
    - Global track integrated at each stage
    """
    def __init__(self, ngf=64):
        super(SVBRDFNetwork, self).__init__()
        
        # Encoder layers
        self.enc1 = ConvBlock(3, ngf, use_leaky_relu=False, stride=2) # first layer no activation before conv
        self.enc2 = ConvBlock(ngf, ngf*2)
        self.enc3 = ConvBlock(ngf*2, ngf*4)
        self.enc4 = ConvBlock(ngf*4, ngf*8)
        self.enc5 = ConvBlock(ngf*8, ngf*8)
        self.enc6 = ConvBlock(ngf*8, ngf*8)
        self.enc7 = ConvBlock(ngf*8, ngf*8)
        self.enc8 = ConvBlock(ngf*8, ngf*8)
        
        # Decoder layers
        self.dec8 = DeconvBlock(ngf*8, ngf*8, use_dropout=True)
        self.dec7 = DeconvBlock(ngf*8*2, ngf*8, use_dropout=True)
        self.dec6 = DeconvBlock(ngf*8*2, ngf*8, use_dropout=True)
        self.dec5 = DeconvBlock(ngf*8*2, ngf*8, use_dropout=False)
        self.dec4 = DeconvBlock(ngf*8*2, ngf*4, use_dropout=False)
        self.dec3 = DeconvBlock(ngf*4*2, ngf*2, use_dropout=False)
        self.dec2 = DeconvBlock(ngf*2*2, ngf,   use_dropout=False)
        self.dec1 = DeconvBlock(ngf*2,   9,     use_dropout=False)  # final output has 9 channels
        
        # Prepare global track FC layers
        # We have 16 stages total: enc1..enc8, dec8..dec1
        # For each stage, global_vec = SELU(FC([global_vec; mean]))
        # Initially, global_vec dim = 3 (from input mean)
        # The final dimension after each stage matches that stage's output channels.
        
        layer_channels = [ngf, ngf*2, ngf*4, ngf*8, ngf*8, ngf*8, ngf*8, ngf*8,  # encoder
                          ngf*8, ngf*8, ngf*8, ngf*8, ngf*4, ngf*2, ngf, 9]    # decoder
        
        self.fc_layers = nn.ModuleList()
        prev_dim = 3  # start with global_vec = input_mean(3)
        for ch in layer_channels:
            input_dim = prev_dim + ch
            fc = FullyConnectedGlobal(input_dim, ch)
            self.fc_layers.append(fc)
            prev_dim = ch  # now global_vec dimension = ch

    def forward(self, x):
        B = x.size(0)
        # Initialize global vector from input mean
        global_vec = x.mean(dim=(2,3))  # [B,3]
        
        # ---------------- ENCODER ----------------
        # Stage 1 (enc1)
        enc1_out, enc1_mean = self.enc1(x)
        global_vec = self.fc_layers[0](torch.cat([global_vec, enc1_mean], dim=1)) # update global
        enc1_out = enc1_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        enc2_out, enc2_mean = self.enc2(enc1_out)
        global_vec = self.fc_layers[1](torch.cat([global_vec, enc2_mean], dim=1))
        enc2_out = enc2_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        enc3_out, enc3_mean = self.enc3(enc2_out)
        global_vec = self.fc_layers[2](torch.cat([global_vec, enc3_mean], dim=1))
        enc3_out = enc3_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        enc4_out, enc4_mean = self.enc4(enc3_out)
        global_vec = self.fc_layers[3](torch.cat([global_vec, enc4_mean], dim=1))
        enc4_out = enc4_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        enc5_out, enc5_mean = self.enc5(enc4_out)
        global_vec = self.fc_layers[4](torch.cat([global_vec, enc5_mean], dim=1))
        enc5_out = enc5_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        enc6_out, enc6_mean = self.enc6(enc5_out)
        global_vec = self.fc_layers[5](torch.cat([global_vec, enc6_mean], dim=1))
        enc6_out = enc6_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        enc7_out, enc7_mean = self.enc7(enc6_out)
        global_vec = self.fc_layers[6](torch.cat([global_vec, enc7_mean], dim=1))
        enc7_out = enc7_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        enc8_out, enc8_mean = self.enc8(enc7_out)
        global_vec = self.fc_layers[7](torch.cat([global_vec, enc8_mean], dim=1))
        enc8_out = enc8_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        # ---------------- DECODER ----------------
        # dec8
        dec8_out, dec8_mean = self.dec8(enc8_out, None)
        global_vec = self.fc_layers[8](torch.cat([global_vec, dec8_mean], dim=1))
        dec8_out = dec8_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        # dec7
        dec7_out, dec7_mean = self.dec7(dec8_out, enc7_out)
        global_vec = self.fc_layers[9](torch.cat([global_vec, dec7_mean], dim=1))
        dec7_out = dec7_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        # dec6
        dec6_out, dec6_mean = self.dec6(dec7_out, enc6_out)
        global_vec = self.fc_layers[10](torch.cat([global_vec, dec6_mean], dim=1))
        dec6_out = dec6_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        # dec5
        dec5_out, dec5_mean = self.dec5(dec6_out, enc5_out)
        global_vec = self.fc_layers[11](torch.cat([global_vec, dec5_mean], dim=1))
        dec5_out = dec5_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        # dec4
        dec4_out, dec4_mean = self.dec4(dec5_out, enc4_out)
        global_vec = self.fc_layers[12](torch.cat([global_vec, dec4_mean], dim=1))
        dec4_out = dec4_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        # dec3
        dec3_out, dec3_mean = self.dec3(dec4_out, enc3_out)
        global_vec = self.fc_layers[13](torch.cat([global_vec, dec3_mean], dim=1))
        dec3_out = dec3_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        # dec2
        dec2_out, dec2_mean = self.dec2(dec3_out, enc2_out)
        global_vec = self.fc_layers[14](torch.cat([global_vec, dec2_mean], dim=1))
        dec2_out = dec2_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        # dec1
        dec1_out, dec1_mean = self.dec1(dec2_out, enc1_out)
        global_vec = self.fc_layers[15](torch.cat([global_vec, dec1_mean], dim=1))
        dec1_out = dec1_out + global_vec.unsqueeze(2).unsqueeze(3)
        
        # dec1_out now has shape [B,9,H,W]
        # Apply tanh to output
        dec1_out = torch.tanh(dec1_out)

        return dec1_out


# Example usage:
if __name__ == "__main__":
    model = SVBRDFNetwork()
    print(model)
    # read an image from the dataset
    dataset = MaterialDataset('DummyData', image_size=256)
    dummy_input = dataset[0]['input'].unsqueeze(0)
    out = model(dummy_input)
    print(out.shape)  # should be [1, 9, 256, 256]
    print(out.min(), out.max())  # should be [-1, 1]

