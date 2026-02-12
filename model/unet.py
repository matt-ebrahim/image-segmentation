import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """Attention Block for Attention U-Net"""
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of channels from gating signal (upsampling)
            F_l: Number of channels from skip connection (encoding)
            F_int: Number of intermediate channels
        """
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        Args:
            g: gating signal from decoder
            x: skip connection from encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class ConvBlock(nn.Module):
    """Convolutional Block with residual connection"""
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
        
        # Residual connection if channels don't match
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual = None
            
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.residual is not None:
            residual = self.residual(x)
            
        out += residual
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
            
        return out

class DownBlock(nn.Module):
    """Downsampling block with max pooling"""
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels, dropout_p)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.conv_block(x)
        return x

class UpBlock(nn.Module):
    """Upsampling block with attention mechanism"""
    def __init__(self, in_channels, out_channels, dropout_p=0.0, bilinear=True):
        super(UpBlock, self).__init__()
        # In U-Net decoding: in_channels is from previous layer, out_channels is from skip connection
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else \
                  nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
                  
        # After upsampling and concatenation with skip connection
        self.conv_block = ConvBlock(in_channels + out_channels, out_channels, dropout_p)
            
        # Attention block
        # F_g is the gating signal channels (upsampled features), F_l is skip connection channels
        self.attention = AttentionBlock(F_g=in_channels, F_l=out_channels, F_int=out_channels // 2)
        
    def forward(self, x, bridge):
        x = self.up(x)
        
        # Apply attention mechanism - gating signal is upsampled features, skip connection is encoder features
        attended_bridge = self.attention(g=x, x=bridge)
        
        # Padding for concatenation if needed
        diff_y = attended_bridge.shape[2] - x.shape[2]
        diff_x = attended_bridge.shape[3] - x.shape[3]
        
        if diff_y > 0 or diff_x > 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([attended_bridge, x], dim=1)
        x = self.conv_block(x)
        return x

class AttentionUNet(nn.Module):
    """Attention U-Net with deep supervision"""
    def __init__(self, in_channels=1, num_classes=2, base_channels=32, dropout_p=0.1, deep_supervision=True):
        super(AttentionUNet, self).__init__()
        self.deep_supervision = deep_supervision
        base_ch = base_channels
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch, dropout_p)
        self.enc2 = DownBlock(base_ch, base_ch * 2, dropout_p)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4, dropout_p)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8, dropout_p)
        self.enc5 = DownBlock(base_ch * 8, base_ch * 16, dropout_p)
        
        # Decoder
        self.dec4 = UpBlock(base_ch * 16, base_ch * 8, dropout_p)
        self.dec3 = UpBlock(base_ch * 8, base_ch * 4, dropout_p)
        self.dec2 = UpBlock(base_ch * 4, base_ch * 2, dropout_p)
        self.dec1 = UpBlock(base_ch * 2, base_ch, dropout_p)
        
        # Deep supervision outputs
        if deep_supervision:
            self.deep_supervision_out4 = nn.Conv2d(base_ch * 8, num_classes, kernel_size=1)
            self.deep_supervision_out3 = nn.Conv2d(base_ch * 4, num_classes, kernel_size=1)
            self.deep_supervision_out2 = nn.Conv2d(base_ch * 2, num_classes, kernel_size=1)
            
        # Final output
        self.final_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        # Decoder
        dec4 = self.dec4(enc5, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)
        
        # Output
        out = self.final_conv(dec1)
        
        if self.deep_supervision and self.training:
            # Deep supervision outputs during training
            out4 = self.deep_supervision_out4(dec4)
            out3 = self.deep_supervision_out3(dec3)
            out2 = self.deep_supervision_out2(dec2)
            return out, out2, out3, out4
        else:
            return out

# Test the model architecture
if __name__ == "__main__":
    # Create model instance
    model = AttentionUNet(in_channels=1, num_classes=2)
    print(f"Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with sample input
    sample_input = torch.randn(1, 1, 256, 256)  # batch_size=1, channels=1, height=256, width=256
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape if isinstance(output, torch.Tensor) else output[0].shape}")