import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2  # 유지되는 padding
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, input_channels: int, encoder_channels: list, decoder_channels: list, kernel_sizes: list):
        """
        input_channels: 입력 feature 채널 수 (e.g. 11)
        encoder_channels: 인코더 각 블록의 출력 채널 리스트 (e.g. [32, 64, 128])
        decoder_channels: 디코더 각 블록의 출력 채널 리스트 (e.g. [64, 32])
        kernel_sizes: 각 블록에서 사용할 커널 사이즈 리스트 (encoder + decoder 길이와 동일)
        """
        super(UNet, self).__init__()

        assert len(kernel_sizes) == len(encoder_channels) + len(decoder_channels) + 1, \
            "kernel_sizes must match total number of encoder and decoder blocks"

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = input_channels
        for i, out_ch in enumerate(encoder_channels):
            self.encoders.append(ConvBlock(in_ch, out_ch, kernel_sizes[i]))
            in_ch = out_ch

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(in_ch, in_ch * 2, kernel_sizes[len(encoder_channels)])

        # 디코더 채널 설정
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        decoder_input_channels = [encoder_channels[-1] * 2] + decoder_channels[:-1]  # 업샘플 전 채널
        skip_channels = list(reversed(encoder_channels))  # skip connection 채널 수

        for i, (in_ch, out_ch) in enumerate(zip(decoder_input_channels, decoder_channels)):
            self.upsamples.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            merged_channels = out_ch + skip_channels[i]  # skip + upsample 출력 채널 수
            self.decoders.append(
                ConvBlock(merged_channels, out_ch, kernel_sizes[len(encoder_channels) + 1 + i])
            )

        # Final output layer
        self.output_layer = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        enc_features = []

        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            enc_features.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x = upsample(x)
            skip = enc_features[-(i + 1)]

            # Size mismatch due to pooling/upsample rounding? Use center crop to fix
            if x.shape[2:] != skip.shape[2:]:
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        x = self.output_layer(x)
        return torch.sigmoid(x)
