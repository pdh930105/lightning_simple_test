from torch import nn


class SimpleCifarNet(nn.Module):
    def __init__(
        self,
        input_size: int = 32,
        lin1_channel_size: int = 256,
        lin2_channel_size: int = 256,
        lin3_channel_size: int = 256,
        output_size: int = 10,
        drop_out: float = 0
    ):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, lin1_channel_size, 3, stride=2, padding=1), # 16
            nn.BatchNorm2d(lin1_channel_size),
            nn.ReLU(),
            nn.Conv2d(lin1_channel_size, lin2_channel_size, 3, stride=2, padding=1), # 8
            nn.BatchNorm2d(lin2_channel_size),
            nn.ReLU(),
            
            nn.Conv2d(lin2_channel_size, lin3_channel_size, 3, stride=2, padding=1), # 4
            nn.BatchNorm2d(lin3_channel_size),
            nn.ReLU(),
            
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(drop_out),
            nn.Linear(lin3_channel_size, output_size),

        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.feature(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    _ = SimpleCifarNet()
