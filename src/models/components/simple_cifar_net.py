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

        self.model = nn.Sequential(
            nn.Conv2d(lin1_channel_size, 3, 3, stride=1, padding=1), # 16
            nn.BatchNorm2d(lin1_channel_size),
            nn.ReLU(),
            nn.Conv2d(lin2_channel_size, lin1_channel_size, 3, stride=1, padding=1), # 8
            nn.BatchNorm2d(lin2_channel_size),
            nn.ReLU(),
            
            nn.Conv2d(lin3_channel_size, lin2_channel_size, 3, stride=1, padding=1), # 4
            nn.BatchNorm2d(lin3_channel_size),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(drop_out),
            nn.Linear(4*4*lin3_channel_size, output_size),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleCifarNet()
