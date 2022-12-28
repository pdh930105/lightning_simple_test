from torch import nn
from timm.models import create_model
import timm
class SimpleTimmCifarNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        input_size: int = 32,
        num_classes: int = 10,
        drop_out: float = 0,
        pretrained: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        try:
            net = create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        except:
            raise f"model name {model_name} does not exists in timm.create_model plz check timm site"
        
        self.net = net
        
        if input_size == 32:
            print("change cifar module")
            if hasattr(self.net, 'conv1'):
                old_conv = self.net.conv1
                self.net.conv1 = nn.Conv2d(old_conv.in_channels, old_conv.out_channels, kernel_size=3, stride=1, padding=0)
                del old_conv
            else:
                raise "change timm imagenet model to cifar10 is required conv1 module"
    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.net(x)
        return x


if __name__ == "__main__":
    _ = SimpleTimmNet()
