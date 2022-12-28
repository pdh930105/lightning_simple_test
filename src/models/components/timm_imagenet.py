from torch import nn
from timm.models import create_model
import timm
class SimpleTimmNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        input_size: int = 224,
        num_classes: int = 1000,
        drop_out: float = 0,
        pretrained: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        try:
            net = create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        except:
            raise f"model name {model_name} does not exists in timm.create_model plz check timm site"
        
        self.net = net
        
    def forward(self, x):
        x = self.net(x)
        return x


if __name__ == "__main__":
    _ = SimpleTimmNet()
