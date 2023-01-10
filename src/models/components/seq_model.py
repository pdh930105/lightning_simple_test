from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification

class SequenceClassificiationModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        configs,
        pretrained=True
    ):
        super().__init__()
        self.model_name = model_name
        self.configs = configs
        if pretrained:
            try:
                net = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.configs)
            except:
                raise f"model name {model_name} does not exists in AutoModelForSequenceClassification.from_pretrained"
        else:
            try:
                net = AutoModelForSequenceClassification.from_config(self.configs)
            except:
                raise f"model config {self.configs} does not exist in AutoModelForSequenceClassification.from_conifg"
        
        self.net = net
        
    def forward(self, **inputs):
        x = self.net(**inputs)
        return x


if __name__ == "__main__":
    model_name= "distilbert-base-cased"
    configs = AutoConfig.from_pretrained(model_name, num_labels=3)
    _ = SequenceClassificiationModel(model_name, configs)
