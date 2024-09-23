from turtle import mode
from torch import nn
from torchvision import models




# This works with pip install timm==0.3.4 for this to work with the pretrained stuff
__all__ = ['efficientnetv2s']

class EfficientNetV2S(nn.Module):

    """ ViT model extended for Hand Recognition 


    Attributes
    ----------
    pretrained: bool
        If set to `True` uses the pretrained DenseNet model as the base. If set to `False`, the network
        will be trained from scratch. 
        default: True    
    """

    def __init__(self, num_features = 512, pretrained=True, qs = 1):
        """ Init function

        Parameters
        ----------
        pretrained: bool
            If set to `True` uses the pretrained densenet model as the base. Else, it uses the default network
            default: True
        """
        super(EfficientNetV2S, self).__init__()
        
        weights = 'IMAGENET1K_V1' if pretrained else None

        self.model = models.efficientnet_v2_s(weights=weights)

        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_features, bias=False)

        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        self.qs=nn.Linear(num_features, qs)

        # for param in  self.vit.parameters():
        #     param.requires_grad = False

    
    def forward(self, x):
        """ Propagate data through the network

        Parameters
        ----------
        img: :py:class:`torch.Tensor` 
          The data to forward through the network. Expects RGB image of size 3x224x224

        Returns
        -------
        feat: :py:class:`torch.Tensor` 
            Embedding
        op: :py:class:`torch.Tensor`
            Final binary score.  

        """
        x = self.model(x)
        x = self.features(x)
        return x, self.qs(x)


def _efficientnet(arch, pretrained, progress, **kwargs):
    model = EfficientNetV2S(pretrained=pretrained, **kwargs)
    return model

def efficientnetv2s(pretrained=True, progress=True, **kwargs):
    return _efficientnet('efficientnetv2s', pretrained,
                    progress, **kwargs)


# input=torch.randn(10, 3, 356, 356,dtype=torch.float)
# model=efficientnetv2s(pretrained=True)
# print(model)
# y=model(input)
