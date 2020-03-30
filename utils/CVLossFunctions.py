
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# ## Pixel-wise Loss function

# In[2]:


class PixelwiseLoss(nn.Module):
    def forward(self, inputs, targets):
        return F.smooth_l1_loss(inputs, targets)


# # Loss functions based on trained models (VGG)
# Perceptual, Texture, Topological, Content/Style Losses all follow the same principle -
# 1. Extract the feature map's activation output from a trained VGG network for the both the predicted and the target image
# 2. Compare these outputs using a user specified loss function
# 
# So, we can reuse some of our code across all these loss functions.
# We define a feature loss class, that takes in -
# 1. loss function, 
# 2. VGG block indices from which the feature maps need to be extracted, 
# 3. weights to be assigned to the feature map outputs when computing the cumulative loss
# 
# Note: 
# - You can use any trained CNN network in these loss functions (ResNet, GoogLeNet, VGG etc), but it has been observed that vgg works better when used in this use-case. 
# - In fact adding trained VGG with batch norm performs even better, hence we choose to work with the vgg16_bn from pytorch. 
# - You can also use VGG19 network, which is deeper than VGG16, however it only improves the performance by a slight margin, while adding heavily to the computational cost.

# In[3]:


from torchvision.models import vgg16_bn


# In[4]:


class FeatureLoss(nn.Module):
    def __init__(self, loss, blocks, weights, device):
        super().__init__()
        self.feature_loss = loss
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)

        self.weights = torch.tensor(weights).to(device)
        #VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end
        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        vgg = vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        vgg = vgg.to(device)

        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, inputs, targets):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        inputs = F.normalize(inputs, 2, 1)
        targets = F.normalize(targets, 2, 1)

        # extract feature maps
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0
        
        # compare their weighted loss
        for lhs, rhs, w in zip(input_features, target_features, self.weights):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            loss += self.feature_loss(lhs, rhs) * w

        return loss


# In[5]:


class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()


# ## Perceptual Loss

# In[6]:


def perceptual_loss(x, y):
    loss = F.mse_loss(x, y)
    return loss
    
def PerceptualLoss(blocks, weights, device):
    return FeatureLoss(perceptual_loss, blocks, weights, device)


# ## Texture Loss

# In[7]:


def gram_matrix(x):
    c, h, w = x.size()
    x = x.view(c, -1)
    x = torch.mm(x, x.t()) / (c * h * w)
    return x

def gram_loss(x, y):
    return F.mse_loss(gram_matrix(x), gram_matrix(y))

def TextureLoss(blocks, weights, device):
    return FeatureLoss(gram_loss, blocks, weights, device)


# ## Content/Style Loss

# In[8]:


def content_loss(content, pred):
    return FeatureLoss(perceptual_loss, blocks, weights, device)

def style_loss(style, pred):
    return FeatureLoss(gram_loss, blocks, weights, device)

def content_style_loss(content, style, pred, alpha, beta):
    return alpha * content_loss(content, pred) + beta * style_loss(style, pred)


# ## Topology-aware Loss

# In[9]:


class TopologyAwareLoss(nn.Module):

    def __init__(self, criteria, weights): 
        # Here criteria -> [PixelwiseLoss, PerceptualLoss], 
        #weights -> [1, mu] (or any other combination weights)
        assert len(weights) == len(criteria)

        self.criteria = criteria
        self.weights = weights

    def forward(self, inputs, targets):
        loss = 0.0
        for criterion, w in zip(self.criteria, self.weights):
            each = w * criterion(inputs, targets)
            loss += each

        return loss


# # GAN losses
# Strongly suggest checking out this repository for GAN implementations. https://github.com/eriklindernoren/PyTorch-GAN

# In[10]:


class MinMaxGeneratorLoss(nn.Module):
    def forward(self, fake, discriminator):
        return torch.log(1 - discriminator(fake))

class MinMaxDiscriminatorLoss(nn.Module):
    def forward(self, real, fake, discriminator):
        return -1.0*(log(discriminator(real)) + log(1-discriminator(fake)))


# In[11]:


class NonSaturatingGeneratorLoss(nn.Module):
    def forward(self, fake, discriminator):
        return -torch.log(discriminator(fake))


# In[12]:


class LeastSquaresGeneratorLoss(nn.Module):
    def forward(self, fake, discriminator):
        return (discriminator(fake)-1)**2

class LeastsquaresDiscriminatorLoss(nn.Module):
    def forward(self, real, fake, discriminator):
        return (discriminator(real)-1)**2 + discriminator(fake)**2


# In[13]:


#wgan has an additional step of clipping the weights between 0, 1
#refer - https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py 
class WGANGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake, discriminator):
        return -discriminator(fake).mean()


class WGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, fake, discriminator):
        return discriminator(fake).mean() - discriminator(real).mean()


# In[14]:


class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        self.loss = nn.L1Loss()
    def forward(self, F, G, x, y):
        return self.loss(F(G(x)), x) + self.loss(G(F(y)), y)

