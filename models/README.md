Modified ResNet18 Model (Small-Image Version)
=============================================

This folder contains the full implementation of our custom **Modified ResNet-18** architecture used for facial emotion recognition. The model is specifically adapted for **small input images**, which is why it uses a modified stem instead of the original ResNet stem.

Unlike the standard ResNet-18 (7×7 conv + stride 2 + maxpool), our version starts with a **3×3 convolution with stride 1**, preserving more spatial information in early layers.

This folder only defines the model architecture and initialization utilities. It does **not** contain any training, data loading, or evaluation logic.

Architectural Overview
----------------------

The model follows the classical ResNet-18 structure with 4 stages and residual connections, but uses a custom small-image stem.

The overall structure is:

Stem → Layer1 → Layer2 → Layer3 → Layer4 → Global Average Pooling → Fully Connected Layer

Each stage consists of **BasicBlock residual blocks** with skip connections.

The number of blocks per stage is:

\[2, 2, 2, 2\]

This matches the standard ResNet-18 depth.

SmallImageStem (stem.py)
------------------------

The SmallImageStem replaces the original large-kernel ResNet stem.

It consists of:

*   Conv2d (kernel\_size=3, stride=1, padding=1, bias=False)
    
*   BatchNorm2d
    
*   ReLU (inplace=True)
    

Because stride=1 is used, spatial resolution is preserved in the early layers. This is more suitable for smaller input images compared to the classical 7×7 stride-2 stem.

BasicBlock (blocks.py)
----------------------

The BasicBlock implements the residual building block used in ResNet-18.

Each block follows this structure:

(conv → bn → relu) → (conv → bn) → add skip connection → relu

The helper function conv3x3(...) creates a 3×3 convolution with padding=1 and no bias.

If either:

*   the stride is not equal to 1, or
    
*   the number of input channels differs from the number of output channels,
    

then a projection shortcut is used:

1×1 convolution (with matching stride) followed by BatchNorm

This ensures that the residual addition is dimensionally valid.

The block expansion factor is 1, meaning the output channels remain equal to out\_ch.

ModifiedResNet18 (resnet.py)
----------------------------

The main model class is ModifiedResNet18.

Constructor parameters:

*   num\_classes (default: 6)
    
*   in\_channels (default: 3)
    
*   base\_width (default: 64)
    

The architecture is constructed as follows:

*   Layer1: 2 blocks, output channels = base\_width, stride=1
    
*   Layer2: 2 blocks, output channels = base\_width × 2, stride=2
    
*   Layer3: 2 blocks, output channels = base\_width × 4, stride=2
    
*   Layer4: 2 blocks, output channels = base\_width × 8, stride=2
    

Downsampling happens at the first block of layers 2, 3, and 4 using stride=2.

After the final stage:

*   AdaptiveAvgPool2d((1,1)) is applied
    
*   The tensor is flattened
    
*   A fully connected layer maps to num\_classes
    

If base\_width=64 and expansion=1, the final feature dimension is 512.

Model Builder (**init**.py)
---------------------------

The recommended way to construct the model is via:

build\_model(...)

This function allows creating the model either:

*   from a configuration dictionary, or
    
*   via explicit keyword arguments.
    

If a configuration dictionary is provided, it expects:

config\["model"\]\["architecture"\] config\["model"\]\["num\_classes"\] config\["model"\]\["in\_channels"\]

Supported architecture names:

*   "modified\_resnet18"
    
*   "modifiedresnet18"
    
*   "resnet18\_smallstem"
    
*   "smallstem\_resnet18"
    

If an unsupported architecture is passed, a ValueError is raised.

The function also validates that:

*   num\_classes > 0
    
*   in\_channels > 0
    

Optional Weight Initialization
------------------------------

If init\_weights=True is passed to build\_model, the function initialize\_weights\_(model) is applied.

The initialization strategy is:

*   Conv2d → Kaiming Normal (fan\_out, relu)
    
*   BatchNorm2d → weight=1, bias=0
    
*   Linear → Normal(mean=0, std=0.01), bias=0
    

This is a simple and standard initialization for ReLU-based convolutional networks.

File Structure (models folder)
------------------------------

models/ **init**.py # public API: build\_model, initialization stem.py # small-image stem implementation blocks.py # BasicBlock and helper convolution resnet.py # ModifiedResNet18 definition