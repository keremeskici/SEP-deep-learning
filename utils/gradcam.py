import torch
import torch.nn.functional as F
import numpy as np

# GradCam implementation for the project
# based on the paper / tutorials I found
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval() # make sure its in eval mode
        
        # placeholders for the data we need to catch
        self.gradients = None
        self.activations = None
        
        # WAITING FOR INPUT
        # usually its the last conv layer, maybe layer4?
        self.target_layer = None 
        
        # self.target_layer.register_forward_hook(self.save_activation)
        # self.target_layer.register_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_input, grad_output):
        # save gradients for later
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        # save the feature maps
        self.activations = output

    def get_cam(self, x, class_idx=None):
        # x is the image tensor
        
        # run the model
        output = self.model(x)
        
        if class_idx is None:
            # if no class provided, pick the best one
            class_idx = torch.argmax(output)

        # clear any old gradients
        self.model.zero_grad()
        
        # get the score for the target class
        score = output[0, class_idx]
        
        # compute gradients
        score.backward()
        
        # getting the values from hooks
        grads = self.gradients
        fmaps = self.activations
        
        # calculating the weights (pooling)
        # shape is usually (batch, channels, h, w)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        
        # multiply feature maps by weights
        # this is the weighted combination part
        cam = torch.sum(weights * fmaps, dim=1, keepdim=True)
        
        # apply relu because we only want positive impact
        cam = F.relu(cam)
        
        # resize it back to image size 64x64
        cam = F.interpolate(cam, size=(64, 64), mode='bilinear', align_corners=False)
        
        # normalize to 0-1 range so we can plot it
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        
        # return as a simple numpy array for the demo script
        return cam.data.cpu().numpy()[0, 0]
