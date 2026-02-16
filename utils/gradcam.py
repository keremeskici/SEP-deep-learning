import torch
import torch.nn.functional as F
import numpy as np

# GradCam implementation for the project
# based on the paper / tutorials I found
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        # hook handles so we can remove them later
        self.h1 = self.target_layer.register_forward_hook(self.save_activation)
        self.h2 = self.target_layer.register_full_backward_hook(self.save_gradient)
        
        self.gradients = None
        self.activations = None

    def save_gradient(self, module, grad_input, grad_output):
        # save gradients during backward pass
        self.gradients = grad_output[0].detach()

    def save_activation(self, module, input, output):
        # save activations during forward pass
        self.activations = output.detach()

    def remove_hooks(self):
        # cleanup
        self.h1.remove()
        self.h2.remove()

    # using __call__ so we can use it like a function
    def __call__(self, input_tensor, target_class=None):
        # handle the case where input is missing batch dim
        if len(input_tensor.shape) == 3:
             input_tensor = input_tensor.unsqueeze(0)

        # make sure model is in eval mode
        self.model.eval()
        
        # need gradients for backward pass
        input_tensor = input_tensor.requires_grad_(True)
        
        # run the model
        output = self.model(input_tensor)
        
        if target_class is None:
            # .item() to get the number from tensor
            target_class = torch.argmax(output).item()

        self.model.zero_grad()
        
        # get score for target class
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # get saved gradients and activations
        grads = self.gradients
        fmaps = self.activations
        
        # pooling weights
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        
        # weighted combination
        cam = torch.sum(weights * fmaps, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # resize to match input size (64x64)
        # assuming input is (B, C, H, W)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), 
                           mode='bilinear', align_corners=False)
        
        # normalize to 0-1 range
        cam = cam.squeeze()
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8) # avoid div by zero
        
        # return the heatmap (numpy) and the class we used
        return cam.cpu().numpy(), target_class
