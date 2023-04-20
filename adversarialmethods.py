import torch
import torch.nn.functional as F
from torch.autograd import Variable

from DeepFool.Python.deepfool import *

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Return the perturbed image
    return perturbed_image

def jsma_attack(model, x, y, gamma=0.1, iters=50):
    # Convert x and y to variables
    # x = Variable(x, requires_grad=True)
    # y = Variable(y)
    
    # Define the target class (the class we want the model to classify the input as)
    device = next(model.parameters()).device
    y_target = y
    perturbation = torch.zeros_like(x)
    # Loop over iterations of the attack
    for i in range(iters):
        # Zero-out the gradients of the model parameters
        model.zero_grad()

        # Compute the forward pass of the model on the perturbed input
        out = model(x + perturbation)

        # Compute the loss function as the negative log-likelihood of the target class
        loss = F.cross_entropy(out, y_target)

        # Compute the gradient of the loss function with respect to the input
        grad = torch.autograd.grad(loss, x)[0]

        # Compute the saliency map as the absolute value of the gradient
        saliency_map = torch.abs(grad)

        # Compute the mask of pixels to perturb as the top gamma fraction of the saliency map
        num_pixels = saliency_map.numel()
        num_perturb = int(gamma * num_pixels)
        saliency_flat = saliency_map.flatten()
        _, indices = saliency_flat.topk(num_perturb)
        mask_flat = torch.zeros(num_pixels, dtype=torch.bool, device=device)
        mask_flat[indices] = True
        mask = mask_flat.view_as(saliency_map)

        # Compute the maximum amount of perturbation per pixel
        max_perturb = torch.min(torch.abs(x - 0), torch.abs(x - 1))
        max_perturb[mask] = gamma / 255.0

        # Compute the perturbation tensor by clipping the sum of the current perturbation and the
        # maximum perturbation to the range [0, 1] and subtracting the current perturbation

        # Check if the perturbation has succeeded in changing the class of the input
        out_perturbed = model(x + perturbation)
        _, y_pred = torch.max(out_perturbed, dim=1)
        if y_pred.item() != y_target:
            break

    # Compute the adversarial example tensor by adding the perturbation to the input
    adv_x = x + perturbation
    
    return adv_x, y_pred