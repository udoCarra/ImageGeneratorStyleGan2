import torch
import torch.nn as nn
import torchvision.models as models

class MultiScalePerceptualLoss(nn.Module):
    def __init__(self, device):
        super(MultiScalePerceptualLoss, self).__init__()
        self.device = device
        
        # Charger VGG pré-entrainé
        vgg = models.vgg16(pretrained=True).features.to(self.device).eval()
        
        # Nous utilisons différentes couches de VGG pour capturer les caractéristiques à différentes échelles
        self.layers = [vgg[:4], vgg[:9], vgg[:16], vgg[:23]]
        for layer in self.layers:
            layer.requires_grad_(False)  # Geler les poids de VGG

    def forward(self, generated_image, target_image):
        losses = []

        for layer in self.layers:
            # Extraire les caractéristiques de l'image générée et de l'image cible à chaque échelle
            gen_feats = layer(generated_image)
            target_feats = layer(target_image)
            
            # Calculer la perte L2 (ou une autre fonction de perte comme L1) entre les caractéristiques
            loss = torch.nn.functional.mse_loss(gen_feats, target_feats)
            losses.append(loss)

        # Combiner les pertes des différentes échelles (en les sommant ici, mais tu peux aussi les pondérer)
        total_loss = sum(losses)
        return total_loss