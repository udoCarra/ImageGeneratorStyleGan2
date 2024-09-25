import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import dnnlib
import legacy
import logging
from logger import init_log, LOGGER_NAME


class ImageGenerator:
    """
    Classe responsable de la génération de timelapse basée sur les vecteurs latents et le modèle StyleGAN3.
    """
    
    IMAGE_TEST_FILE_NAME = 'test_gen.png'

    def __init__(self, device, path_input, path_output, path_calculted, path_model, path_direction_vector_age):
        """
        Initialise la classe ImageGenerator avec les chemins et les paramètres nécessaires.

        Args:
            device (str): Le type d'appareil (e.g., 'cuda' ou 'cpu').
            path_input (str): Chemin de l'image d'entrée.
            path_output (str): Répertoire de sortie pour les images générées.
            path_model (str): Chemin vers le modèle StyleGAN3 pré-entraîné.
            path_direction_vector_age (str): Chemin vers le vecteur de direction d'âge utilisé pour la manipulation des vecteurs latents.
        """
        self.logging = logging.getLogger(LOGGER_NAME)  # Utiliser le logger configuré
        self.__device = torch.device(device)  # Appareil privé (cuda/cpu)
        self.path_input = path_input  # Chemin de l'image d'entrée
        self.path_output = path_output  # Répertoire de sortie
        self.path_model = path_model  # Chemin du modèle
        self.path_direction_vector_age = path_direction_vector_age  # Vecteur de direction d'âge
        self.path_calculted = path_calculted
        self.path_vecteur_calculted = None
        self.model = self.__load_model()  # Chargement du modèle
        self.ouput_path_image_test =os.path.join(path_output, ImageGenerator.IMAGE_TEST_FILE_NAME)

        self.logging.info("ImageGenerator initialisé avec succès.")

    def load_image(self):
        """
        Charge et prépare l'image d'entrée pour la génération.

        Cette fonction redimensionne l'image à 1024x1024 et applique une normalisation.

        Returns:
            torch.Tensor: L'image transformée sous forme de tenseur PyTorch.
        """
        self.logging.info(f"Chargement de l'image depuis {self.path_input}...")
        image = Image.open(self.path_input)

        # Redimensionner l'image à 1024x1024
        image = image.resize((1024, 1024))
        self.logging.info("Image redimensionnée à 1024x1024.")

        # Transformation pour redimensionner et normaliser l'image d'entrée
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        image = Image.open(self.path_input).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.__device)

        self.logging.info("Image transformée et prête pour la génération.")
        return image

    def __load_model(self):
        """
        Charge le modèle StyleGAN3 pré-entraîné.

        Returns:
            torch.nn.Module: Le modèle chargé.
        """
        self.logging.info(f"Chargement du modèle depuis {self.path_model}...")
        with dnnlib.util.open_url(self.path_model) as f:
            model = legacy.load_network_pkl(f)['G_ema'].to(self.__device)
        self.logging.info("Modèle chargé avec succès.")
        return model

    def __perceptual_loss(self, vgg, gen_img, target_img):
        """
        Calcule la perte perceptuelle basée sur les caractéristiques de VGG16 entre deux images.

        Args:
            vgg (torch.nn.Module): Le modèle VGG16 tronqué pour la perte perceptuelle.
            gen_img (torch.Tensor): L'image générée.
            target_img (torch.Tensor): L'image cible.

        Returns:
            torch.Tensor: La perte MSE entre les caractéristiques des deux images.
        """
        gen_features = vgg(gen_img)
        target_features = vgg(target_img)
        return torch.nn.functional.mse_loss(gen_features, target_features)

    def project(self, nb_steps=2000, reg_weight=0.001):
        """
        Projette une image dans l'espace latent du modèle StyleGAN3 en optimisant un vecteur latent.

        Args:
            nb_steps (int, optional): Le nombre d'étapes d'optimisation. Par défaut 2000.
            reg_weight (float, optional): Le poids de régularisation L2. Par défaut 0.001.
        """
        self.logging.info(f"Début de la projection avec {nb_steps} étapes.")
        # Charger l'image d'entrée
        image = self.load_image()

        # Charger VGG pré-entraîné pour la perte perceptuelle
        vgg = models.vgg16(pretrained=True).features[:16].to(self.__device).eval()

        # Initialiser le vecteur latent dans Z et mapper dans W
        z = torch.randn([1, self.model.z_dim], device=self.__device)
        w = self.model.mapping(z, None)

        w_plus = w.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([w_plus], lr=0.001)

        for step in range(nb_steps):
            optimizer.zero_grad()
            img_gen = self.model.synthesis(w_plus, noise_mode='const')

            loss = torch.nn.functional.mse_loss(img_gen, image) + 0.1 * self.__perceptual_loss(vgg, img_gen, image)
            reg_loss = reg_weight * torch.norm(w_plus, p=2)
            total_loss = loss + reg_loss

            total_loss.backward()
            optimizer.step()

            if step % 100 == 0:
                self.logging.info(f"Étape {step}/{nb_steps}, Perte totale: {total_loss.item()}.")

        # Sauvegarder le vecteur latent W+ résultant dans le répertoire de sortie
        self.path_vecteur_calculted = os.path.join(self.path_calculted, 'projected_latent_{0}.npy'.format(nb_steps))
        np.save(self.path_vecteur_calculted, w_plus.detach().cpu().numpy())
        self.logging.info(f"Vecteur latent projeté sauvegardé dans {self.path_vecteur_calculted}.")

    def _apply_truncation(self, w_plus, truncation_psi, w_avg):
        """
        Applique la troncature sur le vecteur latent w_plus avec le paramètre truncation_psi.
        """
        return w_avg + (w_plus - w_avg) * truncation_psi

    def load_vector(self):
        """
        Charge un vecteur latent depuis un fichier.

        Returns:
            torch.Tensor: Le vecteur latent chargé.
        """
        self.logging.info(f"Chargement du vecteur latent depuis {self.path_vecteur_calculted}...")
        latent_vector = np.load(self.path_vecteur_calculted)
        latent_vector = torch.from_numpy(latent_vector).to(self.__device)
        self.logging.info("Vecteur latent chargé avec succès.")
        return latent_vector

    def interpolate_age(self, start_age=0, end_age=30, steps=30):
        """
        Interpole entre plusieurs vecteurs latents pour simuler l'effet de l'âge.

        Args:
            start_age (float, optional): L'âge de départ pour l'interpolation. Par défaut 0.
            end_age (float, optional): L'âge de fin pour l'interpolation. Par défaut 30.
            steps (int, optional): Le nombre d'étapes d'interpolation. Par défaut 30.

        Returns:
            list: Liste de vecteurs latents interpolés représentant différentes étapes de l'âge.
        """
        self.logging.info(f"Interpolation des âges de {start_age} à {end_age} sur {steps} étapes.")
        # Charger le vecteur latent
        latent_vector = self.load_vector()

        # Charger la direction de l'âge
        age_direction = np.load(self.path_direction_vector_age)
        age_direction = torch.tensor(age_direction).to(self.__device)

        # Générer des vecteurs latents modifiés par l'âge
        age_range = torch.logspace(start_age, end_age, steps, base=2, dtype=torch.float32)
        latent_vectors = []
        for age_factor in age_range:
            modified_latent = latent_vector + age_factor * age_direction
            latent_vectors.append(modified_latent)

        self.logging.info("Interpolation terminée.")
        return latent_vectors
    
    def save_to_image(self):
        print('save_to_image')

        # Charger le vecteur latent (dans l'espace W+)
        latent_vector_W = self.load_vector()

        # Vérifier que le vecteur latent a bien 3 dimensions : [batch_size, num_layers, latent_dim]
        if latent_vector_W.ndim != 3:
            raise ValueError(f"Le vecteur latent doit avoir 3 dimensions, mais a {latent_vector_W.ndim} dimensions.")

        # Générer l'image avec StyleGAN3 (utiliser directement le vecteur W+)
        manipulated_img = self.model.synthesis(latent_vector_W, noise_mode='const')
        print(manipulated_img.shape)  # Vérifiez la forme avant de permuter

        # Convertir et afficher l'image
        manipulated_img = (manipulated_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        manipulated_img = manipulated_img.permute(0, 2, 3, 1)
        manipulated_pil_img = Image.fromarray(manipulated_img[0].cpu().numpy(), 'RGB')
        manipulated_pil_img.show()

        # Sauvegarder l'image dans un fichier local
        manipulated_pil_img.save(self.ouput_path_image_test)
        return


    def generate_timelapse(self, truncation_psi=0.5):
        """
        Génère un timelapse en utilisant une série de vecteurs latents et sauvegarde les images générées.

        Args:
            truncation_psi (float, optional): Paramètre de troncation. Par défaut 0.5.
        """
        self.logging.info("Génération du timelapse...")
        # Obtenir les vecteurs latents interpolés par l'âge
        latents = self.interpolate_age()

        # Obtenir w_avg (moyenne latente)
        w_avg = self.model.mapping.w_avg

        # Générer des images pour chaque vecteur latent
        for idx, latent in enumerate(latents):
            with torch.no_grad():
                latent = self._apply_truncation(latent, truncation_psi, w_avg)
                generated_image_tuple = self.model.synthesis(latent, noise_mode='const')
                manipulated_img = generated_image_tuple[0]

                # Vérifier la forme du tenseur généré
                print(f"Forme du tenseur généré: {manipulated_img.shape}")  # torch.Size([3, 1024, 1024])

                # Convertir l'image générée pour PIL
                manipulated_img = (manipulated_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                manipulated_img = manipulated_img.permute(1, 2, 0)  # Passer de [C, H, W] à [H, W, C]

                # Convertir en image PIL
                manipulated_pil_img = Image.fromarray(manipulated_img.cpu().numpy(), 'RGB')

                # Sauvegarder l'image
                outputPathTimelapse = os.path.join(self.path_output, f"image_{idx:03d}.png")
                manipulated_pil_img.save(outputPathTimelapse)
                print(f"Image sauvegardée : {outputPathTimelapse}")

        return
