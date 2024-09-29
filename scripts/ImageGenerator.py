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
import lpips
from MultiScalePerceptualLoss import MultiScalePerceptualLoss
import torch.nn.functional as F

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
        self.model = self.__load_model()  # Chargement du modèle
        self.ouput_path_image_test =os.path.join(path_output, ImageGenerator.IMAGE_TEST_FILE_NAME)
        self.path_vecteur_calculted = None
        self.vectors_calculated = self._load_vector_calculated(path_calculted)
        
        self.logging.info("ImageGenerator initialisé avec succès.")

    def _load_vector_calculated(self, path_calculted):
        dictVector = {}
        for root, dirs, files in os.walk(path_calculted):
            for file in files:
                dictVector[file] = os.path.join(root, file)
        return dictVector
        
    def load_image(self, image_to_load):
        """
        Charge et prépare l'image d'entrée pour la génération.

        Cette fonction redimensionne l'image à 1024x1024 et applique une normalisation.

        Returns:
            torch.Tensor: L'image transformée sous forme de tenseur PyTorch.
        """
        self.logging.info(f"Chargement de l'image depuis : {image_to_load}")
        image = Image.open(image_to_load)

        # Redimensionner l'image à 1024x1024
        image = image.resize((1024, 1024))
        self.logging.info("Image redimensionnée à 1024x1024.")

        # Transformation pour redimensionner et normaliser l'image d'entrée
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        image = Image.open(image_to_load).convert('RGB')
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


    def _lerp(self, a, b, t):
        """
        Interpolation linéaire entre deux tenseurs a et b selon un facteur t.
        """
        return a + t * (b - a)

    def project_with_progressive_growing(self, initial_resolution=64, final_resolution=1024, nb_steps_per_resolution=5000, stop_threshold=1e-5, reg_weight=0.0005):
        """
        Projette une image dans l'espace latent du modèle StyleGAN3 en utilisant la méthode de Progressive Growing avec interpolation linéaire (lerp).

        Args:
            initial_resolution (int): La résolution initiale pour démarrer la projection.
            final_resolution (int): La résolution finale à atteindre (jusqu'à 1024x1024).
            nb_steps_per_resolution (int): Le nombre d'étapes d'optimisation pour chaque niveau de résolution.
            stop_threshold (float): Seuil de stagnation pour l'arrêt anticipé.
            reg_weight (float): Le poids de la régularisation L2.
        """
        self.model.train()  # Assure-toi que le modèle est en mode d'entraînement

        # Résolutions intermédiaires, en doublant à chaque étape jusqu'à la résolution finale de 1024x1024
        resolutions = [initial_resolution * (2 ** i) for i in range(int(np.log2(final_resolution // initial_resolution)) + 1)]

        # Charger l'image d'entrée et la préparer à la résolution maximale
        original_image = self.load_image(self.path_input)  # Utilise ta fonction load_image

        # Initialiser le modèle LPIPS
        loss_fn = lpips.LPIPS(net='vgg').to(self.__device).eval()

        # Utiliser la moyenne de l'espace latent W comme initialisation
        w_avg = self.model.mapping.w_avg
        num_ws = self.model.mapping.num_ws

        # Initialiser w_plus à partir de w_avg et le reformater
        w_plus = w_avg.unsqueeze(0).repeat(1, num_ws, 1).clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([w_plus], lr=0.0005)

        previous_loss = float('inf')  # Initialiser la perte précédente à une valeur très élevée
        stagnation_counter = 0  # Initialiser le compteur de stagnation
        latents_dict = {}  # Suivi des vecteurs latents à chaque résolution

        for resolution in resolutions:
            self.logging.info(f"Optimisation à la résolution {resolution}x{resolution}")

            # Redimensionner dynamiquement l'image cible à chaque résolution
            target_image = F.interpolate(original_image, size=(resolution, resolution), mode='bilinear', align_corners=False)

            # Charger les vecteurs latents de la résolution précédente s'ils existent
            if resolution in latents_dict:
                old_w_plus = latents_dict[resolution]
            else:
                old_w_plus = w_plus.clone().detach()

            for step in range(nb_steps_per_resolution):
                optimizer.zero_grad()

                # Calculer l'alpha en fonction de la progression
                alpha = step / nb_steps_per_resolution

                # Interpolation progressive des latents avec lerp
                w_plus_interpolated = self._lerp(old_w_plus, w_plus, alpha)

                # Générer une image à partir du vecteur latent interpolé
                img_gen = self.model.synthesis(w_plus_interpolated, noise_mode='const', force_fp32=True)
                img_gen = F.interpolate(img_gen, size=(resolution, resolution), mode='bilinear', align_corners=False)

                # Calculer la perte perceptuelle LPIPS
                perceptual_loss = loss_fn(img_gen, target_image)

                # Calculer la perte MSE
                mse_loss = torch.nn.functional.mse_loss(img_gen, target_image)

                # Combiner les pertes avec régularisation
                total_loss = 0.2 * perceptual_loss + 0.8 * mse_loss + reg_weight * torch.norm(w_plus, p=2)
                total_loss.backward(retain_graph=True)
                optimizer.step()

                # Enregistrer les progrès et ajuster le taux d'apprentissage
                if step % 100 == 0:
                    # Ajouter une vérification pour voir la valeur d'alpha
                    self.logging.info(f"Alpha à l'étape {step}: {alpha}")
                    self.logging.info(f"Étape {step}/{nb_steps_per_resolution} à {resolution}x{resolution}, Perte totale: {total_loss.item()}")

                # Sauvegarder l'image toutes les 500 étapes
                if step % 500 == 0:
                    img_save = (img_gen * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    img_save = img_save.permute(0, 2, 3, 1).cpu().numpy()[0]
                    pil_img = Image.fromarray(img_save, 'RGB')
                    image_path_vecteur_calculted = os.path.join(self.path_output, f'output_step_{step}_res_{resolution}.png')
                    pil_img.save(image_path_vecteur_calculted)
                    print(f"Image sauvegardée à l'étape {step} pour la résolution {resolution}x{resolution}")

                previous_loss = total_loss.item()

            # Sauvegarder les vecteurs latents pour la résolution actuelle
            latents_dict[resolution] = w_plus.clone().detach()

            # Réduction du taux d'apprentissage
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(0.0001, param_group['lr'] * 0.5)



        # Sauvegarder le vecteur latent W+ résultant
        self.path_vecteur_calculted = os.path.join(self.path_calculted, 'projected_latent_final.npy')
        np.save(self.path_vecteur_calculted, w_plus.detach().cpu().numpy())
        self.logging.info(f"Vecteur latent projeté sauvegardé dans {self.path_vecteur_calculted}")



    def project(self, nb_steps=10000, reg_weight=0.001, stop_threshold=1e-5):
        """
        Projette une image dans l'espace latent du modèle StyleGAN3 en optimisant un vecteur latent.

        Args:
            nb_steps (int, optional): Le nombre d'étapes d'optimisation. Par défaut 15000.
            reg_weight (float, optional): Le poids de régularisation L2. Par défaut 0.00005.
            stop_threshold (float, optional): Le seuil d'arrêt anticipé basé sur la stagnation de la perte. Par défaut 1e-5.
        """
        self.logging.info(f"Début de la projection avec {nb_steps} étapes.")

        # Charger l'image d'entrée
        image = self.load_image(self.path_input)

        # Initialiser le modèle VGG pour LPIPS
        vgg = models.vgg16(pretrained=True).features[:16].to(self.__device).eval()
        loss_fn = lpips.LPIPS(net='vgg').to(self.__device)

        # Utiliser la moyenne de l'espace latent W comme initialisation
        w_avg = self.model.mapping.w_avg

        # Obtenir le nombre de couches (num_ws) pour le modèle
        num_ws = self.model.mapping.num_ws

        # Initialiser w_plus à partir de w_avg et le reformater avec la bonne forme
        w_plus = w_avg.unsqueeze(0).repeat(1, num_ws, 1).clone().detach().requires_grad_(True)

        # Utiliser Adam pour optimiser W+
        optimizer = torch.optim.Adam([w_plus], lr=0.0005)

        previous_loss = float('inf')
        stagnation_counter = 0

        for step in range(nb_steps):
            optimizer.zero_grad()

            # Générer l'image à partir du vecteur latent
            img_gen = self.model.synthesis(w_plus, noise_mode='const')

            # Calculer la perte perceptuelle avec LPIPS
            perceptual_loss = loss_fn(img_gen, image)

            # Calculer la perte MSE pour la correspondance pixel par pixel
            mse_loss = torch.nn.functional.mse_loss(img_gen, image)

            # Combiner les pertes LPIPS et MSE, et ajouter la régularisation
            total_loss = 0.2 * perceptual_loss + 0.8 * mse_loss + reg_weight * torch.norm(w_plus, p=2)
            
            # Backpropagation et mise à jour des paramètres
            total_loss.backward()
            optimizer.step()
            
            
            # Réduire le learning rate de manière progressive au lieu d'un seul changement brusque
            if step > 7500:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.0001

            if step > 9000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.00001

            # Enregistrer la progression et afficher la perte toutes les 100 étapes
            if step % 100 == 0:
                self.logging.info(f"Étape {step}/{nb_steps}, Perte totale: {total_loss.item()}.")

            # Sauvegarder une image toutes les 500 étapes pour suivre la progression
            if step % 500 == 0:
                img_save = (img_gen * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_save = img_save.permute(0, 2, 3, 1).cpu().numpy()[0]
                pil_img = Image.fromarray(img_save, 'RGB')
                image_path_vecteur_calculted = os.path.join(self.path_output, f'output_step_{step}.png')
                pil_img.save(image_path_vecteur_calculted)
                print(f"Image sauvegardée à l'étape {step}")

                    
            # Arrêt anticipé si la perte stagne (basée sur le seuil de stagnation)
            if abs(previous_loss - total_loss.item()) < stop_threshold:
                stagnation_counter += 1
                if stagnation_counter >= 500:  # Arrête si la perte n'a pas évolué pendant 500 itérations
                    self.logging.info(f"Arrêt anticipé à l'étape {step}, Perte totale: {total_loss.item()}.")
                    break
            else:
                stagnation_counter = 0

            previous_loss = total_loss.item()

        # Sauvegarder le vecteur latent W+ résultant dans le répertoire de sortie
        self.path_vecteur_calculted = os.path.join(self.path_calculted, 'projected_latent_{0}.npy'.format(nb_steps))
        np.save(self.path_vecteur_calculted, w_plus.detach().cpu().numpy())
        self.logging.info(f"Vecteur latent projeté sauvegardé dans {self.path_vecteur_calculted}.")

    def _apply_truncation(self, w_plus, truncation_psi, w_avg):
        """
        Applique la troncature sur le vecteur latent w_plus avec le paramètre truncation_psi.
        """
        return w_avg + (w_plus - w_avg) * truncation_psi

    def load_vector(self, vector_to_use):
        """
        Charge un vecteur latent depuis un fichier.

        Returns:
            torch.Tensor: Le vecteur latent chargé.
        """
        if vector_to_use is not None and vector_to_use in self.vectors_calculated :
            self.path_vecteur_calculted = self.vectors_calculated[vector_to_use]
            
        self.logging.info(f"Chargement du vecteur latent depuis {self.path_vecteur_calculted}...")
        latent_vector = np.load(self.path_vecteur_calculted)
        latent_vector = torch.from_numpy(latent_vector).to(self.__device)
        self.logging.info("Vecteur latent chargé avec succès.")
        return latent_vector

    def interpolate_age(self, vector_to_use, start_age=0, end_age=30, steps=30):
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
        latent_vector = self.load_vector(vector_to_use)

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
    
    def save_to_image(self, vector_to_use, ouput_path_image_test):
        self.logging.info('save_to_image')

        # Charger le vecteur latent (dans l'espace W+)
        latent_vector_W = self.load_vector(vector_to_use)

        # Vérifier que le vecteur latent a bien 3 dimensions : [batch_size, num_layers, latent_dim]
        if latent_vector_W.ndim != 3:
            raise ValueError(f"Le vecteur latent doit avoir 3 dimensions, mais a {latent_vector_W.ndim} dimensions.")

        # Générer l'image avec StyleGAN3 (utiliser directement le vecteur W+)
        manipulated_img = self.model.synthesis(latent_vector_W, noise_mode='const')

        # Convertir et afficher l'image
        manipulated_img = (manipulated_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        manipulated_img = manipulated_img.permute(0, 2, 3, 1)
        manipulated_pil_img = Image.fromarray(manipulated_img[0].cpu().numpy(), 'RGB')
        manipulated_pil_img.show()


        manipulated_pil_img.save(ouput_path_image_test)
        return


    def generate_timelapse(self,vector_to_use, truncation_psi=0.5):
        """
        Génère un timelapse en utilisant une série de vecteurs latents et sauvegarde les images générées.

        Args:
            truncation_psi (float, optional): Paramètre de troncation. Par défaut 0.5.
        """
        self.logging.info("Génération du timelapse...")
        # Obtenir les vecteurs latents interpolés par l'âge
        latents = self.interpolate_age(vector_to_use)

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


    def calculate_lpips_distance(self, generated_image_path, original_image_path):
        """
        Calcule la distance perceptuelle LPIPS entre deux images.

        Args:
            generated_image (torch.Tensor): Image générée de taille [1, 3, H, W], normalisée entre [-1, 1].
            original_image (torch.Tensor): Image originale de taille [1, 3, H, W], normalisée entre [-1, 1].

        Returns:
            float: La distance LPIPS entre les deux images.
        """
        # Initialiser le modèle LPIPS (basé sur VGG)
        loss_fn = lpips.LPIPS(net='vgg').eval()

        # S'assurer que les images sont sur le même dispositif que le modèle
        generated_image = self.load_image(generated_image_path)
        original_image = self.load_image(original_image_path)
        loss_fn = loss_fn.to(self.__device)

        # Calcul de la distance LPIPS
        with torch.no_grad():
            lpips_distance = loss_fn(generated_image, original_image)
        print(lpips_distance)
        print(lpips_distance.item())
        return lpips_distance.item()

