import os,sys,shutil
from PIL import Image
import numpy as np
import torch
from torch.utils.cpp_extension import load
sys.path.insert(0, "../src/stylegan3")
import dnnlib
import legacy
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import logging
import traceback
from ImageGenerator import ImageGenerator
from logger import init_log  # Importation de la fonction pour initialiser le logger

# CONFIG
PATH_INPUT = '../data/test_photo.png'
PATH_OUTPUT = '../output/'
PATH_CALCUL = '../calculated/'
PATH_MODEL = '../src/stylegan3-t-ffhq-1024x1024.pkl'
PATH_DIRECTION_VECTOR_AGE = '../src/interfacegan/boundaries/stylegan_ffhq_age_boundary.npy'
DEVICE_TYPE = 'cuda'
PROJ_NB_STEPS = 10000
START_AGE = 0
END_AGE = 30
RELOAD_PROJ = True
TRUNCATION_PSI = 0.7


import shutil
import logging
import os


def main():
    """
    Fonction principale qui initialise le générateur de timelapse et gère le flux global de l'exécution.
    """
    try:
        # Supprimer le répertoire de sortie s'il existe déjà, puis le recréer
        if os.path.exists(PATH_OUTPUT):
            shutil.rmtree(PATH_OUTPUT)
        os.makedirs(PATH_OUTPUT)
                    
        # Initialiser les logs
        logging = init_log(PATH_OUTPUT)  # Initialisation du logger

        logging.info("Démarrage de l'exécution")

        # Initialiser le générateur de timelapse
        generator = ImageGenerator(
            device=DEVICE_TYPE,
            path_input=PATH_INPUT,
            path_output=PATH_OUTPUT,
            path_calculted =PATH_CALCUL,
            path_model=PATH_MODEL,
            path_direction_vector_age=PATH_DIRECTION_VECTOR_AGE
        )
        
        if RELOAD_PROJ:
            # Projeter l'image dans l'espace latent
            generator.project(nb_steps=PROJ_NB_STEPS)
        
        generator.save_to_image()
        # Générer le timelapse à partir des vecteurs latents interpolés
        generator.generate_timelapse(truncation_psi=TRUNCATION_PSI)

        logging.info("Exécution terminée avec succès")

    except Exception as e:
        logging.error("Une erreur est survenue")
        logging.exception("Détails de l'exception : ")
        raise


if __name__ == "__main__":
    main()