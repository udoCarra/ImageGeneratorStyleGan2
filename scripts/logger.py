# logger_config.py

import logging
import os

LOGGER_NAME = 'timelapse_logger'

def init_log(path_output):
    """
    Initialise le logger pour écrire les messages à la fois dans la console et un fichier de log.
    
    Args:
        path_output (str): Répertoire de sortie pour sauvegarder le fichier de log.
    """
    logger = logging.getLogger(LOGGER_NAME)  # Utilisation d'un nom spécifique pour le logger
    logger.setLevel(logging.INFO)

    path_log = os.path.join(path_output, 'execution.log')
    file_handler = logging.FileHandler(path_log)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format des logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Ajouter les handlers au logger s'ils ne sont pas déjà présents
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
