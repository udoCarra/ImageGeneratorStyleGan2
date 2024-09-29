import logging
import os
from datetime import datetime

LOGGER_NAME = 'timelapse_logger'
MAX_LOG_FILES = 10

def cleanup_old_logs(path_logs):
    """
    Supprime les fichiers de log les plus anciens si le nombre de fichiers dépasse MAX_LOG_FILES.
    
    Args:
        path_logs (str): Répertoire contenant les fichiers de log.
    """
    log_files = [f for f in os.listdir(path_logs) if f.startswith('execution.log_')]
    log_files.sort(key=lambda f: os.path.getmtime(os.path.join(path_logs, f)))

    if len(log_files) > MAX_LOG_FILES:
        files_to_delete = log_files[:-MAX_LOG_FILES]
        for file in files_to_delete:
            os.remove(os.path.join(path_logs, file))

def init_log(path_logs):
    """
    Initialise le logger pour écrire les messages à la fois dans la console et un fichier de log,
    tout en assurant que les anciens fichiers de log sont supprimés si plus de 10 logs existent.
    
    Args:
        path_logs (str): Répertoire de sortie pour sauvegarder le fichier de log.
    """
    cleanup_old_logs(path_logs)
    
    logger = logging.getLogger(LOGGER_NAME)  # Utilisation d'un nom spécifique pour le logger
    logger.setLevel(logging.INFO)
    
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_log = os.path.join(path_logs, 'execution.log_{0}'.format(now))
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
