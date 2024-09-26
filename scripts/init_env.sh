cd ../src/

# Cloner le dépôt Pixel2Style2Pixel
git clone https://github.com/genforce/interfacegan.git
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
git clone https://github.com/NVlabs/stylegan3.git

# TELECHARGER LES MODELES DEPUIS LE NAVIGATEUR
rm stylegan3-t-ffhq-1024x1024.pkl
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl


# Mettre à jour pip
pip install --upgrade pip

# Installer les dépendances système via pip
pip install numpy>=1.20 click>=8.0 pillow==8.3.1 scipy==1.7.3 torch==2.4.0+cu122 torchvision==0.15.0+cu122 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu122

pip install  tqdm==4.62.2 ninja==1.10.2
# Installer les autres dépendances spécifiques via pip
pip install imgui==1.4.0 glfw==2.2.0 pyopengl==3.1.5 imageio-ffmpeg==0.4.3 pyspng
pip install requests==2.31.0
pip install matplotlib==3.7.1
pip install imageio==2.33.0

apt-get install build-essential -y
apt-get install cuda-toolkit-12-1 -y

rm -rf /root/.cache/torch_extensions/
