mkdir ../src/
cd ../src/

# Cloner le dépôt Pixel2Style2Pixel
git clone https://github.com/genforce/interfacegan.git
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
git clone https://github.com/NVlabs/stylegan3.git

# TELECHARGER LES MODELES DEPUIS LE NAVIGATEUR
rm stylegan3-t-ffhq-1024x1024.pkl
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl

