# Confirm that the script is running on the host
uname -a

# Install common packages
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq golang neofetch zsh mosh byobu aria2

# Install Python 3.11
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.11-full python3.11-dev

# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
sudo chsh $USER -s /usr/bin/zsh


# Create venv
python3.11 -m venv $HOME/.venv311
. $HOME/.venv311/bin/activate

# Install JAX with TPU support
pip install -U pip
pip install -U wheel
pip install -U "jax[tpu]==0.4.13" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install jupyterlab matplotlib tensorflow tensorboard-plugin-profile flax optax torch datasets librosa evaluate sentencepiece jiwer audiomentations
pip install git+https://github.com/zhenlan0426/transformers.git
pip install kaggle
pip install cached-property
pip install tensorrt