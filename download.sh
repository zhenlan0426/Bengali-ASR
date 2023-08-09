# download data from kaggle
# copy kaggle.json to gcp, kaggle.json can be found under https://www.kaggle.com/settings/account
mv kaggle.json ./.kaggle/kaggle.json
kaggle competitions download -c bengaliai-speech
kaggle competitions download -c dlsprint
kaggle datasets download -d zhenlanwang/esc-50
kaggle datasets download -d zhenlanwang/model-add-data-all

mkdir data
unzip bengaliai-speech.zip -d data
rm bengaliai-speech.zip
mkdir dlspint
unzip dlsprint.zip -d dlspint
rm dlsprint.zip 
mkdir background
unzip esc-50.zip -d background
rm esc-50.zip
unzip model-add-data-all.zip
rm model-add-data-all.zip

sudo wget -O 'bn_train_5splits_split1.tar.gz' https://ee.iisc.ac.in/madasr23dataset/download/bn_train_5splits_split1.tar.gz
sudo wget -O 'bn_train_5splits_split2.tar.gz' https://ee.iisc.ac.in/madasr23dataset/download/bn_train_5splits_split2.tar.gz
sudo wget -O 'bn_train_5splits_split3.tar.gz' https://ee.iisc.ac.in/madasr23dataset/download/bn_train_5splits_split3.tar.gz
sudo wget -O 'bn_train_5splits_split4.tar.gz' https://ee.iisc.ac.in/madasr23dataset/download/bn_train_5splits_split4.tar.gz
sudo wget -O 'bn_train_5splits_split5.tar.gz' https://ee.iisc.ac.in/madasr23dataset/download/bn_train_5splits_split5.tar.gz
#wget -O 'bn_dev.tar.gz' https://ee.iisc.ac.in/madasr23dataset/download/bn_dev.tar.gz
mkdir RESPIN
sudo tar -xvzf bn_train_5splits_split1.tar.gz -C RESPIN
sudo tar -xvzf bn_train_5splits_split2.tar.gz -C RESPIN
sudo tar -xvzf bn_train_5splits_split3.tar.gz -C RESPIN
sudo tar -xvzf bn_train_5splits_split4.tar.gz -C RESPIN
sudo tar -xvzf bn_train_5splits_split5.tar.gz -C RESPIN


sudo sh -c "ulimit -S -s 10000000 && your-command-here"
"/mnt/disks/persist/RESPIN"
wget https://raw.githubusercontent.com/bloodraven66/RESPIN_ASRU_Challenge_2023/main/corpus/bn/train/text
#wget https://raw.githubusercontent.com/bloodraven66/RESPIN_ASRU_Challenge_2023/main/corpus/bn/dev/text