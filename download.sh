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