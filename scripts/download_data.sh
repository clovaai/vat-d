mkdir -p data

wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19TndCXs9aQnPhRctUegfT60mGUqe_-p7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19TndCXs9aQnPhRctUegfT60mGUqe_-p7" -O ./data/AG_NEWS.zip && rm -rf ~/cookies.txt
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uzyjeMQXjNVppvhEeBhaOO0ynkkvVIZq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uzyjeMQXjNVppvhEeBhaOO0ynkkvVIZq" -O ./data/YAHOO.zip && rm -rf ~/cookies.txt
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10g7tLQAE-XxpBgQRgcbps47dGmaNAGl_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10g7tLQAE-XxpBgQRgcbps47dGmaNAGl_" -O ./data/DBpedia.zip && rm -rf ~/cookies.txt

unzip ./data/AG_NEWS.zip -d ./data/AG_NEWS
unzip ./data/YAHOO.zip -d ./data/YAHOO
unzip ./data/DBpedia.zip -d ./data/DBpedia

rm -rf ./data/*.zip
