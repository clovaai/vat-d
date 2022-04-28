mkdir -p data

wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=162U8VA8k9SgF2gQX0FUYHSpBK9Fl4sIH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=162U8VA8k9SgF2gQX0FUYHSpBK9Fl4sIH" -O ./data/AG_NEWS.zip && rm -rf ~/cookies.txt
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1SxG9xY0ngMCXEARfLCl58nYxwRjLniM4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1SxG9xY0ngMCXEARfLCl58nYxwRjLniM4" -O ./data/YAHOO.zip && rm -rf ~/cookies.txt
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1W1GefoaaodEL_xegYZLcFSHohANIFKs0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1W1GefoaaodEL_xegYZLcFSHohANIFKs0" -O ./data/DBpedia.zip && rm -rf ~/cookies.txt

unzip ./data/AG_NEWS.zip -d ./data/AG_NEWS
unzip ./data/YAHOO.zip -d ./data/YAHOO
unzip ./data/DBpedia.zip -d ./data/DBpedia

rm -rf ./data/*.zip