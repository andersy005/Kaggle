#! /usr/bin/env bash

export HOME_PROJ=IEEE

cd ~

mkdir -p $HOME_PROJ
cd $HOME_PROJ

wget --header 'Host: storage.googleapis.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.kaggle.com/c/sp-society-camera-model-identification/data' --header 'DNT: 1' --header 'Upgrade-Insecure-Requests: 1' 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/8078/train.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1515201358&Signature=B4ojnFCAC8k4YtD2ugGtTks3MCJFWT0hddpBCxYtjFdplHbCKGyoz5BeQ%2FLtNrzI9fPoMUL%2FxglFCGFlK2pltV22q8YWGQww42dHveOlAdD%2Bnye16CCW2suS9kjJq3KiVzrSrECBkK%2Bq6W2SD9Z%2Frpko4%2FRypdQq4c13%2F1HGW6K%2BjKWmMFEjTOAuYh8Cw2xCfIrp4o0XVqP3RZg9UZn%2Fwx%2FAsp3TNjjE8xZ4RxXwXQs1pu5yrxMm9fX6H0NuLLoxtK9VLycx5dXeg%2BNqCDYg56y%2FlbO8twePkN3JRXkYwO1nR7cXEp7XLuDW9UoJ0Qbb87bNk3yB48dpXs%2FliMKcLw%3D%3D' --output-document 'train.zip'
unzip train.zip 
rm -r train.zip

wget --header 'Host: storage.googleapis.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.kaggle.com/c/sp-society-camera-model-identification/data' --header 'DNT: 1' --header 'Upgrade-Insecure-Requests: 1' 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/8078/test.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1515201834&Signature=R6xctR2ej4KhOnjNol5Mxwo5uM1smLEvrdwNx3bmtcLTNIjaNvhSpqrtkAMhoCyzUO8vURgtQwjX4y9Y8gnYRna7I77tCwicpMoydcPsI2TWDe%2Br0nWaPKvCMTLn8qyCrQvc%2Bqfg%2BoAWNE981CIA9Q4TaZ6oWvCLP%2FPzrqILjfXzrjDr8W3x5%2BSfMHSoVW4nDDxbqCjX2L5iApmiVepxllHgIYt2Oiya71cSV0h%2BT8zZOo6aKnOS37A20jAYbyv4ZCl%2Fwbzbuy4CsAOLV83PX%2FhU6YzVTKKYB2MKPhy5oLLXmc8FHwXFDBBYSA5VcKN4t02GhHDkSZfTR6RoFVfL6w%3D%3D' --output-document 'test.zip'
unzip test.zip
rm -r test.zip 


