# CSI_5386_NLP_Project
image captioning: CNN+RNN

Backup of this project: https://github.com/RichardChangCA/CSI_5386_NLP_Project

Contents: CNN encoders(VGG16,InceptionV3,MobileNet,ResNet) by transfer learning, RNN decoders(stacked LSTM, GRU with attention mechanism), Evaluation Metrics(BLEU,CIDEr,METEOR), Datasets(Flickr8k, COCO), Django Web Application with the best performance model.

To run the web:

python3 -m pip install django

cd project_web_

python3 manage.py runserver

![avator](https://github.com/RichardChangCA/Image_Captioning/blob/master/django_web_result/web_upload_1.png)

git reset --soft HEAD^ to undo commit

git reset to undo git add
