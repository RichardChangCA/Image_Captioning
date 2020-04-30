from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators import csrf
import os
import sys
from PIL import Image 
sys.path.insert(0,'../Comparison')
print("system path: ",sys.path)
from Comparison import prediction

os.chdir("../project_web_")
IMAGE_DIR='./static/images'
RESULT_BASE_DIR = '../Comparison/final_results'
attention_img_name = 'attention_plot.png'
print(attention_img_name)
def upload(request):
    print("current path: ",os.getcwd())
    context = {}
    if request.POST:
        img_file = request.FILES.get("image")
        img_name = img_file.name
        absolute_image_name = os.path.join('static/images',img_name)
        f = open(os.path.join(IMAGE_DIR,img_name), 'wb')
        for chunk in img_file.chunks(chunk_size=1024):
            f.write(chunk)
        f.close()
        os.chdir("../Comparison")
        whether_attention,dataset_name,model_name,encoder_name = prediction(os.path.join("../project_web_",absolute_image_name))
        os.chdir("../project_web_")
        with open(os.path.join(RESULT_BASE_DIR,'predict_caption_result.txt'),'r') as f:
            predict_caption=f.read()
        # call image captioning machine
        context['rlt'] = "Predict Caption: " + predict_caption
        context['original_image'] = str(os.path.join('images',img_name))
        if(whether_attention==True):
            attention_image_path = os.path.join(RESULT_BASE_DIR,attention_img_name)
            attention_image = Image.open(attention_image_path)
            attention_image.save(os.path.join(IMAGE_DIR,attention_img_name))
            context['attention_image'] = str(os.path.join('images',attention_img_name))
    return render(request, 'upload.html', context)