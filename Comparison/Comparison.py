# Contents: CNN encoders(VGG16,InceptionV3,MobileNet,ResNet),
# RNN decoders(stacked LSTM/GRU with bidirection, attention mechanism with various score functions, 
# GNMT:Google's Neural Machine Translation model), 
# Evaluation Metrics(BLEU,CIDEr,METEOR), 
# Datasets(Flickr8k, Flickr30k, COCO), 
# Django Web with best performance model.

import tensorflow as tf
# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt
# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
import time
import json
from tqdm import tqdm
import glob
import datetime
import gc
import pickle
from PIL import Image
# BLEU: Bilingual Evaluation Understudy
# evaluating a generated sentence to a referene sentence
# A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0.
# paper: https://www.aclweb.org/anthology/P02-1040.pdf
# n-grams
# Andrew Ng deeplearning.ai https://www.youtube.com/watch?v=DejHQYAGb7Q&t=2s
# more packages:
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from cider import Cider

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("physical_devices-------------", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

class get_datasets_class:
    def __init__(self,method,dataset_name,feature_extracted):
        self.feature_extracted = feature_extracted
        self.dataset_name = dataset_name
        self.method = method
        # Choose the top 5000 words from the vocabulary
        top_k = 5000
        image_batch = 16
        # Feel free to change these parameters according to your system's configuration
        BATCH_SIZE = 64
        BUFFER_SIZE = 1000
        Original_Path = '../Image_captioning_with_visual_attention'
        self.top_k = top_k
        self.image_batch = image_batch
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.Original_Path = Original_Path
        Flickr8k_Original_Path = '/home/lingfeng/Downloads/flicker8k-dataset/'
        self.Flickr8k_Original_Path = Flickr8k_Original_Path

    def get_coco_dataset(self):
        # COCO Dataset
        
        annotation_file = self.Original_Path + '/annotations/captions_train2014.json'

        PATH = self.Original_Path + '/train2014/'

        # Read the json file
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # Store captions and image names in vectors
        all_captions = []
        all_img_name_vector = []

        for annot in annotations['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

            all_img_name_vector.append(full_coco_image_path)
            all_captions.append(caption)

        # Shuffle captions and image_names together
        # Set a random state
        train_captions, img_name_vector = shuffle(all_captions,
                                                all_img_name_vector,
                                                random_state=1)
        return train_captions, img_name_vector

    def load_image(self,image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        if(self.method=='InceptionV3'):
            image_size = 299
            img = tf.image.resize(img, (image_size, image_size))
            img = tf.keras.applications.inception_v3.preprocess_input(img)
        elif(self.method=='MobileNetV2'):
            image_size = 224
            img = tf.image.resize(img, (image_size, image_size))
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        elif(self.method=='ResNet50'):
            image_size = 256
            img = tf.image.resize(img, (image_size, image_size))
            img = tf.keras.applications.resnet50.preprocess_input(img)
        elif(self.method=='VGG16'):
            image_size = 224
            img = tf.image.resize(img, (image_size, image_size))
            print("shape:",img)
            img = tf.keras.applications.vgg16.preprocess_input(img)
        else:
            print("wrong model")
        return img, image_path

    def extract_image_features(self,img_name_vector):
        if(self.method=='InceptionV3'):
            image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
            temp = tf.zeros([229,229,3])  # Or tf.zeros
            tf.keras.applications.inception_v3.preprocess_input(temp)
        elif(self.method=='MobileNetV2'):
            image_model = tf.keras.applications.MobileNetV2(include_top=False,
                                                    weights='imagenet')
            temp = tf.zeros([224,224,3])  # Or tf.zeros
            tf.keras.applications.mobilenet_v2.preprocess_input(temp)
        elif(self.method=='ResNet50'):
            image_model = tf.keras.applications.ResNet50(include_top=False,
                                                    weights='imagenet')
            temp = tf.zeros([256,256,3])  # Or tf.zeros
            tf.keras.applications.resnet50.preprocess_input(temp)
        elif(self.method=='VGG16'):
            image_model = tf.keras.applications.VGG16(include_top=False,
                                                    weights='imagenet')
            temp = tf.zeros([224,224,3])  # Or tf.zeros
            tf.keras.applications.vgg16.preprocess_input(temp)
        else:
            print("wrong model")
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output
        print("output_shape:",hidden_layer)

        image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

        # Get unique images
        encode_train = sorted(set(img_name_vector))

        # Feel free to change batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        
        image_dataset = image_dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.image_batch)

        for img, path in tqdm(image_dataset):
            batch_features = image_features_extract_model(img)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())

    # Find the maximum length of any caption in our dataset
    def calc_max_length(self,tensor):
        return max(len(t) for t in tensor)

    def tokenization(self,train_captions,img_name_vector):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.top_k,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        tokenizer.fit_on_texts(train_captions)
        train_seqs = tokenizer.texts_to_sequences(train_captions)

        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'

        # Create the tokenized vectors
        train_seqs = tokenizer.texts_to_sequences(train_captions)

        # Calculates the max_length, which is used to store the attention weights
        max_length = self.calc_max_length(train_seqs)

        # Pad each vector to the max_length of the captions
        # If you do not provide a max_length value, pad_sequences calculates it automatically
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(sequences=train_seqs, maxlen=max_length, padding='post')

        # Create training and validation sets using an 80-20 split
        img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                            cap_vector,
                                                                            test_size=0.2,
                                                                            random_state=0)
        tokenizer_name = "./tokenizer/"+self.dataset_name+"_tokenizer.pickle"
        # saving
        with open(tokenizer_name, 'wb') as f:
            pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        return img_name_train, img_name_val, cap_train, cap_val, tokenizer, max_length

    def map_func(self,img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8')+'.npy')
        return img_tensor, cap

    def create_dataset(self,img_name_train,cap_train):
        num_steps = len(img_name_train) // self.BATCH_SIZE

        dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

        # Use map to load the numpy files in parallel
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(
                self.map_func, [item1, item2], [tf.float32, tf.int32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Shuffle and batch
        dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
    
    # load doc into memory
    def load_doc(self,filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text

    # load a pre-defined list of photo identifiers
    def load_set(self,filename):
        doc = self.load_doc(filename)
        dataset = list()
        # process line by line
        for line in doc.split('\n'):
            # skip empty lines
            if len(line) < 1:
                continue
            # get the image identifier
            identifier = line.split('.')[0]
            dataset.append(identifier)
        return set(dataset)

    # load clean descriptions into memory
    def load_clean_descriptions(self,filename, dataset):
        # load document
        doc = self.load_doc(filename)
        # descriptions = dict()
        descriptions = []
        img_name = []
        for line in doc.split('\n'):
            # split line by white space
            tokens = line.split()
            # split id from description
            image_id, image_desc = tokens[0], tokens[1:]
            # skip images not in the set
            if image_id in dataset:
                # create list
                img_name.append(self.Flickr8k_Original_Path + 'Flickr8k_Dataset/Flicker8k_Dataset/' + str(image_id) + '.jpg')
                # wrap description in tokens
                desc = '<start> ' + ' '.join(image_desc) + ' <end>'
                # store
                descriptions.append(desc)
        img_name,descriptions = shuffle(img_name,descriptions,random_state=1)
        return img_name,descriptions

    def get_Flicker8k_datasets(self):
        # Below path contains all the images
        images = self.Flickr8k_Original_Path + 'Flickr8k_Dataset/Flicker8k_Dataset/'
        # Create a list of all image names in the directory
        img = glob.glob(images + '*.jpg')

        filename = self.Flickr8k_Original_Path + 'Flickr8k_text/Flickr_8k.trainImages.txt'
        train = self.load_set(filename)

        # descriptions
        train_img, train_descriptions = self.load_clean_descriptions(self.Flickr8k_Original_Path + 'Pickle/descriptions.txt', train)
        # train_img is the list of image location, 
        # original train_descriptions is the dictionary with key is image_id without .jpg appendix and value is the list of captions

        filename = self.Flickr8k_Original_Path + 'Flickr8k_text/Flickr_8k.testImages.txt'
        test = self.load_set(filename)
    
        test_img,test_descriptions = self.load_clean_descriptions(self.Flickr8k_Original_Path + 'Pickle/descriptions.txt', test)
        return train_img, train_descriptions, test_img,test_descriptions
    
    def call(self):
        if(self.dataset_name=='coco'):
            train_captions, img_name_vector = self.get_coco_dataset()
            if(self.feature_extracted==False):
                self.extract_image_features(img_name_vector)
            img_name_train, img_name_val, cap_train, cap_val,tokenizer,max_length = self.tokenization(train_captions,img_name_vector)

        if(self.dataset_name=='flickr8k'):
            train_img, train_descriptions, test_img,test_descriptions = self.get_Flicker8k_datasets()
            if(self.feature_extracted==False):
                self.extract_image_features(train_img)
                self.extract_image_features(test_img)
            img_name_train, img_name_val, cap_train, cap_val,tokenizer,max_length = self.tokenization(train_descriptions+test_descriptions,train_img+test_img)
        num_steps = len(img_name_train) //self.BATCH_SIZE
        train_dataset = self.create_dataset(img_name_train,cap_train)
        # val_dataset = self.create_dataset(img_name_val,cap_val)
        return train_dataset, img_name_val,cap_val, tokenizer, num_steps,max_length

# GRU with Bahdanau Attention
# Attention Model: Andrew Ng. https://www.youtube.com/watch?v=quoGRI-1l0A
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.lstm = tf.keras.layers.LSTM(self.units,
                                    recurrent_initializer='glorot_uniform',dropout=0.4)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def gru_call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def lstm_call(self, x, features):

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        processed_feature = tf.reduce_mean(features, axis=1)
        x = tf.concat([tf.expand_dims(processed_feature, 1), x], axis=-1)

        # passing the concatenated vector to the 3-stacked LSTM
        output = self.lstm(x)
        output = self.lstm(tf.expand_dims(output, 1))
        output = self.lstm(tf.expand_dims(output, 1))

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(tf.expand_dims(output, 1))

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class model_training():
    def __init__(self,dataset_name,model_name,encoder_name,dataset,img_name_val,cap_val,tokenizer,num_steps,max_length):
        # Shape of the vector extracted from InceptionV3 is (64, 2048)
        # These two variables represent that vector shape
        self.img_name_val = img_name_val
        self.cap_val = cap_val
        self.max_length = max_length
        self.encoder_name = encoder_name
        self.dataset_name = dataset_name
        self.num_steps = num_steps
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.model_name = model_name
        self.attention_features_shape = 64
        self.embedding_dim = 256
        self.units = 512
        self.top_k = 5000
        self.vocab_size = self.top_k + 1
        if(self.dataset_name=='coco'):
            self.epoch_num = 30
        else:
            self.epoch_num = 100
        self.stop_threshold = 0.001

        self.encoder = CNN_Encoder(self.embedding_dim)
        self.decoder = RNN_Decoder(self.embedding_dim, self.units, self.vocab_size)

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')

        if(self.encoder_name=='InceptionV3'):
            self.features_shape = 2048
            image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
        elif(self.encoder_name=='MobileNetV2'):
            self.features_shape = 1280
            image_model = tf.keras.applications.MobileNetV2(include_top=False,
                                                    weights='imagenet')
        elif(self.encoder_name=='ResNet50'):
            self.features_shape = 2048
            image_model = tf.keras.applications.ResNet50(include_top=False,
                                                    weights='imagenet')
        elif(self.encoder_name=='VGG16'):
            self.features_shape = 512
            image_model = tf.keras.applications.VGG16(include_top=False,
                                                    weights='imagenet')
        else:
            print("wrong model")
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output
        print("output_shape:",hidden_layer)

        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                if(self.model_name=='GRU_with_Bahdanau_Attention'):
                    predictions, hidden, _ = self.decoder.gru_call(dec_input, features, hidden)
                elif(self.model_name=='three_stacked_LSTM'):
                    predictions = self.decoder.lstm_call(dec_input, features)

                loss += self.loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def call(self):

        checkpoint_path = "./checkpoints/"+self.model_name+"_"+self.encoder_name+"_"+self.dataset_name+"/train"
        ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                decoder=self.decoder,
                                optimizer = self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/'+self.model_name+"_"+self.encoder_name+"_"+self.dataset_name+'/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        start_epoch = 0
        if ckpt_manager.latest_checkpoint:
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            # restoring the latest checkpoint in checkpoint_path
            ckpt.restore(ckpt_manager.latest_checkpoint)

        # adding this in a separate cell because if you run the training cell
        # many times, the loss_plot array will be reset
        EPOCHS = self.epoch_num
        pre_loss = 0
        for epoch in range(start_epoch, EPOCHS):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, target)) in enumerate(self.dataset):
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss

                if batch % 100 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
            # storing the epoch end loss value to plot later
            loss_value = total_loss / self.num_steps
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_value, step=epoch)

            if epoch % 5 == 0:
                ckpt_manager.save()

            print ('Epoch {} Loss {:.6f}'.format(epoch + 1,loss_value))
            print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            # if((pre_loss-loss_value)<self.stop_threshold):
            #     break
            # else:
            #     pre_loss = loss_value
            pre_loss = loss_value

    def load_image(self,image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        if(self.encoder_name=='InceptionV3'):
            image_size = 299
            img = tf.image.resize(img, (image_size, image_size))
            img = tf.keras.applications.inception_v3.preprocess_input(img)
        elif(self.encoder_name=='MobileNetV2'):
            image_size = 224
            img = tf.image.resize(img, (image_size, image_size))
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        elif(self.encoder_name=='ResNet50'):
            image_size = 256
            img = tf.image.resize(img, (image_size, image_size))
            img = tf.keras.applications.resnet50.preprocess_input(img)
        elif(self.encoder_name=='VGG16'):
            image_size = 224
            img = tf.image.resize(img, (image_size, image_size))
            print("shape:",img)
            img = tf.keras.applications.vgg16.preprocess_input(img)
        else:
            print("wrong model")
        return img, image_path
    
    def evaluate_preprocessing_single_image(self,image,predict_tag=False):
        if(predict_tag==True and self.model_name=='GRU_with_Bahdanau_Attention'):
            attention_plot = np.zeros((self.max_length, self.attention_features_shape))
        hidden = self.decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(self.load_image(image)[0], 0)
        img_tensor_val = self.image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = self.encoder(img_tensor_val)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = ''

        for i in range(self.max_length):
            if(self.model_name=='GRU_with_Bahdanau_Attention'):
                predictions, hidden, attention_weights = self.decoder.gru_call(dec_input, features, hidden)
                if(predict_tag==True):
                    attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
            elif(self.model_name=='three_stacked_LSTM'):
                predictions = self.decoder.lstm_call(dec_input, features)

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            

            if self.tokenizer.index_word[predicted_id] == '<end>':
                if(predict_tag==True and self.model_name=='GRU_with_Bahdanau_Attention'):
                    return result[:-1], attention_plot
                else:
                    return result[:-1]
            
            result = result + self.tokenizer.index_word[predicted_id] + ' '
            dec_input = tf.expand_dims([predicted_id], 0)

        if(predict_tag==True and self.model_name=='GRU_with_Bahdanau_Attention'):
            attention_plot = attention_plot[:len(result), :]
            return result[:-1], attention_plot
        else:
            return result[:-1]

    def plot_attention(self, image, result, attention_plot, name):
        with open(name+"predict_caption_result.txt",'w+') as f:
            f.write(result)
        temp_image = np.array(Image.open(image))
        fig = plt.figure(figsize=(10, 10))
        result = result.split(' ')
        len_result = len(result)
        for l in range(len_result):
            temp_att = np.resize(attention_plot[l], (8, 8))
            ax = fig.add_subplot(len_result//2, len_result//2, l+1)
            ax.set_title(result[l])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        # plt.show()
        plt.savefig(name+"attention_plot.png")

    def evaluate_preprocessing(self):
        result_file_name_base = "./prediction_results/"+self.model_name+"_"+self.encoder_name+"_"+self.dataset_name
        real_file = result_file_name_base+"_real.txt"
        predict_file = result_file_name_base+"_predict.txt"
        r_f = open(real_file,'w+')
        p_f = open(predict_file,'w+')
        for img_id in tqdm(range(len(self.img_name_val))):
            image = self.img_name_val[img_id]
            real_caption = ' '.join([self.tokenizer.index_word[i] for i in self.cap_val[img_id] if i not in [0]])
            predict_caption = self.evaluate_preprocessing_single_image(image)
            r_f.write(real_caption + ' \n')
            p_f.write("<start> " + predict_caption + ' <end> \n')
        r_f.close()
        p_f.close()
    
    def predict_the_caption(self,img):
        checkpoint_path = "./checkpoints/"+self.model_name+"_"+self.encoder_name+"_"+self.dataset_name+"/train"
        ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                decoder=self.decoder)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        name = "./final_results/"
        if(self.model_name=='GRU_with_Bahdanau_Attention'):
            result, attention_plot = self.evaluate_preprocessing_single_image(img,predict_tag=True)
            self.plot_attention(img, result, attention_plot, name)
        else:
            result = self.evaluate_preprocessing_single_image(img,predict_tag=True)
            with open(name+"predict_caption_result.txt") as f:
                f.write(result)

class evaluation():
    def __init__(self,model_name,encoder_name,dataset_name):
        self.model_name=model_name
        self.encoder_name=encoder_name
        self.dataset_name=dataset_name
        result_file_name_base = "./prediction_results/"+self.model_name+"_"+self.encoder_name+"_"+self.dataset_name
        self.real_file = result_file_name_base+"_real.txt"
        self.predict_file = result_file_name_base+"_predict.txt"

        self.references = []
        self.candidates = []
        f_real = open(self.real_file,'r')
        f_predict = open(self.predict_file,'r')
        for real_caption, predict_caption in zip(f_real,f_predict):
            self.references.append([real_caption.split(' ')[1:-2]])
            self.candidates.append(predict_caption.split(' ')[1:-2])

        result_file_name = "./evaluation_results/"+self.model_name+"_"+self.encoder_name+"_"+self.dataset_name+'.txt'
        self.f = open(result_file_name,'w+')
        self.f.write("model_name: "+self.model_name +'\n')
        self.f.write("encoder_name: "+self.encoder_name +'\n')
        self.f.write("dataset_name: "+self.dataset_name +'\n\n')
    
    def blue_score_call(self):
        gram_1_blue_score = corpus_bleu(self.references,self.candidates,weights=(1, 0, 0, 0))
        gram_2_blue_score = corpus_bleu(self.references,self.candidates,weights=(0, 1, 0, 0))
        gram_3_blue_score = corpus_bleu(self.references,self.candidates,weights=(0, 0, 1, 0))
        gram_4_blue_score = corpus_bleu(self.references,self.candidates,weights=(0, 0, 0, 1))
        comulative_blue_score = corpus_bleu(self.references,self.candidates,weights=(0.4, 0.3, 0.2, 0.1))

        self.f.write("1-gram blue score: " + str(gram_1_blue_score) + '\n')
        self.f.write("2-gram blue score: " + str(gram_2_blue_score) + '\n')
        self.f.write("3-gram blue score: " + str(gram_3_blue_score) + '\n')
        self.f.write("4-gram blue score: " + str(gram_4_blue_score) + '\n')
        self.f.write("comulative blue score with weights (4,3,2,1): " + str(comulative_blue_score) + '\n\n')

    def CIDEr_call(self):
        cider_instance = Cider()
        f_real = open(self.real_file,'r')
        f_predict = open(self.predict_file,'r')
        references = []
        candidates = []
        for real_caption, predict_caption in zip(f_real,f_predict):
            references.append([real_caption])
            candidates.append(predict_caption)
        cider_score = cider_instance.compute_score(candidates,references)
        self.f.write("CIDEr score: " + str(cider_score) + '\n\n')

    def METEOR_call(self):
        meteor_score_value = []
        f_real = open(self.real_file,'r')
        f_predict = open(self.predict_file,'r')
        for real_caption, predict_caption in zip(f_real,f_predict):
            meteor_score_value.append(single_meteor_score(real_caption,predict_caption))
        self.f.write("METEOR score: " + str(np.mean(meteor_score_value)) + '\n\n')
        self.f.close()

def training():
    # dataset_name_list = ['flickr8k','coco']
    model_name_list = ['GRU_with_Bahdanau_Attention','three_stacked_LSTM']
    encoder_name_list = ['InceptionV3','MobileNetV2','ResNet50','VGG16']

    dataset_name_list = ['flickr8k']
    # model_name_list = ['GRU_with_Bahdanau_Attention']
    # encoder_name_list = ['InceptionV3']

    for dataset_name in dataset_name_list:
        for encoder_name in encoder_name_list:
            get_datasets_class_instance = get_datasets_class(encoder_name,dataset_name,False)
            train_dataset, img_name_val,cap_val,tokenizer,num_steps,max_length = get_datasets_class_instance.call()
            print("max_length",max_length)
            for model_name in model_name_list:
                if(model_name == 'GRU_with_Bahdanau_Attention' and dataset_name == 'flickr8k' and encoder_name == 'InceptionV3'):
                    continue
                if(model_name == 'GRU_with_Bahdanau_Attention' and dataset_name == 'flickr8k' and encoder_name == 'MobileNetV2'):
                    continue
                if(model_name == 'three_stacked_LSTM' and dataset_name == 'flickr8k' and encoder_name == 'InceptionV3'):
                    continue
                model_training_instance = model_training(dataset_name,model_name,encoder_name,train_dataset,img_name_val,cap_val,tokenizer,num_steps,max_length)
                model_training_instance.call()
                model_training_instance.evaluate_preprocessing()
                evaluation_instance = evaluation(model_name,encoder_name,dataset_name)
                evaluation_instance.blue_score_call()
                evaluation_instance.CIDEr_call()
                evaluation_instance.METEOR_call()
                print(dir())
                # del BahdanauAttention,CNN_Encoder,RNN_Decoder,model_training_instance
                del model_training_instance, evaluation_instance
                gc.collect()
                time.sleep(60) #sleep 1 minute
            del get_datasets_class_instance,train_dataset,img_name_val,cap_val,tokenizer,num_steps,max_length
            gc.collect()
    # print(dir())
    print("Finished")

def prediction(img):
    dataset_name = 'flickr8k'
    model_name = 'GRU_with_Bahdanau_Attention'
    encoder_name = 'InceptionV3'
    tokenizer_name = "./tokenizer/"+dataset_name+"_tokenizer.pickle"
    # loading
    with open(tokenizer_name, 'rb') as f:
        tokenizer = pickle.load(f)
    # max_length in the dataset is 34
    max_length = 30
    model_training_instance = model_training(dataset_name,model_name,encoder_name,None,None,None,tokenizer,None,max_length)
    model_training_instance.predict_the_caption(img)
    if(model_name == 'GRU_with_Bahdanau_Attention'):
        return True,dataset_name,model_name,encoder_name
    else:
        return False,dataset_name,model_name,encoder_name

if __name__ == "__main__":
    training()
    # img = "../test_img/guitar_man.jpg"
    # img = "../test_img/luozhixiang.jpg"
    # prediction(img)


