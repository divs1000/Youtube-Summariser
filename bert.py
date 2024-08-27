#pip install emoji
#pip install tensorflow-text==2.13.*
#pip install tf-models-official==2.13.*
import emoji
import os
import googleapiclient.discovery
import re
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

commentlist=[]

def remove_emojis(text):
    return emoji.replace_emoji(text,replace = '')


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    text = re.sub(r'@\w+',' ',text) #Removes usernames/names starting with @
    # Remove multiple dots and replace them with a single dot
    text = re.sub(r'\.+', '.', text)
    # Remove punctuation except for periods
    text = re.sub(r'[^\w\s.]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
    

def main():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "" #Give the youtube data api developer key here

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="snippet,replies",
        maxResults = 100,
        textFormat='plainText',
        videoId="LHCob76kigA" #Give the video id here.
    )

    response = request.execute()
    for j in range(0,100):
      listing=response['items'][j]['snippet']['topLevelComment']['snippet']['textDisplay']
      split_list = listing.split(';')
      count=0
      for i in split_list:
        text=remove_emojis(i)
        text=preprocess_text(text)
        split_list[count]=text
        count=count+1
      commentlist.append(split_list[0])
      print(*split_list,sep="\n")
if __name__ == "__main__":
    main()

tfhub_handle_encoder='https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2'  #Changing the bert model and preprocess would imply training the neural network again.
tfhub_handle_preprocess='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net=tf.keras.layers.Dense(56,activation='relu')(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)
  
epochs = 5
steps_per_epoch = 625 #Cardinality of the train_ds
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
                                          
new_model = build_classifier_model()
new_model.load_weights('/Downloads/Bert_Sentiment (latest)/model.weights.h5') #Give the file path for loading the model weights here.

new_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)
                         
                         
positive=[]
negative=[]
for i in commentlist:
  temp=[]
  temp.append(i)
  value=tf.sigmoid(new_model.predict(x=temp))
  if(value<=0.5):
    negative.append(temp[0])
  else:
    positive.append(temp[0])
    
print(positive)
print(negative)
print(tf.sigmoid(new_model.predict(x=[''])))
print("positive=",float(len(positive)/100))
print("negative=",float(len(negative)/100))
