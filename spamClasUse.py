!pip install -q transformers

from google.colab import drive
drive.mount('/content/drive')
import pandas as pd

#LOAD PARAMS AGAIN
try:
  import transformers
except:
  print("Installing transformers")
  !pip -qq install transformers

from transformers import BertModel, BertConfig, BertForSequenceClassification, BertTokenizer
import os

### MODEL LOADING!!  ###########################################################################################################

#output_dir = './drive/MyDrive/Machine Learning/datos/Spam/modelos/model_save/'
output_dir = '/content/drive/MyDrive/Machine Learning/datos/Spam/modelos/model_save'

if not os.path.exists(output_dir):
  print("ERROR in output_dir")


model3 = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer3 = BertTokenizer.from_pretrained(output_dir)

import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

MAX_LEN = 128

### PREDICTIONS  #####################################################################################################################

def predict(sentence):
  model3.eval()
  inputs = tokenizer(sentence, return_tensors="pt")
  etiqueta = torch.tensor([1]).unsqueeze(0)  # Batch size 1
  with torch.no_grad():
    outputs = model3(**inputs, labels=etiqueta)
  print(outputs['logits'])


print('0 Next example')
predict("average salary may be around four hundred euros but even though salary goes up every single bill people in serbia are paying becomes more expensive")   

print('\n\n0 Next example')
predict("Is that seriously how you spell his name?")   

print('\n\n1 Next example')
predict("Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030")   

print('\n\n1 Next example')
predict("Congrats! 1 year special cinema pass for 2 is yours. call 09061209465 now! C Suprman V, Matrix3, StarWars3, etc all 4 FREE! bx420-ip4-5we. 150pm. Dont miss out!")   

print('\n\n0 Next example')
predict("U don't know how stubborn I am. I didn't even want to go to the hospital. I kept telling Mark I'm not a weak sucker. Hospitals are for weak suckers.")   
