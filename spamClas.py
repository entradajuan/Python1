import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

!pip install -q transformers
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
from pickle import load
np.random.seed(0)

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# Read dataset

#df = pd.read_csv("dataset/spam2.csv", names=['Category', 'Message'])
df = pd.read_csv("dataset/spam2.csv")

messages = df.Message.values
labels = df.Category.values

print(type(messages))
print(messages[0])
print(labels[0])

sentences = ["[CLS] " + m + " [SEP]" for m in messages]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case= True)
tokenized_texts = [tokenizer.tokenize(s) for s in sentences]
inputs_ids = [tokenizer.convert_tokens_to_ids(tt) for tt in tokenized_texts]
print(tokenized_texts[0])
print(inputs_ids[0])
MAX_LEN = 128
inputs_ids = pad_sequences(inputs_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
attention_masks = []
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)


train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                            random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

