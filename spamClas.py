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
drive.mount('/content/gdrive')
from pickle import load
np.random.seed(0)

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

print(torch.cuda.get_device_name(0))

##  PREPROCESS DATASET

#df = pd.read_csv("dataset/spam2.csv", names=['Category', 'Message'])
#df = pd.read_csv("dataset/spam2.csv")

#messages = df.Message.values
#labels = df.Category.values

#print(type(messages))
#print(messages[0])
#print(labels[0])

erase_first_word = lambda s : s[9:]
df = pd.read_csv("dataset/emails1.csv")

df['text'] = df['text'].apply(erase_first_word)

sentences = df.text.values 
labels = df.spam.values

#print(sentences[0])
#print(sentences.size)
#print(labels[0])
#print(labels.size)

erase_first_word = lambda s : s[9:]
df2 = pd.read_csv("dataset/spam2.csv")

sentences2 = df2.Message.values 
labels2 = df2.Category.values

messages = np.append(sentences, sentences2)
labels = np.append(labels, labels2)

sentences = ["[CLS] " + m + " [SEP]" for m in messages]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case= True)
tokenized_texts = [tokenizer.tokenize(s) for s in sentences]
inputs_ids = [tokenizer.convert_tokens_to_ids(tt) for tt in tokenized_texts]
#print(tokenized_texts[0])
#print(inputs_ids[0])
MAX_LEN = 128
input_ids = pad_sequences(inputs_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
attention_masks = []
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)


train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                            random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

## DEFINE MODELS

models = {1 : {"name": "bert-base-uncased",
               "param_no_decay": ['bias', 'LayerNorm.weight'],
               "learning_rate": 2e-5,
               "eps": 1e-8,
               "epochs": 1,
               "num_warmup_steps" : 0},
          2 : {"name": "bert-base-uncased",
               "param_no_decay":['bias'],
               "learning_rate": 1e-5,
               "eps": 1e-9,
               "epochs": 1,
               "num_warmup_steps" : 10},
          3 : {"name": "bert-base-uncased",
               "param_no_decay":[],
               "learning_rate": 3e-5,
               "eps": 5e-8,
               "epochs": 1,
               "num_warmup_steps" : 30}
          }


##  ITERATE ON MODEL

eval_by_model = []

for model in models.items():
  print(model)
  num, model_def = model

  model = BertForSequenceClassification.from_pretrained(model_def.get('name'), num_labels=2)
  model.cuda()

  param_optimizer = list(model.named_parameters())
#  no_decay = ['bias', 'LayerNorm.weight']
  no_decay = model_def.get('param_no_decay')
  optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
      'weight_decay_rate': 0.1},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
      'weight_decay_rate': 0.0}
  ]
  epochs = model_def.get('epochs')
  #epochs = 32
  optimizer = AdamW(optimizer_grouped_parameters,
                    #lr = 5e-5, 
                    lr = model_def.get('learning_rate'), 
                    #eps = 1e-8 
                    eps = model_def.get('eps') 
                    )

  total_steps = len(train_dataloader) * epochs

  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              #num_warmup_steps = 0, 
                                              num_warmup_steps = model_def.get('num_warmup_steps'),
                                              num_training_steps = total_steps)


  ##  TRAIN MODEL

  t = [] 

  train_loss_set = []

  for _ in trange(epochs, desc="Epoch"):
    model.train()
    
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    
    for step, batch in enumerate(train_dataloader):
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch
      optimizer.zero_grad()
      outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
      loss = outputs['loss']
      train_loss_set.append(loss.item())    
      loss.backward()
      optimizer.step()

      scheduler.step()
      
      tr_loss += loss.item()
      nb_tr_examples += b_input_ids.size(0)
      nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))
      
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch
      with torch.no_grad():
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      
      logits = logits['logits'].detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      tmp_eval_accuracy = flat_accuracy(logits, label_ids)
      
      eval_accuracy += tmp_eval_accuracy
      nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

  ##  PLOT LOSS

  plt.figure(figsize=(15,8))
  plt.title("Training loss")
  plt.xlabel("Batch")
  plt.ylabel("Loss")
  plt.plot(train_loss_set)
  plt.show()

  ##  SAVE TRAINED MODEL PARAMETERS

  import os

  output_dir = '/content/gdrive/MyDrive/Machine Learning/datos/Spam/modelos/model_saveTMP'
  output_dir = output_dir + str(num)

  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  print("Saving model to %s" % output_dir)

  model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
  model_to_save.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)

  ##  EVALUATE 
  erase_first_word = lambda s : s[8:]

  #df = pd.read_csv("dataset/emails1.csv")
  #df['text'] = df['text'].apply(erase_first_word)
  #sentences = df.text.values 
  #labels = df.spam.values

  df = pd.read_csv("dataset/spam.csv",  sep=',',  encoding='latin-1')
#  print(df)
  
  sentences = df.v2.values 
  labels = df.v1.values
  labels = labels == 'spam'
  labels = 1 * labels


  sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
#  print(sentences[0])
  tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
#  print(tokenized_texts[0])

  MAX_LEN = 128

  input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# print(input_ids[0])

  attention_masks = []

  for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask) 

  prediction_inputs = torch.tensor(input_ids)
  prediction_masks = torch.tensor(attention_masks)
  prediction_labels = torch.tensor(labels)
    
  batch_size = 32

  prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
  prediction_sampler = SequentialSampler(prediction_data)
  prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

  model.eval()

  predictions , true_labels = [], []

  for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = logits['logits'].detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    predictions.append(logits)
    true_labels.append(label_ids)

#  print(np.argmax(predictions[0], axis=1).flatten()) 
#  print(true_labels[0])

  from sklearn.metrics import matthews_corrcoef
  matthews_set = []

  for i in range(len(true_labels)):
    matthews = matthews_corrcoef(true_labels[i],
                  np.argmax(predictions[i], axis=1).flatten())
    matthews_set.append(matthews)

  print(matthews_set)

  flat_predictions = [item for sublist in predictions for item in sublist]
  flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
  flat_true_labels = [item for sublist in true_labels for item in sublist]

  eval = matthews_corrcoef(flat_true_labels, flat_predictions)
  eval_by_model.append(eval)
  print("bert-base-uncased matthews_corrcoef equals ", eval )

print(eval_by_model)