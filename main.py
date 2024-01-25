import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from collections import Counter
is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device 
variable later in our code.
if is_cuda:
 device = torch.device("cuda")
 print("GPU is available")
else:
 device = torch.device("cpu")
 print("GPU not available, CPU used")
#file path of csv file
base_csv = r'H:\\study\\AI\\labs\\lab 3\\movie_data.csv'
df = pd.read_csv(base_csv)
df.head()
#splitting to train and test data
X,y = df['review'].values,df['sentiment'].values
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)
print(f'shape of train data is {x_train.shape}')
print(f'shape of test data is {x_test.shape}')
#analyzing sentiment
dd = pd.Series(y_train).value_counts()
sns.barplot(x=np.array(['negative','positive']),y=dd.values)
# plt.show()
#pre-processing strings
def preprocess_string(s):
 # Remove all non-word characters (everything except numbers and letters)
 s = re.sub(r"[^\w\s]", '', s)
 # Replace all runs of whitespaces with no space
 s = re.sub(r"\s+", '', s)
 # replace digits with no space
 s = re.sub(r"\d", '', s)
 return s
# this method for preprocessing and tokenizing text data for training tasks
def tockenize(x_train,y_train,x_val,y_val):
 word_list = [] # define empty list as word_list
 stop_words = set(stopwords.words('english'))
 # iterate each sentences and splits those texts in to word
 for sent in x_train:
 for word in sent.lower().split():
 word = preprocess_string(word)
 if word not in stop_words and word != '':
 word_list.append(word)
 corpus = Counter(word_list)
 # sorting on the basis of most common words
 corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
 # creating a dictionary which is map each word to unique integer
 onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
 # tockenize
 final_list_train,final_list_test = [],[]
 max_seq_length = 100
 for sent in x_train:
 tokenized = [onehot_dict[preprocess_string(word)] for word in 
sent.lower().split()
 if preprocess_string(word) in onehot_dict.keys()]
 # Pad or truncate the sequence to the fixed length
 tokenized = tokenized[:max_seq_length] + [0] * (max_seq_length - 
len(tokenized))
 final_list_train.append(tokenized)
 for sent in x_val:
 tokenized = [onehot_dict[preprocess_string(word)] for word in 
sent.lower().split()
 if preprocess_string(word) in onehot_dict.keys()]
 # Pad or truncate the sequence to the fixed length
 tokenized = tokenized[:max_seq_length] + [0] * (max_seq_length - 
len(tokenized))
 final_list_test.append(tokenized)
 encoded_train = [1 if label =='positive' else 0 for label in y_train]
 encoded_test = [1 if label =='positive' else 0 for label in y_val]
 return np.array(final_list_train), 
np.array(encoded_train),np.array(final_list_test), 
np.array(encoded_test),onehot_dict
x_train,y_train,x_test,y_test,vocab = tockenize(x_train,y_train,x_test,y_test)
print(f'Length of vocabulary is {len(vocab)}')
# this class basically use for get binary analyziz like positive or negative 
review.
class SentimentRNN(nn.Module):
 # this is construct method. This called while object create in this class.
 def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5):
 super(SentimentRNN,self).__init__()
 self.output_dim = output_dim
 self.hidden_dim = hidden_dim
 self.no_layers = no_layers
 self.vocab_size = vocab_size
 # embedding input layer
 self.embedding = nn.Embedding(vocab_size, embedding_dim)
 #this is the lstm layer
 self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim, 
num_layers=no_layers, batch_first=True)
 # dropout layer
 self.dropout = nn.Dropout(0.3)
 # linear and sigmoid layer
 self.fc = nn.Linear(self.hidden_dim, output_dim)
 self.sig = nn.Sigmoid()
 # this method describe the forward pass through layers
 def forward(self,x,hidden):
 batch_size = x.size(0)
 # embeddings and lstm_out
 embeds = self.embedding(x) # shape: B x S x Feature since batch = True
 #print(embeds.shape) #[50, 500, 1000]
 lstm_out, hidden = self.lstm(embeds, hidden)
 lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
 # dropout and fully connected layer
 out = self.dropout(lstm_out)
 out = self.fc(out)
 # sigmoid function
 sig_out = self.sig(out)
 # reshape to be batch_size first
 sig_out = sig_out.view(batch_size, -1)
 sig_out = sig_out[:, -1] # get last batch of labels
 # return last sigmoid output and hidden state
 return sig_out, hidden
 # this method initialize the hidden state
 def init_hidden(self, batch_size):
 ''' Initializes hidden state '''
 # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
 # initialized to zero, for hidden state and cell state of LSTM
 h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
 c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
 hidden = (h0,c0)
 return hidden
#lets create Model 1 now
#one single LSTM layer with 256 hidden units and embedding dimension 64
vocab_size = len(vocab) + 1 #extra 1 for padding
output_dim = 1
no_layers = 1
embedding_dim = 64
hidden_dim = 256
modelOne = 
SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)
#moving to gpu
modelOne.to(device)
print(modelOne)
#lets create model 2 now
#2 LSTM layers, same input as Model 1
no_layers = 2
embedding_dim = 64
hidden_dim = 256
modelTwo = 
SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)
#moving to gpu
modelTwo.to(device)
print(modelTwo)
#lets create model 3 now
#Same as Model 2, but embedding dimension 128
no_layers = 2
embedding_dim = 128
hidden_dim = 256
modelThree = 
SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)
#moving to gpu
modelThree.to(device)
print(modelThree)
#lets create model 4 now
#Same as Model 2, but embedding dimension 32
embedding_dim = 32
no_layers = 2
hidden_dim = 256
modelFour = 
SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)
#moving to gpu
modelFour.to(device)
print(modelFour)
#lets create model 5 now
#3 LSTM layers and same input as Model 2
embedding_dim = 64
hidden_dim = 256
no_layers = 3
modelFive = 
SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)
#moving to gpu
modelFive.to(device)
print(modelFive)
#lets create model 6 now
#Same as Model 2, but hidden units = 512
no_layers = 2
embedding_dim = 64
hidden_dim = 512
modelSix = 
SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)
#moving to gpu
modelSix.to(device)
print(modelSix)
###########################calculate loss and accuracy in each model 
now#####################################
lr=0.001
criterion = nn.BCELoss()
def acc(pred,label):
 pred = torch.round(pred.squeeze())
 return torch.sum(pred == label.squeeze()).item()
# dataloaders
batch_size = 50
def padding_(sentences, seq_len):
 features = np.zeros((len(sentences), seq_len),dtype=int)
 for ii, review in enumerate(sentences):
 if len(review) != 0:
 features[ii, -len(review):] = np.array(review)[:seq_len]
 return features
#we have very less number of reviews with length > 500.
#So we will consideronly those below it.
x_train_pad = padding_(x_train,500)
x_test_pad = padding_(x_test,500)
# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), 
torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))
# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
#set optimizer for each model
optimizer = torch.optim.Adam(modelSix.parameters(), lr=lr)
dir_path = r'H:\\study\\AI\\labs\\lab 3\\data\\state_dict.pt'
clip = 5
epochs = 5
valid_loss_min = np.Inf
epoch_tr_loss,epoch_vl_loss = [],[]
epoch_tr_acc,epoch_vl_acc = [],[]
for epoch in range(epochs):
 train_losses = []
 train_acc = 0.0
 modelSix.train()
 # initialize hidden state
 h = modelSix.init_hidden(batch_size)
 for inputs, labels in train_loader:
 inputs, labels = inputs.to(device), labels.to(device)
 # Creating new variables for hidden state, otherwise backprop through the 
entire training history
 h = tuple([each.data for each in h])
 modelSix.zero_grad()
 output,h = modelSix(inputs,h)
 # calculate the loss and perform backprop
 loss = criterion(output.squeeze(), labels.float())
 loss.backward()
 train_losses.append(loss.item())
 # calculating accuracy
 accuracy = acc(output,labels)
 train_acc += accuracy
 #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / 
LSTMs.
 nn.utils.clip_grad_norm_(modelSix.parameters(), clip)
 optimizer.step()
 val_h = modelSix.init_hidden(batch_size)
 val_losses = []
 val_acc = 0.0
 modelSix.eval()
 for inputs, labels in valid_loader:
 val_h = tuple([each.data for each in val_h])
 inputs, labels = inputs.to(device), labels.to(device)
 output, val_h = modelSix(inputs, val_h)
 val_loss = criterion(output.squeeze(), labels.float())
 val_losses.append(val_loss.item())
 accuracy = acc(output,labels)
 val_acc += accuracy
 epoch_train_loss = np.mean(train_losses)
 epoch_val_loss = np.mean(val_losses)
 epoch_train_acc = train_acc/len(train_loader.dataset)
 epoch_val_acc = val_acc/len(valid_loader.dataset)
 epoch_tr_loss.append(epoch_train_loss)
 epoch_vl_loss.append(epoch_val_loss)
 epoch_tr_acc.append(epoch_train_acc)
 epoch_vl_acc.append(epoch_val_acc)
 print(f'Epoch {epoch+1}')
 print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
 print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : 
{epoch_val_acc*100}')
 if epoch_val_loss <= valid_loss_min:
 torch.save(modelSix.state_dict(), dir_path)
 print('Validation loss decreased ({:.6f} --> {:.6f}). Saving 
model ...'.format(valid_loss_min,epoch_val_loss))
 valid_loss_min = epoch_val_loss
 print(25*'==')
# %%
fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 2, 1)
plt.plot(epoch_tr_acc, label='Train Acc')
plt.plot(epoch_vl_acc, label='Validation Acc')
plt.title("Accuracy")
plt.legend()
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(epoch_tr_loss, label='Train loss')
plt.plot(epoch_vl_loss, label='Validation loss')
plt.title("Loss")
plt.legend()
plt.grid()
plt.show()
#Test the model of 3.2.2, with few lines of text as input
def predict_sentiment(text):
 word_seq = np.array([vocab[preprocess_string(word)] for word in text.split()
 if preprocess_string(word) in vocab.keys()])
 word_seq = np.expand_dims(word_seq, axis=0)
 pad = torch.from_numpy(padding_(word_seq, 500))
 inputs = pad.to(device)
 batch_size = 1
 h = modelTwo.init_hidden(batch_size)
 h = tuple([each.data for each in h])
 output, h = modelTwo(inputs, h)
 prediction = output.item()
 if prediction > 0.5:
 print(f'{prediction:0.3}: Positive sentiment')
 else:
 print(f'{prediction:0.3}: Negative sentiment')
# %%
text = """I love this car.
This view is amazing.
I feel great this morning.
I am so excited about the concert.
He is my best friend
"""
predict_sentiment(text)
text="""
I do not like this car.
This view is horrible.
I feel tired this morning.
I am not looking forward to the concert.
He is my enemy
"""
predict_sentiment(text)
# %%
text= "I don't feel good. I hate this feeling of stress. It is very bad"
predict_sentiment(text)
# %%
text= "In school we learn that mistakes are bad, and we are punished for making 
them. Yet, if you look at the way humans are designed to learn, we learn by making 
mistakes. We learn to walk by falling down. If we never fell down, we would never 
walk"
predict_sentiment(text)
###OUTPUT####
# 0.51: Positive sentiment
# 0.499: Negative sentiment
# 0.507: Positive sentiment
# 0.503: Positive sentiment