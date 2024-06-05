import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate import meteor_score
import numpy as np

wikihow_dataset = pd.read_csv("wikihow-cleaned.csv")
wikihow_dataset = wikihow_dataset[['text','summary']]
wikihow_dataset.head()


cnn_dataset = pd.read_parquet("train-00001-of-00003.parquet",engine='pyarrow')
cnn_dataset1 = pd.read_parquet("train-00000-of-00003.parquet",engine='pyarrow')
cnn_dataset2 = pd.read_parquet("train-00002-of-00003.parquet",engine='pyarrow')
combined_df = pd.concat([cnn_dataset, cnn_dataset1, cnn_dataset2], axis=0)
combined_df
combined_df.rename(columns={'article': 'text'}, inplace=True)
combined_df.rename(columns={'highlights': 'summary'}, inplace=True)
cnn_dataset = combined_df[['text','summary']]
cnn_dataset.head()

train_data_wiki, temp_data = train_test_split(wikihow_dataset, test_size=0.3, random_state=42)
validation_data_wiki, test_data_wiki = train_test_split(temp_data, test_size=0.5, random_state=42)
train_data_wiki.head()

train_data_cnn, temp_data = train_test_split(cnn_dataset, test_size=0.3, random_state=42)
validation_data_cnn, test_data_cnn = train_test_split(temp_data, test_size=0.5, random_state=42)
train_data_cnn.head()

train_loader_wiki = DataLoader(train_data_wiki, batch_size=16, shuffle=True)
val_loader_wiki = DataLoader(validation_data_wiki, batch_size=16)
test_loader_wiki = DataLoader(test_data_wiki, batch_size=16)

train_loader_cnn = DataLoader(train_data_cnn, batch_size=16, shuffle=True)
val_loader_cnn = DataLoader(validation_data_cnn, batch_size=16)
test_loader_cnn = DataLoader(test_data_cnn, batch_size=16)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        encoder_outputs, hidden = self.encoder(src)
        output = self.decoder(trg, hidden, encoder_outputs)
        return output

    
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        rnn_input = torch.cat((embedded, encoder_outputs.unsqueeze(0)), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        encoder_outputs = encoder_outputs.squeeze(0)
        output = self.fc_out(torch.cat((output, encoder_outputs, embedded), dim = 1))
        return output

    
def evaluate(model, dataset):
    model.eval()
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    meteor = []
    for example in dataset:
        src = example["text"]
        trg = example["summary"]
        generated_summary = model(src, trg)
        reference_summary = trg
        rouge_scores = rouge.score(generated_summary, reference_summary)
        rouge_scores_avg = np.mean([np.mean(list(score.values())) for score in rouge_scores.values()])
        meteor_score = meteor_score.meteor_score([reference_summary], generated_summary)
        meteor.append(meteor_score)
    avg_meteor = np.mean(meteor)
    return rouge_scores_avg, avg_meteor


INPUT_DIM =  50000
OUTPUT_DIM =  50000
HID_DIM = 256
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
BATCH_SIZE = 16

enc = Encoder(INPUT_DIM, HID_DIM, HID_DIM, DEC_DROPOUT)
dec = Decoder(OUTPUT_DIM, HID_DIM, HID_DIM, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for batch in train_loader_wiki:
        src, trg = batch['text'], batch['summary']
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, :-1, :].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader_wiki)

    rouge_score, meteor_score = evaluate(model, val_loader_wiki)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {avg_loss}, ROUGE: {rouge_score}, METEOR: {meteor_score}")
    
rouge_score, meteor_score = evaluate(model, test_loader_wiki)
print(f"Test ROUGE: {rouge_score}, Test METEOR: {meteor_score}")
    
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for batch in train_loader_cnn:
        src, trg = batch['text'], batch['summary']
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, :-1, :].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader_cnn)

    rouge_score, meteor_score = evaluate(model, val_loader_cnn)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {avg_loss}, ROUGE: {rouge_score}, METEOR: {meteor_score}")


rouge_score, meteor_score = evaluate(model, test_loader_cnn)
print(f"Test ROUGE: {rouge_score}, Test METEOR: {meteor_score}")