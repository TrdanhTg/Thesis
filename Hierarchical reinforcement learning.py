from torch.utils.data import DataLoader
import nltk
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

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

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate import meteor_score
import nltk

nltk.download('wordnet')
torch, nn = try_import_torch()

def compute_intrinsic_reward(summary_tokens, text_tokens):
    return len(set(summary_tokens) & set(text_tokens)) / len(text_tokens)

def compute_extrinsic_reward(summary, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores['rougeL'].fmeasure

def calculate_rouge_scores(summary, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def calculate_meteor_score(summary, reference):
    return meteor_score.meteor_score([reference], summary)

def evaluate_summary(summary, reference):
    rouge_scores = calculate_rouge_scores(summary, reference)
    meteor = calculate_meteor_score(summary, reference)
    return {
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'meteor': meteor
    }

class TextSummarizationEnv(gym.Env):
    def __init__(self, text, summary):
        super(TextSummarizationEnv, self).__init__()
        self.text = text.split()
        self.summary = summary.split()
        self.current_summary = []
        self.done = False
        self.vocab = list(set(self.text + self.summary))
        self.action_space = spaces.Discrete(len(self.vocab))
        self.observation_space = spaces.Box(0, 1, shape=(len(self.vocab),), dtype=np.float32)
        self.state = np.zeros(len(self.vocab))

    def step(self, action):
        word = self.vocab[action]
        self.current_summary.append(word)
        self.state[self.vocab.index(word)] = 1
        intrinsic_reward = compute_intrinsic_reward(self.current_summary, self.text)
        self.done = self._check_done()
        return self.state, intrinsic_reward, self.done, {}

    def reset(self):
        self.current_summary = []
        self.done = False
        self.state = np.zeros(len(self.vocab))
        return self.state

    def _check_done(self):
        return len(self.current_summary) >= 0.2 * len(self.text)

    def render(self, mode='human'):
        print(' '.join(self.current_summary))

class HighLevelPolicy(nn.Module):
    def __init__(self, vocab_size):
        super(HighLevelPolicy, self).__init__()
        self.fc = nn.Linear(vocab_size, vocab_size)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class LowLevelPolicy(nn.Module):
    def __init__(self, vocab_size):
        super(LowLevelPolicy, self).__init__()
        self.fc = nn.Linear(vocab_size, vocab_size)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

ModelCatalog.register_custom_model("high_level_model", HighLevelPolicy)
ModelCatalog.register_custom_model("low_level_model", LowLevelPolicy)

example_text_cnn = cnn_dataset['text']
example_summary_cnn = cnn_dataset['summary']
vocab_size_cnn = len(set(example_text.split() + example_summary.split()))

example_text_wiki = wikihow_dataset['text']
example_summary_wiki = wikihow_dataset['summary']
vocab_size_wiki = len(set(example_text.split() + example_summary.split()))

high_level_config = {
    "env": TextSummarizationEnv,
    "env_config": {"text": example_text, "summary": example_summary},
    "model": {
        "custom_model": "high_level_model",
        "custom_model_config": {"vocab_size": vocab_size}
    },
    "num_workers": 1,
}

low_level_config = {
    "env": TextSummarizationEnv,
    "env_config": {"text": example_text, "summary": example_summary},
    "model": {
        "custom_model": "low_level_model",
        "custom_model_config": {"vocab_size": vocab_size}
    },
    "num_workers": 1,
}

high_level_trainer = PPOTrainer(config=high_level_config)
low_level_trainer = PPOTrainer(config=low_level_config)

NXE = 5
NREINFORCE = 15
batch_size = 16

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader_wiki):
        inputs = batch["text"] 
        targets = batch["summary"]

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader_wiki)}], Loss: {running_loss / 100}")
            running_loss = 0.0

for epoch in range(num_epochs_ce, num_epochs_reinforce):
    for batch in train_loader_wiki:
        text, summary = example_text, example_summary
        env = TextSummarizationEnv(text, summary)

        state = env.reset()
        done = False
        while not done:
            sub_goal_probs = high_level_trainer.compute_action(state)
            sub_goal = np.random.choice(range(len(sub_goal_probs)), p=sub_goal_probs)

            action_probs = low_level_trainer.compute_action(state, sub_goal)
            action = np.random.choice(range(len(action_probs)), p=action_probs)

            state, intrinsic_reward, done, _ = env.step(action)

            extrinsic_reward = compute_extrinsic_reward(' '.join(env.current_summary), summary)

            low_level_trainer.optimizer.zero_grad()
            low_level_loss = -torch.log(action_probs[action]) * intrinsic_reward
            low_level_loss.backward()
            low_level_trainer.optimizer.step()

            high_level_trainer.optimizer.zero_grad()
            high_level_loss = -torch.log(sub_goal_probs[sub_goal]) * extrinsic_reward
            high_level_loss.backward()
            high_level_trainer.optimizer.step()
            
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader_cnn):
        inputs = batch["text"] 
        targets = batch["summary"]

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader_cnn)}], Loss: {running_loss / 100}")
            running_loss = 0.0

for epoch in range(num_epochs_ce, num_epochs_reinforce):
    for batch in train_loader_cnn:
        text, summary = example_text, example_summary
        env = TextSummarizationEnv(text, summary)

        state = env.reset()
        done = False
        while not done:
            sub_goal_probs = high_level_trainer.compute_action(state)
            sub_goal = np.random.choice(range(len(sub_goal_probs)), p=sub_goal_probs)

            action_probs = low_level_trainer.compute_action(state, sub_goal)
            action = np.random.choice(range(len(action_probs)), p=action_probs)

            state, intrinsic_reward, done, _ = env.step(action)

            extrinsic_reward = compute_extrinsic_reward(' '.join(env.current_summary), summary)

            low_level_trainer.optimizer.zero_grad()
            low_level_loss = -torch.log(action_probs[action]) * intrinsic_reward
            low_level_loss.backward()
            low_level_trainer.optimizer.step()

            high_level_trainer.optimizer.zero_grad()
            high_level_loss = -torch.log(sub_goal_probs[sub_goal]) * extrinsic_reward
            high_level_loss.backward()
            high_level_trainer.optimizer.step()

wikihow_scores = []
for example in test_loader_wiki:
    text = example["text"]
    reference_summary = example["summary"]
    generated_summary = model.generate_summary(text)
    rouge_scores = rouge_scorer.score(reference_summary, generated_summary)
    meteor_score = meteor_score.meteor_score([reference_summary], generated_summary)
    wikihow_scores.append((rouge_scores, meteor_score))


cnn_scores = []
for example in test_loader_cnn:
    text = example["text"]
    reference_summary = example["summary"]
    generated_summary = model.generate_summary(text)
    rouge_scores = rouge_scorer.score(reference_summary, generated_summary)
    meteor_score = meteor_score.meteor_score([reference_summary], generated_summary)
    cnn_scores.append((rouge_scores, meteor_score))

avg_rouge_wikihow = np.mean([score[0]['rougeL'].fmeasure for score in wikihow_scores])
avg_meteor_wikihow = np.mean([score[1] for score in wikihow_scores])
avg_rouge_cnn = np.mean([score[0]['rougeL'].fmeasure for score in cnn_scores])
avg_meteor_cnn = np.mean([score[1] for score in cnn_scores])

print("WikiHow Dataset Average Scores:")
print(f"ROUGE-L: {avg_rouge_wikihow}, METEOR: {avg_meteor_wikihow}")

print("\nCNN/DailyMail Dataset Average Scores:")
print(f"ROUGE-L: {avg_rouge_cnn}, METEOR: {avg_meteor_cnn}")