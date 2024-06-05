from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate import meteor_score
from rouge_score import rouge_scorer
from datasets import load_dataset
import nltk
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')

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

class TextRankSummarizer:
    def __init__(self, text, num_sentences=3):
        self.text = text
        self.num_sentences = num_sentences
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self):
        sentences = sent_tokenize(self.text)
        word_tokens = [word_tokenize(sentence.lower()) for sentence in sentences]
        return sentences, word_tokens

    def sentence_similarity(self, sentence1, sentence2):
        all_words = list(set(sentence1 + sentence2))
        vector1 = [1 if word in sentence1 else 0 for word in all_words]
        vector2 = [1 if word in sentence2 else 0 for word in all_words]
        return cosine_similarity([vector1], [vector2])[0][0]

    def build_similarity_matrix(self, sentences, word_tokens):
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self.sentence_similarity(word_tokens[i], word_tokens[j])
        return similarity_matrix

    def summarize(self):
        summarized_text = ""
        sentences, word_tokens = self.preprocess_text()
        similarity_matrix = self.build_similarity_matrix(sentences, word_tokens)
        scores = np.sum(similarity_matrix, axis=1)
        ranked_sentences = [sentences[i] for i in np.argsort(scores)[-self.num_sentences:]]
        return ' '.join(ranked_sentences)

class Lead3Summarizer:
    def __init__(self, text):
        self.text = text

    def summarize(self):
        sentences = sent_tokenize(self.text)
        return ' '.join(sentences[:3])

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


results_textrank = []

for i in range(len(wikihow_dataset)):
    reference_summary = train_dataset.iloc[i][0]
    text = train_dataset.iloc[0][i]

    textrank_summarizer = TextRankSummarizer(text)
    textrank_summary = textrank_summarizer.summarize()

    rouge_textrank = calculate_rouge_scores(textrank_summary, reference_summary)

    meteor_textrank = calculate_meteor_score(textrank_summary, reference_summary)

    results_textrank.append({'rouge': rouge_textrank, 'meteor': meteor_textrank})


avg_results_textrank = {
    'rouge1': np.mean([r['rouge']['rouge1'] for r in results_textrank]),
    'rouge2': np.mean([r['rouge']['rouge2'] for r in results_textrank]),
    'rougeL': np.mean([r['rouge']['rougeL'] for r in results_textrank]),
    'meteor': np.mean([r['meteor'] for r in results_textrank])
}

print("TextRank Summarizer Results:")
print(avg_results_textrank)


results_lead3 = []

for i in range(len(cnn_dataset):
    reference_summary = cnn_dataset.iloc[i][0]
    text = cnn_dataset.iloc[0][i]

    lead3_summarizer = Lead3Summarizer(text)
    lead3_summary = lead3_summarizer.summarize()

    rouge_lead3 = calculate_rouge_scores(lead3_summary, reference_summary)
    meteor_lead3 = calculate_meteor_score(lead3_summary, reference_summary)

    results_textrank.append({'rouge': rouge_textrank, 'meteor': meteor_textrank})
    results_lead3.append({'rouge': rouge_lead3, 'meteor': meteor_lead3})

avg_results_lead3 = {
    'rouge1': np.mean([r['rouge']['rouge1'] for r in results_lead3]),
    'rouge2': np.mean([r['rouge']['rouge2'] for r in results_lead3]),
    'rougeL': np.mean([r['rouge']['rougeL'] for r in results_lead3]),
    'meteor': np.mean([r['meteor'] for r in results_lead3])
}

print("\nLead-3 Summarizer Results:")
print(avg_results_lead3)