import torch
from torch.utils.data import Dataset
import jieba
import pandas as pd
import gensim
import numpy as np

def read_csv(file_path):
    df = pd.read_csv(file_path)
    texts = df['文本'].tolist()
    features = df['分类'].tolist()
    emotions = [1 if emotion == 'positive' else 0 for emotion in df['情感'].tolist()]
    return texts, features, emotions

class TextDataset(Dataset):
    def __init__(self, texts, features, emotions, tokenizer, vocab, feature_vocab):
        self.texts = texts
        self.features = features
        self.emotions = emotions
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.feature_vocab = feature_vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        tokens = self.tokenizer(self.texts[index])
        text_tensor = torch.tensor([self.vocab[token] for token in tokens], dtype=torch.long)
        feature_tensor = torch.tensor(self.feature_vocab[self.features[index]], dtype=torch.long)
        emotion_tensor = torch.tensor(self.emotions[index], dtype=torch.long)
        return text_tensor, feature_tensor, emotion_tensor

def build_vocab_from_iterator(iterator):
    vocab = {}
    for tokens in iterator:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def build_feature_vocab(features):
    feature_vocab = {}
    for feature in features:
        if feature not in feature_vocab:
            feature_vocab[feature] = len(feature_vocab)
    return feature_vocab

def get_tokenizer():
    return lambda text: list(jieba.cut(text))

def load_embeddings(embedding_file, vocab):
    pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    embedding_matrix = np.zeros((len(vocab), pretrained_model.vector_size))

    for word, idx in vocab.items():
        if word in pretrained_model:
            embedding_matrix[idx] = pretrained_model[word]

    return torch.tensor(embedding_matrix, dtype=torch.float32)
