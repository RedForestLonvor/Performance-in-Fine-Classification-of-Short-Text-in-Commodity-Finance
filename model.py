import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention

class TextClassificationModel_BiLSTM_CNN_MHA(nn.Module):
    def __init__(self, vocab_size, feature_size, embed_dim, lstm_hidden_dim, attn_heads, num_classes, feature_classes, embeddings=None, kernel_size=3, num_filters=64):
        super(TextClassificationModel_BiLSTM_CNN_MHA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.name = 'TextClassificationModel_BiLSTM_CNN_MHA'
        if embeddings is not None:
            self.embedding.weight = nn.Parameter(embeddings)
            self.embedding.weight.requires_grad = False
        self.feature_embedding = nn.Embedding(feature_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.conv1d = nn.Conv1d(lstm_hidden_dim * 2, num_filters, kernel_size)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.self_attn = MultiheadAttention(num_filters, attn_heads)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_filters, lstm_hidden_dim)
        self.fc2_emotion = nn.Linear(lstm_hidden_dim, num_classes)
        self.fc2_feature = nn.Linear(lstm_hidden_dim, feature_classes)

    def forward(self, text, features):
        embedded_text = self.embedding(text)
        embedded_features = self.feature_embedding(features).unsqueeze(1)
        embedded = torch.cat((embedded_text, embedded_features), dim=1)
        lstm_output, _ = self.lstm(embedded)
        cnn_input = lstm_output.permute(0, 2, 1)
        cnn_output = F.relu(self.conv1d(cnn_input))
        max_pool_output = self.max_pool(cnn_output).permute(0, 2, 1)
        attn_output, _ = self.self_attn(max_pool_output, max_pool_output, max_pool_output)
        feature_vector = torch.mean(attn_output, dim=1)
        feature_vector = self.dropout(feature_vector)
        fc1_output = F.relu(self.fc1(feature_vector))
        fc1_output = self.dropout(fc1_output)
        emotion_logits = self.fc2_emotion(fc1_output)
        feature_logits = self.fc2_feature(fc1_output)
        return emotion_logits, feature_logits


class TextClassificationModel_BiLSTM_MHA(nn.Module):
    def __init__(self, vocab_size, feature_size, embed_dim, lstm_hidden_dim, attn_heads, num_classes, feature_classes, embeddings=None):
        super(TextClassificationModel_BiLSTM_MHA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.name = 'TextClassificationModel_BiLSTM_MHA'
        if embeddings is not None:
            self.embedding.weight = nn.Parameter(embeddings)
            self.embedding.weight.requires_grad = False
        self.feature_embedding = nn.Embedding(feature_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.self_attn = MultiheadAttention(lstm_hidden_dim * 2, attn_heads)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim)
        self.fc2_emotion = nn.Linear(lstm_hidden_dim, num_classes)
        self.fc2_feature = nn.Linear(lstm_hidden_dim, feature_classes)

    def forward(self, text, features):
        embedded_text = self.embedding(text)
        embedded_features = self.feature_embedding(features).unsqueeze(1)
        embedded = torch.cat((embedded_text, embedded_features), dim=1)
        lstm_output, _ = self.lstm(embedded)
        attn_output, _ = self.self_attn(lstm_output, lstm_output, lstm_output)
        feature_vector = torch.mean(attn_output, dim=1)
        feature_vector = self.dropout(feature_vector)
        fc1_output = F.relu(self.fc1(feature_vector))
        fc1_output = self.dropout(fc1_output)
        emotion_logits = self.fc2_emotion(fc1_output)
        feature_logits = self.fc2_feature(fc1_output)
        return emotion_logits, feature_logits

class TextClassificationModel_CNN_MHA(nn.Module):
    def __init__(self, vocab_size, feature_size, embed_dim, attn_heads, num_classes, feature_classes, embeddings=None, kernel_size=3, num_filters=64):
        super(TextClassificationModel_CNN_MHA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.name = 'TextClassificationModel_CNN_MHA'
        if embeddings is not None:
            self.embedding.weight = nn.Parameter(embeddings)
            self.embedding.weight.requires_grad = False
        self.feature_embedding = nn.Embedding(feature_size, embed_dim)
        self.conv1d = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.self_attn = MultiheadAttention(num_filters, attn_heads)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_filters, embed_dim)
        self.fc2_emotion = nn.Linear(embed_dim, num_classes)
        self.fc2_feature = nn.Linear(embed_dim, feature_classes)

    def forward(self, text, features):
        embedded_text = self.embedding(text)
        embedded_features = self.feature_embedding(features).unsqueeze(1)
        embedded = torch.cat((embedded_text, embedded_features), dim=1)
        cnn_output = F.relu(self.conv1d(embedded.permute(0,2,1)))
        max_pool_output = self.max_pool(cnn_output).permute(0, 2, 1)
        attn_output, _ = self.self_attn(max_pool_output, max_pool_output, max_pool_output)
        feature_vector = torch.mean(attn_output, dim=1)
        feature_vector = self.dropout(feature_vector)
        fc1_output = F.relu(self.fc1(feature_vector))
        fc1_output = self.dropout(fc1_output)
        emotion_logits = self.fc2_emotion(fc1_output)
        feature_logits = self.fc2_feature(fc1_output)
        return emotion_logits, feature_logits



class TextClassificationModel_GRU_MHA(nn.Module):
    def __init__(self, vocab_size, feature_size, embed_dim, gru_hidden_dim, attn_heads, num_classes, feature_classes, embeddings=None):
        super(TextClassificationModel_GRU_MHA, self).__init__()
        self.name = 'TextClassificationModel_GRU_MHA'
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embeddings is not None:
            self.embedding.weight = nn.Parameter(embeddings)
            self.embedding.weight.requires_grad = False
        self.feature_embedding = nn.Embedding(feature_size, embed_dim)
        self.gru = nn.GRU(embed_dim, gru_hidden_dim, batch_first=True)
        self.self_attn = MultiheadAttention(gru_hidden_dim, attn_heads)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(gru_hidden_dim, gru_hidden_dim)
        self.fc2_emotion = nn.Linear
        self.fc2_emotion = nn.Linear(gru_hidden_dim, num_classes)
        self.fc2_feature = nn.Linear(gru_hidden_dim, feature_classes)

    def forward(self, text, features):
        embedded_text = self.embedding(text)
        embedded_features = self.feature_embedding(features).unsqueeze(1)
        embedded = torch.cat((embedded_text, embedded_features), dim=1)
        gru_output, _ = self.gru(embedded)
        attn_output, _ = self.self_attn(gru_output, gru_output, gru_output)
        feature_vector = torch.mean(attn_output, dim=1)
        feature_vector = self.dropout(feature_vector)
        fc1_output = F.relu(self.fc1(feature_vector))
        fc1_output = self.dropout(fc1_output)
        emotion_logits = self.fc2_emotion(fc1_output)
        feature_logits = self.fc2_feature(fc1_output)
        return emotion_logits, feature_logits

class TextClassificationModel_BiGRU_MHA(nn.Module):
    def __init__(self, vocab_size, feature_size, embed_dim, gru_hidden_dim, attn_heads, num_classes, feature_classes, embeddings=None):
        super(TextClassificationModel_BiGRU_MHA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.name = 'TextClassificationModel_BiGRU_MHA'
        if embeddings is not None:
            self.embedding.weight = nn.Parameter(embeddings)
            self.embedding.weight.requires_grad = False
        self.feature_embedding = nn.Embedding(feature_size, embed_dim)
        self.gru = nn.GRU(embed_dim, gru_hidden_dim, batch_first=True, bidirectional=True)
        self.self_attn = MultiheadAttention(gru_hidden_dim * 2, attn_heads)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(gru_hidden_dim * 2, gru_hidden_dim)
        self.fc2_emotion = nn.Linear(gru_hidden_dim, num_classes)
        self.fc2_feature = nn.Linear(gru_hidden_dim, feature_classes)

    def forward(self, text, features):
        embedded_text = self.embedding(text)
        embedded_features = self.feature_embedding(features).unsqueeze(1)
        embedded = torch.cat((embedded_text, embedded_features), dim=1)
        gru_output, _ = self.gru(embedded)
        attn_output, _ = self.self_attn(gru_output, gru_output, gru_output)
        feature_vector = torch.mean(attn_output, dim=1)
        feature_vector = self.dropout(feature_vector)
        fc1_output = F.relu(self.fc1(feature_vector))
        fc1_output = self.dropout(fc1_output)
        emotion_logits = self.fc2_emotion(fc1_output)
        feature_logits = self.fc2_feature(fc1_output)
        return emotion_logits, feature_logits

class TextClassificationModel_BiGRU(nn.Module):
    def __init__(self, vocab_size, feature_size, embed_dim, gru_hidden_dim, attn_heads, num_classes,feature_classes, embeddings=None):
        super(TextClassificationModel_BiGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.name = 'TextClassificationModel_BiGRU'
        if embeddings is not None:
            self.embedding.weight = nn.Parameter(embeddings)
            self.embedding.weight.requires_grad = False
        self.feature_embedding = nn.Embedding(feature_size, embed_dim)
        self.gru = nn.GRU(embed_dim, gru_hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(gru_hidden_dim * 2, gru_hidden_dim)
        self.fc2_emotion = nn.Linear(gru_hidden_dim, num_classes)
        self.fc2_feature = nn.Linear(gru_hidden_dim, feature_classes)

    def forward(self, text, features):
        embedded_text = self.embedding(text)
        embedded_features = self.feature_embedding(features).unsqueeze(1)
        embedded = torch.cat((embedded_text, embedded_features), dim=1)
        gru_output, _ = self.gru(embedded)
        feature_vector = torch.mean(gru_output, dim=1)
        feature_vector = self.dropout(feature_vector)
        fc1_output = F.relu(self.fc1(feature_vector))
        fc1_output = self.dropout(fc1_output)
        emotion_logits = self.fc2_emotion(fc1_output)
        feature_logits = self.fc2_feature(fc1_output)
        return emotion_logits, feature_logits
