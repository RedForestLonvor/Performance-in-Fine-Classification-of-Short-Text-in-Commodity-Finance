import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from dataset import read_csv, TextDataset, build_vocab_from_iterator, build_feature_vocab, get_tokenizer, load_embeddings
from model import TextClassificationModel_BiLSTM_MHA
from model import TextClassificationModel_CNN_MHA
from model import TextClassificationModel_GRU_MHA
from model import TextClassificationModel_BiGRU
from model import TextClassificationModel_BiLSTM_CNN_MHA
from model import TextClassificationModel_BiGRU_MHA

def pad_collate(batch):
    (texts, features, emotions) = zip(*batch)
    text_lengths = [len(t) for t in texts]
    text_padded = torch.zeros(len(texts), max(text_lengths), dtype=torch.long)
    for i, t in enumerate(texts):
        text_padded[i, :len(t)] = t
    return text_padded, torch.stack(features), torch.stack(emotions)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        text, feature_label, emotion_label = batch
        text = text.to(device)
        feature_label = feature_label.to(device)
        emotion_label = emotion_label.to(device)
        optimizer.zero_grad()
        emotion_output, feature_output = model(text, feature_label)
        emotion_loss = criterion(emotion_output, emotion_label)
        feature_loss = criterion(feature_output, feature_label)
        loss = emotion_loss + feature_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    emotion_correct = 0
    feature_correct = 0
    total = 0
    all_emotion_preds = []
    all_feature_preds = []

    with torch.no_grad():
        for batch in dataloader:
            text, feature_label, emotion_label = batch
            text = text.to(device)
            feature_label = feature_label.to(device)
            emotion_label = emotion_label.to(device)
            emotion_output, feature_output = model(text, feature_label)
            emotion_loss = criterion(emotion_output, emotion_label)
            feature_loss = criterion(feature_output, feature_label)
            loss = emotion_loss + feature_loss
            epoch_loss += loss.item()
            emotion_correct += (emotion_output.argmax(1) == emotion_label).sum().item()
            feature_correct += (feature_output.argmax(1) == feature_label).sum().item()
            total += emotion_label.size(0)

            emotion_preds = torch.argmax(emotion_output, dim=1)
            feature_preds = torch.argmax(feature_output, dim=1)

            all_emotion_preds.extend(emotion_preds.tolist())
            all_feature_preds.extend(feature_preds.tolist())

    return epoch_loss / len(dataloader), emotion_correct / total, feature_correct / total, all_emotion_preds, all_feature_preds

def main():
    datasetPath = '/dev/shm/datasets/'
    data_file = datasetPath + 'processed_data.csv'
    embedding_file = datasetPath + 'sgns.weibo.bigram-char'
    texts, features, emotions = read_csv(data_file)
    tokenizer = get_tokenizer()
    tokenized_texts = [tokenizer(text) for text in texts]
    vocab = build_vocab_from_iterator(tokenized_texts)
    feature_vocab = build_feature_vocab(features)

    train_texts, test_texts, train_features, test_features, train_emotions, test_emotions = train_test_split(texts, features, emotions, test_size=0.2, random_state=42)

    train_dataset = TextDataset(train_texts, train_features, train_emotions, tokenizer, vocab, feature_vocab)
    test_dataset = TextDataset(test_texts, test_features, test_emotions, tokenizer, vocab, feature_vocab)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=pad_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    print(f"Using device: {device}")

    embeddings = load_embeddings(embedding_file, vocab)
    modelList = list()

    modelList.append(
        TextClassificationModel_CNN_MHA(
        len(vocab),
        len(feature_vocab),
        300,
        8,
        2,
        len(feature_vocab),
        embeddings=embeddings,
        kernel_size=3,
        num_filters=64))
    modelList.append(TextClassificationModel_BiLSTM_CNN_MHA(
        len(vocab),
        len(feature_vocab),
        300,
        128,
        8,
        2,
        len(feature_vocab),
        embeddings=embeddings,
        kernel_size=3,
        num_filters=64
    ).to(device))
    modelList.append(TextClassificationModel_BiGRU(len(vocab), len(feature_vocab), 300, 32, 8, 2, len(feature_vocab), embeddings=embeddings))
    modelList.append(TextClassificationModel_BiLSTM_MHA(len(vocab), len(feature_vocab), 300, 128, 8, 2, len(feature_vocab), embeddings=embeddings))
    modelList.append(TextClassificationModel_GRU_MHA(len(vocab), len(feature_vocab), 300, 128, 8, 2, len(feature_vocab), embeddings=embeddings))
    modelList.append(TextClassificationModel_BiGRU_MHA(len(vocab), len(feature_vocab), 300, 128, 8, 2, len(feature_vocab), embeddings=embeddings))

    for model in  modelList:
        name = model.name
        print("IN:",model.name)

        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            model = nn.DataParallel(model)
        model.to(device)
        if num_gpus > 1:
            model = nn.DataParallel(model)
        else:
            model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.0002)
        criterion = torch.nn.CrossEntropyLoss()

        num_epochs = 100
        best_test_loss = float('inf')

        for epoch in range(num_epochs):
            train_loss = train(model, train_dataloader, optimizer, criterion, device)
            test_loss, test_emotion_accuracy, test_feature_accuracy, test_emotion_preds, test_feature_preds = evaluate(
                model, test_dataloader, criterion, device)
            print(
                f'Test Loss: {test_loss:.4f}, Train Loss: {train_loss:.4f}, Test Emotion Accuracy: {test_emotion_accuracy:.4f}, Test Feature Accuracy: {test_feature_accuracy:.4f}')

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), 'best_model'+name+'.pt')
                with open(name+'.txt','w', encoding='utf-8') as file:
                    for i, (text, true_feature, true_emotion) in enumerate(zip(test_texts, test_features, test_emotions)):
                        pred_emotion = 'positive' if test_emotion_preds[i] == 1 else 'negative'
                        pred_feature = list(feature_vocab.keys())[list(feature_vocab.values()).index(test_feature_preds[i])]
                        file.write(
                            f"Text: {text}, True Feature: {true_feature}, True Emotion: {true_emotion}, Predicted Feature: {pred_feature}, Predicted Emotion: {pred_emotion}")


if __name__ == '__main__':
    main()
