import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1. Load and preprocess data
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X_raw, y_raw = newsgroups.data, newsgroups.target

# Vectorize text (TF-IDF)
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X_raw).toarray()

# Encode labels (already encoded in this dataset, but for generality)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 2. PyTorch Dataset
class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = NewsDataset(X_train, y_train)
val_dataset = NewsDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# 3. MLP Model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)

input_dim = X_tfidf.shape[1]
hidden_dim = 512
num_classes = len(newsgroups.target_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPClassifier(input_dim, hidden_dim, num_classes).to(device)

# 4. Train the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(model, loader):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds)
            all_labels.extend(y_batch)
    return classification_report(all_labels, all_preds, target_names=newsgroups.target_names)

# Training loop
for epoch in range(50):
    train_loss = train(model, train_loader)
    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")

# 5. Evaluation
print("\nEvaluation on validation set:")
print(evaluate(model, val_loader))