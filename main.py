import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from datetime import datetime

# ── Load data ──────────────────────────────────────────────────────────────────
with open("data.json") as f:
    raw = json.load(f)

matches = [m for m in raw["results"] if m.get("isResult")]
matches.sort(key=lambda m: m["datetime"])

# ── Team encoding ──────────────────────────────────────────────────────────────
all_teams = sorted({m["h"]["id"] for m in matches} | {m["a"]["id"] for m in matches})
team_le = LabelEncoder()
team_le.fit(all_teams)
NUM_TEAMS = len(all_teams)

# ── Rolling feature builder (last N matches per team) ─────────────────────────
WINDOW = 5

def make_rolling_features(matches):
    """
    For each match (in chronological order), compute rolling stats for both
    home and away teams using their last WINDOW completed matches.
    Returns feature matrix X and label vector y.
    """
    # Per-team history: list of dicts with goals_for, goals_against, xg_for, xg_against, won, drew
    team_history = defaultdict(list)

    def get_team_stats(tid):
        hist = team_history[tid][-WINDOW:]
        if not hist:
            return [0.0] * 6
        gf  = np.mean([h["gf"]  for h in hist])
        ga  = np.mean([h["ga"]  for h in hist])
        xgf = np.mean([h["xgf"] for h in hist])
        xga = np.mean([h["xga"] for h in hist])
        wr  = np.mean([h["won"] for h in hist])
        dr  = np.mean([h["drew"] for h in hist])
        return [gf, ga, xgf, xga, wr, dr]

    X, y = [], []
    for m in matches:
        hid = m["h"]["id"]
        aid = m["a"]["id"]
        hgoals = int(m["goals"]["h"])
        agoals = int(m["goals"]["a"])
        hxg    = float(m["xG"]["h"])
        axg    = float(m["xG"]["a"])

        # Rolling stats before this match
        h_stats = get_team_stats(hid)
        a_stats = get_team_stats(aid)

        # Team embeddings (one-hot index for embedding layer)
        h_enc = int(team_le.transform([hid])[0])
        a_enc = int(team_le.transform([aid])[0])

        # Forecast probs as extra features
        fw = float(m["forecast"]["w"])
        fd = float(m["forecast"]["d"])
        fl = float(m["forecast"]["l"])

        feat = h_stats + a_stats + [fw, fd, fl]
        X.append((h_enc, a_enc, feat))

        # Label: 0 = home win, 1 = draw, 2 = away win
        if hgoals > agoals:
            label = 0
        elif hgoals == agoals:
            label = 1
        else:
            label = 2
        y.append(label)

        # Update histories
        team_history[hid].append({"gf": hgoals, "ga": agoals, "xgf": hxg, "xga": axg,
                                   "won": int(hgoals > agoals), "drew": int(hgoals == agoals)})
        team_history[aid].append({"gf": agoals, "ga": hgoals, "xgf": axg, "xga": hxg,
                                   "won": int(agoals > hgoals), "drew": int(hgoals == agoals)})

    return X, y, team_history

X_raw, y_raw, team_history = make_rolling_features(matches)

# ── Dataset ────────────────────────────────────────────────────────────────────
class MatchDataset(Dataset):
    def __init__(self, data, labels):
        self.data   = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        h_enc, a_enc, feat = self.data[idx]
        return (torch.tensor(h_enc, dtype=torch.long),
                torch.tensor(a_enc, dtype=torch.long),
                torch.tensor(feat, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))

FEAT_DIM = len(X_raw[0][2])  # 6 + 6 + 3 = 15

dataset    = MatchDataset(X_raw, y_raw)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

# ── Model ──────────────────────────────────────────────────────────────────────
EMBED_DIM = 8

class MatchPredictor(nn.Module):
    def __init__(self, num_teams, embed_dim, feat_dim):
        super().__init__()
        self.team_embed = nn.Embedding(num_teams, embed_dim)
        in_dim = embed_dim * 2 + feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
        )

    def forward(self, h_idx, a_idx, feats):
        h_emb = self.team_embed(h_idx)
        a_emb = self.team_embed(a_idx)
        x = torch.cat([h_emb, a_emb, feats], dim=1)
        return self.net(x)

model     = MatchPredictor(NUM_TEAMS, EMBED_DIM, FEAT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS = 100

def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for h_idx, a_idx, feats, labels in loader:
            out = model(h_idx, a_idx, feats)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total else 0

print("Training model on match data...\n")
for epoch in range(1, EPOCHS + 1):
    model.train()
    for h_idx, a_idx, feats, labels in train_loader:
        optimizer.zero_grad()
        out  = model(h_idx, a_idx, feats)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        train_acc = evaluate(train_loader)
        val_acc   = evaluate(val_loader)
        print(f"Epoch {epoch:3d}/{EPOCHS}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

torch.save(model.state_dict(), "model.pt")
print("\nModel saved to model.pt")

# ── Prediction helper ──────────────────────────────────────────────────────────
OUTCOME = ["Home Win", "Draw", "Away Win"]

def predict_match(home_title, away_title):
    """
    Predict the outcome of a future match given team names.
    Uses each team's most recent rolling stats from training data.
    """
    # Find team IDs by title
    def find_id(title):
        for m in matches:
            if m["h"]["title"].lower() == title.lower():
                return m["h"]["id"]
            if m["a"]["title"].lower() == title.lower():
                return m["a"]["id"]
        raise ValueError(f"Team not found: {title}")

    hid = find_id(home_title)
    aid = find_id(away_title)

    def get_stats(tid):
        hist = team_history[tid][-WINDOW:]
        if not hist:
            return [0.0] * 6
        return [
            np.mean([h["gf"]  for h in hist]),
            np.mean([h["ga"]  for h in hist]),
            np.mean([h["xgf"] for h in hist]),
            np.mean([h["xga"] for h in hist]),
            np.mean([h["won"] for h in hist]),
            np.mean([h["drew"] for h in hist]),
        ]

    h_stats = get_stats(hid)
    a_stats = get_stats(aid)
    # No pre-match forecast available, use neutral 1/3 each
    feat = h_stats + a_stats + [1/3, 1/3, 1/3]

    h_enc = int(team_le.transform([hid])[0])
    a_enc = int(team_le.transform([aid])[0])

    model.eval()
    with torch.no_grad():
        h_t = torch.tensor([h_enc], dtype=torch.long)
        a_t = torch.tensor([a_enc], dtype=torch.long)
        f_t = torch.tensor([feat],  dtype=torch.float32)
        logits = model(h_t, a_t, f_t)
        probs  = torch.softmax(logits, dim=1).squeeze().tolist()

    print(f"\n{'='*45}")
    print(f"  {home_title}  vs  {away_title}")
    print(f"{'='*45}")
    for label, p in zip(OUTCOME, probs):
        bar = "█" * int(p * 30)
        print(f"  {label:<12} {p*100:5.1f}%  {bar}")
    print(f"  Prediction: {OUTCOME[int(np.argmax(probs))]}")
    print(f"{'='*45}\n")
    return probs

# ── Example predictions ────────────────────────────────────────────────────────
print("\n--- Example Predictions ---")
predict_match("Liverpool", "Arsenal")
predict_match("Manchester City", "Chelsea")
predict_match("Tottenham", "Aston Villa")
