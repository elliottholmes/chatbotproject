import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────────
WINDOW      = 5       # rolling match window for form features
EMBED_DIM   = 8       # team embedding size
EPOCHS      = 150
LR          = 1e-3
TEMPERATURE = 2.0     # higher = softer/more uncertain predictions
MAX_GOALS   = 10      # max goals considered per team for Poisson grid
SEED        = 42

torch.manual_seed(SEED)

# ── Load & sort data ───────────────────────────────────────────────────────────
with open("data.json") as f:
    raw = json.load(f)

matches = [m for m in raw["results"] if m.get("isResult")]
matches.sort(key=lambda m: m["datetime"])
print(f"Loaded {len(matches)} completed matches.\n")

# ── Team encoding ──────────────────────────────────────────────────────────────
all_teams = sorted({m["h"]["id"] for m in matches} | {m["a"]["id"] for m in matches})
team_le   = LabelEncoder().fit(all_teams)
NUM_TEAMS = len(all_teams)

# Build title → id lookup
title_to_id = {}
for m in matches:
    title_to_id[m["h"]["title"].lower()] = m["h"]["id"]
    title_to_id[m["a"]["title"].lower()] = m["a"]["id"]

# ── Rolling feature builder ────────────────────────────────────────────────────
def make_dataset(matches):
    team_history = defaultdict(list)

    def team_stats(tid):
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

    records = []
    for m in matches:
        hid    = m["h"]["id"]
        aid    = m["a"]["id"]
        hgoals = int(m["goals"]["h"])
        agoals = int(m["goals"]["a"])
        hxg    = float(m["xG"]["h"])
        axg    = float(m["xG"]["a"])

        h_stats = team_stats(hid)
        a_stats = team_stats(aid)
        forecast = [float(m["forecast"]["w"]),
                    float(m["forecast"]["d"]),
                    float(m["forecast"]["l"])]

        records.append({
            "h_enc":   int(team_le.transform([hid])[0]),
            "a_enc":   int(team_le.transform([aid])[0]),
            "feat":    h_stats + a_stats + forecast,
            "hgoals":  hgoals,
            "agoals":  agoals,
        })

        team_history[hid].append({"gf": hgoals, "ga": agoals, "xgf": hxg, "xga": axg,
                                   "won": int(hgoals > agoals), "drew": int(hgoals == agoals)})
        team_history[aid].append({"gf": agoals, "ga": hgoals, "xgf": axg, "xga": hxg,
                                   "won": int(agoals > hgoals), "drew": int(hgoals == agoals)})

    return records, team_history

records, team_history = make_dataset(matches)
FEAT_DIM = len(records[0]["feat"])

# ── Dataset ────────────────────────────────────────────────────────────────────
class MatchDataset(Dataset):
    def __init__(self, recs):
        self.recs = recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        r = self.recs[idx]
        return (
            torch.tensor(r["h_enc"],  dtype=torch.long),
            torch.tensor(r["a_enc"],  dtype=torch.long),
            torch.tensor(r["feat"],   dtype=torch.float32),
            torch.tensor(r["hgoals"], dtype=torch.float32),
            torch.tensor(r["agoals"], dtype=torch.float32),
        )

n_train = int(0.8 * len(records))
train_ds = MatchDataset(records[:n_train])
val_ds   = MatchDataset(records[n_train:])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

# ── Model — predicts log(λ_home) and log(λ_away) ──────────────────────────────
class PoissonMatchPredictor(nn.Module):
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
            nn.Linear(32, 2),   # → [log_lambda_home, log_lambda_away]
        )

    def forward(self, h_idx, a_idx, feats):
        h_emb = self.team_embed(h_idx)
        a_emb = self.team_embed(a_idx)
        x     = torch.cat([h_emb, a_emb, feats], dim=1)
        log_lambdas = self.net(x)
        # clamp to avoid exploding lambdas
        log_lambdas = torch.clamp(log_lambdas, -2.0, 2.0)
        return torch.exp(log_lambdas)   # shape (B, 2): [lambda_h, lambda_a]

model     = PoissonMatchPredictor(NUM_TEAMS, EMBED_DIM, FEAT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
poisson_loss = nn.PoissonNLLLoss(log_input=False, full=True)

# ── Poisson outcome probabilities ──────────────────────────────────────────────
def poisson_pmf(lam, k):
    """P(X=k) for Poisson(lam)."""
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def outcome_probs_from_lambdas(lam_h, lam_a, temperature=TEMPERATURE):
    """
    Compute P(home win), P(draw), P(away win) by summing over a
    scoreline grid, then apply temperature scaling.
    """
    p_hw = p_draw = p_aw = 0.0
    for h in range(MAX_GOALS + 1):
        ph = poisson_pmf(lam_h, h)
        for a in range(MAX_GOALS + 1):
            pa  = poisson_pmf(lam_a, a)
            p   = ph * pa
            if h > a:
                p_hw   += p
            elif h == a:
                p_draw += p
            else:
                p_aw   += p

    # Temperature scaling: soften via log-space division
    log_probs = np.array([
        math.log(max(p_hw,   1e-12)),
        math.log(max(p_draw, 1e-12)),
        math.log(max(p_aw,   1e-12)),
    ])
    scaled = log_probs / temperature
    scaled -= scaled.max()                      # numerical stability
    probs  = np.exp(scaled)
    probs /= probs.sum()
    return probs.tolist()

# ── Training ───────────────────────────────────────────────────────────────────
def eval_mae(loader):
    model.eval()
    total_err, total_n = 0.0, 0
    with torch.no_grad():
        for h_idx, a_idx, feats, hg, ag in loader:
            lambdas = model(h_idx, a_idx, feats)
            err = torch.abs(lambdas[:, 0] - hg) + torch.abs(lambdas[:, 1] - ag)
            total_err += err.sum().item()
            total_n   += hg.size(0)
    return total_err / total_n if total_n else 0

print("Training Poisson model...\n")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for h_idx, a_idx, feats, hg, ag in train_loader:
        optimizer.zero_grad()
        lambdas = model(h_idx, a_idx, feats)
        loss = (poisson_loss(lambdas[:, 0], hg) +
                poisson_loss(lambdas[:, 1], ag))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 15 == 0:
        val_mae = eval_mae(val_loader)
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss:.4f}  val_MAE={val_mae:.4f}")

torch.save({
    "model_state": model.state_dict(),
    "team_le":     team_le,
    "team_history": dict(team_history),
    "title_to_id":  title_to_id,
    "num_teams":    NUM_TEAMS,
    "embed_dim":    EMBED_DIM,
    "feat_dim":     FEAT_DIM,
}, "model.pt")
print("\nModel saved to model.pt")

# ── Prediction helper ──────────────────────────────────────────────────────────
OUTCOME = ["Home Win", "Draw", "Away Win"]

def predict_match(home_title, away_title):
    hid = title_to_id.get(home_title.lower())
    aid = title_to_id.get(away_title.lower())
    if not hid:
        raise ValueError(f"Team not found: {home_title}")
    if not aid:
        raise ValueError(f"Team not found: {away_title}")

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

    feat = get_stats(hid) + get_stats(aid) + [1/3, 1/3, 1/3]
    h_enc = int(team_le.transform([hid])[0])
    a_enc = int(team_le.transform([aid])[0])

    model.eval()
    with torch.no_grad():
        h_t = torch.tensor([h_enc], dtype=torch.long)
        a_t = torch.tensor([a_enc], dtype=torch.long)
        f_t = torch.tensor([feat],  dtype=torch.float32)
        lambdas = model(h_t, a_t, f_t).squeeze().tolist()

    lam_h, lam_a = lambdas
    probs = outcome_probs_from_lambdas(lam_h, lam_a, temperature=TEMPERATURE)

    print(f"\n{'='*50}")
    print(f"  {home_title}  vs  {away_title}")
    print(f"  Expected goals: {lam_h:.2f} – {lam_a:.2f}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"{'='*50}")
    for label, p in zip(OUTCOME, probs):
        bar = "█" * int(p * 30)
        print(f"  {label:<12} {p*100:5.1f}%  {bar}")
    print(f"  Prediction: {OUTCOME[int(np.argmax(probs))]}")
    print(f"{'='*50}\n")
    return probs

# ── Example predictions ────────────────────────────────────────────────────────
print("\n--- Example Predictions (temp={}) ---".format(TEMPERATURE))
predict_match("Liverpool", "Arsenal")
predict_match("Manchester City", "Chelsea")
predict_match("Tottenham", "Aston Villa")
