import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────────
DECAY_RATE    = 0.12   # exponential decay per match back in time
RECENT_BOOST  = 2.5    # extra weight multiplier for the last 5 matches
LEAGUE_AVG_G  = 1.4    # approximate EPL average goals per team per game
EMBED_DIM     = 8
EPOCHS        = 150
LR            = 1e-3
TEMPERATURE   = 2.0
MAX_GOALS     = 10
SEED          = 42

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

title_to_id = {}
for m in matches:
    title_to_id[m["h"]["title"].lower()] = m["h"]["id"]
    title_to_id[m["a"]["title"].lower()] = m["a"]["id"]

# ── Feature engineering ────────────────────────────────────────────────────────
# Per-team running totals used to estimate opponent quality at time of each match
# { team_id: { "gf": total, "ga": total, "n": count } }
def make_dataset(matches):
    team_history     = defaultdict(list)   # full match-by-match log
    team_running     = defaultdict(lambda: {"gf": 0.0, "ga": 0.0, "n": 0})

    def opp_quality(opp_id):
        """
        Returns (attack_strength, defense_strength) for an opponent
        based on their season stats so far.
        attack_strength  = avg goals scored  (higher = tougher to defend against)
        defense_strength = avg goals conceded (higher = easier to score against)
        """
        s = team_running[opp_id]
        if s["n"] == 0:
            return LEAGUE_AVG_G, LEAGUE_AVG_G
        return s["gf"] / s["n"], s["ga"] / s["n"]

    def compute_features(tid):
        """
        Compute 8 features for a team using ALL their season history,
        with exponential decay weighting (older = less weight) and
        an extra boost on the most recent 5 matches.
        Each match's stats are adjusted by the quality of the opponent faced.
        """
        hist = team_history[tid]
        if not hist:
            return [0.0] * 8

        n = len(hist)

        # Build decay weights
        raw_w = []
        for i in range(n):
            age = n - 1 - i          # 0 = most recent
            w   = math.exp(-DECAY_RATE * age)
            if age < 5:
                w *= RECENT_BOOST    # boost last 5
            raw_w.append(w)
        total_w = sum(raw_w)
        weights = [w / total_w for w in raw_w]

        # Opponent quality per match
        opp_att = [h["opp_att"] for h in hist]
        opp_def = [h["opp_def"] for h in hist]

        # Quality-adjusted stats:
        #   Goals scored vs strong defense → amplified (harder to score, so more impressive)
        #   Goals conceded vs strong attack → dampened (less shameful)
        adj_gf  = sum(w * h["gf"]  * (oa / LEAGUE_AVG_G)
                      for w, h, oa in zip(weights, hist, opp_def))
        adj_ga  = sum(w * h["ga"]  * (LEAGUE_AVG_G / max(oa, 0.1))
                      for w, h, oa in zip(weights, hist, opp_att))
        adj_xgf = sum(w * h["xgf"] * (oa / LEAGUE_AVG_G)
                      for w, h, oa in zip(weights, hist, opp_def))
        adj_xga = sum(w * h["xga"] * (LEAGUE_AVG_G / max(oa, 0.1))
                      for w, h, oa in zip(weights, hist, opp_att))

        wr  = sum(w * h["won"]  for w, h in zip(weights, hist))
        dr  = sum(w * h["drew"] for w, h in zip(weights, hist))

        # Average opponent attack/defense quality faced (fixture difficulty)
        avg_opp_att = sum(w * oa for w, oa in zip(weights, opp_att))
        avg_opp_def = sum(w * od for w, od in zip(weights, opp_def))

        return [adj_gf, adj_ga, adj_xgf, adj_xga, wr, dr, avg_opp_att, avg_opp_def]

    records = []
    for m in matches:
        hid    = m["h"]["id"]
        aid    = m["a"]["id"]
        hgoals = int(m["goals"]["h"])
        agoals = int(m["goals"]["a"])
        hxg    = float(m["xG"]["h"])
        axg    = float(m["xG"]["a"])

        # Features computed BEFORE updating history (no data leakage)
        h_feats  = compute_features(hid)
        a_feats  = compute_features(aid)
        forecast = [float(m["forecast"]["w"]),
                    float(m["forecast"]["d"]),
                    float(m["forecast"]["l"])]

        records.append({
            "h_enc":  int(team_le.transform([hid])[0]),
            "a_enc":  int(team_le.transform([aid])[0]),
            "feat":   h_feats + a_feats + forecast,
            "hgoals": hgoals,
            "agoals": agoals,
        })

        # Opponent quality at the time of this match
        h_opp_att, h_opp_def = opp_quality(aid)   # home team faced away team
        a_opp_att, a_opp_def = opp_quality(hid)   # away team faced home team

        # Update history
        team_history[hid].append({
            "gf": hgoals, "ga": agoals, "xgf": hxg, "xga": axg,
            "won": int(hgoals > agoals), "drew": int(hgoals == agoals),
            "opp_att": h_opp_att, "opp_def": h_opp_def,
        })
        team_history[aid].append({
            "gf": agoals, "ga": hgoals, "xgf": axg, "xga": hxg,
            "won": int(agoals > hgoals), "drew": int(hgoals == agoals),
            "opp_att": a_opp_att, "opp_def": a_opp_def,
        })

        # Update running totals for future opponent quality lookups
        team_running[hid]["gf"] += hgoals
        team_running[hid]["ga"] += agoals
        team_running[hid]["n"]  += 1
        team_running[aid]["gf"] += agoals
        team_running[aid]["ga"] += hgoals
        team_running[aid]["n"]  += 1

    return records, team_history

records, team_history = make_dataset(matches)
FEAT_DIM = len(records[0]["feat"])   # 8 + 8 + 3 = 19
print(f"Feature dimension: {FEAT_DIM}\n")

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

n_train      = int(0.8 * len(records))
train_ds     = MatchDataset(records[:n_train])
val_ds       = MatchDataset(records[n_train:])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

# ── Model ──────────────────────────────────────────────────────────────────────
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
            nn.Linear(32, 2),
        )

    def forward(self, h_idx, a_idx, feats):
        h_emb = self.team_embed(h_idx)
        a_emb = self.team_embed(a_idx)
        x = torch.cat([h_emb, a_emb, feats], dim=1)
        log_lambdas = self.net(x)
        log_lambdas = torch.clamp(log_lambdas, -2.0, 2.0)
        return torch.exp(log_lambdas)

model        = PoissonMatchPredictor(NUM_TEAMS, EMBED_DIM, FEAT_DIM)
optimizer    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
poisson_loss = nn.PoissonNLLLoss(log_input=False, full=True)

# ── Poisson outcome probabilities ──────────────────────────────────────────────
def poisson_pmf(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def outcome_probs_from_lambdas(lam_h, lam_a, temperature=TEMPERATURE):
    p_hw = p_draw = p_aw = 0.0
    for h in range(MAX_GOALS + 1):
        ph = poisson_pmf(lam_h, h)
        for a in range(MAX_GOALS + 1):
            pa = poisson_pmf(lam_a, a)
            p  = ph * pa
            if   h > a: p_hw   += p
            elif h == a: p_draw += p
            else:        p_aw   += p

    log_probs = np.array([
        math.log(max(p_hw,   1e-12)),
        math.log(max(p_draw, 1e-12)),
        math.log(max(p_aw,   1e-12)),
    ])
    scaled  = log_probs / temperature
    scaled -= scaled.max()
    probs   = np.exp(scaled)
    probs  /= probs.sum()
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
    "model_state":  model.state_dict(),
    "team_le":      team_le,
    "team_history": dict(team_history),
    "title_to_id":  title_to_id,
    "num_teams":     NUM_TEAMS,
    "embed_dim":     EMBED_DIM,
    "feat_dim":      FEAT_DIM,
    "decay_rate":    DECAY_RATE,
    "recent_boost":  RECENT_BOOST,
    "league_avg_g":  LEAGUE_AVG_G,
}, "model.pt")
print("\nModel saved to model.pt")

# ── Prediction helper ──────────────────────────────────────────────────────────
OUTCOME = ["Home Win", "Draw", "Away Win"]

def compute_features_for_predict(tid):
    hist = team_history[tid]
    if not hist:
        return [0.0] * 8
    n = len(hist)
    raw_w = []
    for i in range(n):
        age = n - 1 - i
        w   = math.exp(-DECAY_RATE * age)
        if age < 5:
            w *= RECENT_BOOST
        raw_w.append(w)
    total_w = sum(raw_w)
    weights = [w / total_w for w in raw_w]
    opp_att = [h["opp_att"] for h in hist]
    opp_def = [h["opp_def"] for h in hist]
    adj_gf  = sum(w * h["gf"]  * (oa / LEAGUE_AVG_G) for w, h, oa in zip(weights, hist, opp_def))
    adj_ga  = sum(w * h["ga"]  * (LEAGUE_AVG_G / max(oa, 0.1)) for w, h, oa in zip(weights, hist, opp_att))
    adj_xgf = sum(w * h["xgf"] * (oa / LEAGUE_AVG_G) for w, h, oa in zip(weights, hist, opp_def))
    adj_xga = sum(w * h["xga"] * (LEAGUE_AVG_G / max(oa, 0.1)) for w, h, oa in zip(weights, hist, opp_att))
    wr  = sum(w * h["won"]  for w, h in zip(weights, hist))
    dr  = sum(w * h["drew"] for w, h in zip(weights, hist))
    avg_opp_att = sum(w * oa for w, oa in zip(weights, opp_att))
    avg_opp_def = sum(w * od for w, od in zip(weights, opp_def))
    return [adj_gf, adj_ga, adj_xgf, adj_xga, wr, dr, avg_opp_att, avg_opp_def]

def predict_match(home_title, away_title):
    hid = title_to_id.get(home_title.lower())
    aid = title_to_id.get(away_title.lower())
    if not hid:
        raise ValueError(f"Team not found: {home_title}")
    if not aid:
        raise ValueError(f"Team not found: {away_title}")

    feat  = compute_features_for_predict(hid) + compute_features_for_predict(aid) + [1/3, 1/3, 1/3]
    h_enc = int(team_le.transform([hid])[0])
    a_enc = int(team_le.transform([aid])[0])

    model.eval()
    with torch.no_grad():
        lambdas = model(
            torch.tensor([h_enc], dtype=torch.long),
            torch.tensor([a_enc], dtype=torch.long),
            torch.tensor([feat],  dtype=torch.float32),
        ).squeeze().tolist()

    lam_h, lam_a = lambdas
    probs = outcome_probs_from_lambdas(lam_h, lam_a)

    print(f"\n{'='*52}")
    print(f"  {home_title}  vs  {away_title}")
    print(f"  Expected goals: {lam_h:.2f} – {lam_a:.2f}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"{'='*52}")
    for label, p in zip(OUTCOME, probs):
        bar = "█" * int(p * 30)
        print(f"  {label:<12} {p*100:5.1f}%  {bar}")
    print(f"  Prediction: {OUTCOME[int(np.argmax(probs))]}")
    print(f"{'='*52}\n")
    return probs

# ── Example predictions ────────────────────────────────────────────────────────
print("\n--- Example Predictions ---")
predict_match("Liverpool", "Arsenal")
predict_match("Manchester City", "Chelsea")
predict_match("Tottenham", "Aston Villa")
