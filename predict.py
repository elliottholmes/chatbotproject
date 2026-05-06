import math
import numpy as np
import torch
import torch.nn as nn

TEMPERATURE = 1.5
MAX_GOALS   = 10
OUTCOME     = ["Home Win", "Draw", "Away Win"]

# ── Model definition (must match train.py) ────────────────────────────────────
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

# ── Load saved model ───────────────────────────────────────────────────────────
checkpoint   = torch.load("model.pt", weights_only=False)
team_le      = checkpoint["team_le"]
team_history = checkpoint["team_history"]
final_table  = checkpoint["final_table"]
title_to_id  = checkpoint["title_to_id"]
num_teams    = checkpoint["num_teams"]
embed_dim    = checkpoint["embed_dim"]
feat_dim     = checkpoint["feat_dim"]
DECAY_RATE   = checkpoint["decay_rate"]
RECENT_BOOST = checkpoint["recent_boost"]
LEAGUE_AVG_G = checkpoint["league_avg_g"]

model = PoissonMatchPredictor(num_teams, embed_dim, feat_dim)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ── Feature computation (mirrors train.py) ─────────────────────────────────────
def compute_features(tid):
    hist = list(team_history.get(tid, []))
    t    = final_table.get(tid, {"pts": 0, "gd": 0, "gp": 0})
    gp   = max(t["gp"], 1)
    pts_pg = t["pts"] / gp
    gd_pg  = t["gd"]  / gp

    all_pts  = [(final_table[t2]["pts"], final_table[t2]["gd"])
                for t2 in final_table if final_table[t2]["gp"] > 0]
    my_score = (t["pts"], t["gd"])
    rank     = sum(1 for s in all_pts if s > my_score)
    pos_norm = 1.0 - rank / max(len(all_pts) - 1, 1)
    table_feats = [pts_pg, gd_pg, pos_norm]

    if not hist:
        return [0.0] * 8 + table_feats

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
    wr      = sum(w * h["won"]  for w, h in zip(weights, hist))
    dr      = sum(w * h["drew"] for w, h in zip(weights, hist))
    avg_opp_att = sum(w * oa for w, oa in zip(weights, opp_att))
    avg_opp_def = sum(w * od for w, od in zip(weights, opp_def))

    return [adj_gf, adj_ga, adj_xgf, adj_xga, wr, dr,
            avg_opp_att, avg_opp_def] + table_feats

# ── Poisson helpers ────────────────────────────────────────────────────────────
def poisson_pmf(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def outcome_probs(lam_h, lam_a, temperature=TEMPERATURE):
    p_hw = p_draw = p_aw = 0.0
    for h in range(MAX_GOALS + 1):
        ph = poisson_pmf(lam_h, h)
        for a in range(MAX_GOALS + 1):
            pa = poisson_pmf(lam_a, a)
            p  = ph * pa
            if   h > a:  p_hw   += p
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

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(home, away):
    hid = title_to_id.get(home.lower())
    aid = title_to_id.get(away.lower())

    if not hid or not aid:
        return None

    feat  = compute_features(hid) + compute_features(aid) + [1/3, 1/3, 1/3]
    h_enc = int(team_le.transform([hid])[0])
    a_enc = int(team_le.transform([aid])[0])

    with torch.no_grad():
        lambdas = model(
            torch.tensor([h_enc], dtype=torch.long),
            torch.tensor([a_enc], dtype=torch.long),
            torch.tensor([feat],  dtype=torch.float32),
        ).squeeze().tolist()

    lam_h, lam_a = lambdas
    probs = outcome_probs(lam_h, lam_a)

    return {
        "lam_h": lam_h,
        "lam_a": lam_a,
        "probs": probs,
        "prediction": OUTCOME[int(np.argmax(probs))]
    }

def predict_with_output(home, away):
    result = predict(home, away)

    if result is None:
        return None

    lam_h = result["lam_h"]
    lam_a = result["lam_a"]
    probs = result["probs"]

    lines = []
    lines.append("=" * 52)
    lines.append(f"  {home.title():<22} vs  {away.title()}")
    lines.append(f"  Expected goals:  {lam_h:.2f}  –  {lam_a:.2f}")
    lines.append(f"  Temperature:     {TEMPERATURE}")
    lines.append("=" * 52)

    for label, p in zip(OUTCOME, probs):
        bar = "█" * int(p * 30)
        lines.append(f"  {label:<12} {p*100:5.1f}%  {bar}")

    prediction = OUTCOME[int(np.argmax(probs))]
    lines.append(f"\n  Prediction: {prediction}")
    lines.append("=" * 52)

    return "\n".join(lines), result

def predict_with_output2(home, away):
    result = predict(home, away)

    if result is None:
        return None

    lam_h = result["lam_h"]
    lam_a = result["lam_a"]
    probs = result["probs"]

    lines = []
    lines.append("=" * 52)
    lines.append(f"  {home.title():<22} vs  {away.title()}")
    lines.append(f"  Expected goals:  {lam_h:.2f}  –  {lam_a:.2f}")
    lines.append(f"  Temperature:     {TEMPERATURE}")
    lines.append("=" * 52)

    for label, p in zip(OUTCOME, probs):
        bar = "█" * int(p * 30)
        lines.append(f"  {label:<12} {p*100:5.1f}%  {bar}")

    prediction = OUTCOME[int(np.argmax(probs))]
    lines.append(f"\n  Prediction: {prediction}")
    lines.append("=" * 52)
    return "\n".join(lines)

def list_teams():
    print("\nAvailable teams:")
    for i, t in enumerate(sorted(title_to_id.keys()), 1):
        print(f"  {i:2d}. {t.title()}")
    print()

# ── Interactive prompt ─────────────────────────────────────────────────────────
def run():
    print("Match Predictor — model loaded.")
    print("Type 'teams' to list all available teams, or 'quit' to exit.\n")

    while True:
        try:
            home = input("Home team: ").strip()
            if home.lower() == "quit":
                break
            if home.lower() == "teams":
                list_teams()
                continue
            away = input("Away team: ").strip()
            if away.lower() == "quit":
                break
            print(predict_with_output2(home, away))
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    run()
