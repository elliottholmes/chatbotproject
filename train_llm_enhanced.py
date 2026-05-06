"""
train.py — QLoRA fine-tuning of TinyLlama-1.1B-Chat-v1.0
for the EPL football chatbot.

Hardware target: RTX 3070 8 GB VRAM, 32 GB RAM
Technique:       4-bit NF4 quantisation (bitsandbytes) + LoRA adapters (PEFT)
Base model:      TinyLlama/TinyLlama-1.1B-Chat-v1.0
Output:          ./lora_adapters/   (~40-60 MB, commit this folder to Replit)

Estimated training time: 20-30 mins on RTX 3070

Install deps first:
    pip install -r requirements.txt
"""

import json
import math
import random
import os
import numpy as np
import torch
from collections import defaultdict

# Force CPU-only training to avoid GPU initialization hang
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda: False
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

if __name__ == "__main__":
    
    # ── Reproducibility ───────────────────────────────────────────────────────────
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # ── Config ────────────────────────────────────────────────────────────────────
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    OUTPUT_DIR = "./lora_adapters"
    DATA_FILE  = "data.json"
    
    # LoRA — smaller rank suits a 1.1B model
    LORA_R           = 8
    LORA_ALPHA       = 16
    LORA_DROPOUT     = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Training
    MAX_SEQ_LENGTH = 512
    TRAIN_STEPS    = 300      # ~20-30 mins on RTX 3070
    BATCH_SIZE     = 2        # reduced to avoid dataloader hang
    GRAD_ACCUM     = 2        # effective batch = 4
    LR             = 3e-4     # slightly higher LR works better for smaller models
    WARMUP_STEPS   = 30
    WEIGHT_DECAY   = 0.01
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 1. Load match data
    # ─────────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  EPL Football Chatbot — QLoRA Training Script")
    print("  Base model: TinyLlama-1.1B-Chat-v1.0")
    print("=" * 60)
    print()
    print("Loading match data...")
    
    with open(DATA_FILE) as f:
        raw = json.load(f)
    
    matches = [m for m in raw["results"] if m.get("isResult")]
    matches.sort(key=lambda m: m["datetime"])
    print(f"  Loaded {len(matches)} completed matches\n")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Build team statistics
    # ─────────────────────────────────────────────────────────────────────────────
    print("Building team statistics...")
    
    title_to_id = {}
    id_to_title = {}
    
    team_stats = defaultdict(lambda: {
        "w": 0, "d": 0, "l": 0,
        "gf": 0, "ga": 0,
        "gp": 0, "pts": 0,
        "xgf": 0.0, "xga": 0.0,
        "clean_sheets": 0,
        "btts": 0,
        "recent": [],
        "home_w": 0, "home_d": 0, "home_l": 0,
        "away_w": 0, "away_d": 0, "away_l": 0,
        "home_gp": 0, "away_gp": 0,
        "current_streak": 0,
    })
    
    h2h_store = defaultdict(list)
    
    for m in matches:
        h_title = m["h"]["title"].lower()
        a_title = m["a"]["title"].lower()
        hid, aid = m["h"]["id"], m["a"]["id"]
    
        title_to_id[h_title] = hid
        title_to_id[a_title] = aid
        id_to_title[hid] = m["h"]["title"]
        id_to_title[aid] = m["a"]["title"]
    
        hg  = int(m["goals"]["h"])
        ag  = int(m["goals"]["a"])
        hxg = float(m["xG"]["h"])
        axg = float(m["xG"]["a"])
    
        key = frozenset([hid, aid])
        h2h_store[key].append({"home": hid, "away": aid, "hg": hg, "ag": ag})
    
        for is_home, (tid, gf, ga, xgf, xga) in enumerate(
            [(hid, hg, ag, hxg, axg), (aid, ag, hg, axg, hxg)]
        ):
            s = team_stats[tid]
            s["gf"]  += gf
            s["ga"]  += ga
            s["xgf"] += xgf
            s["xga"] += xga
            s["gp"]  += 1
    
            if ga == 0:
                s["clean_sheets"] += 1
            if gf > 0 and ga > 0:
                s["btts"] += 1
    
            if gf > ga:
                s["w"]   += 1
                s["pts"] += 3
                r = "W"
                s["current_streak"] = max(0, s["current_streak"]) + 1
            elif gf == ga:
                s["d"]   += 1
                s["pts"] += 1
                r = "D"
                s["current_streak"] = 0
            else:
                s["l"] += 1
                r = "L"
                s["current_streak"] = min(0, s["current_streak"]) - 1
    
            if is_home == 0:
                s["home_gp"] += 1
                if r == "W": s["home_w"] += 1
                elif r == "D": s["home_d"] += 1
                else: s["home_l"] += 1
            else:
                s["away_gp"] += 1
                if r == "W": s["away_w"] += 1
                elif r == "D": s["away_d"] += 1
                else: s["away_l"] += 1
    
            s["recent"].append({
                "result": r, "gf": gf, "ga": ga, "xgf": xgf, "xga": xga,
            })
    
    print(f"  Teams indexed: {len(id_to_title)}\n")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Stat helpers
    # ─────────────────────────────────────────────────────────────────────────────
    
    def safe_pct(num, den):
        return (num / max(den, 1)) * 100
    
    def weighted_form(tid, n=10, decay=0.85):
        recent = team_stats[tid]["recent"][-n:]
        if not recent:
            return 0.0
        weight_total, score = 0.0, 0.0
        for i, r in enumerate(recent):
            w   = decay ** (len(recent) - 1 - i)
            pts = {"W": 3, "D": 1, "L": 0}[r["result"]]
            score        += pts * w
            weight_total += 3 * w
        return score / max(weight_total, 1e-9)
    
    def form_volatility(tid, n=8):
        recent = team_stats[tid]["recent"][-n:]
        if len(recent) < 2:
            return 0.0
        pts = [{"W": 3, "D": 1, "L": 0}[r["result"]] for r in recent]
        return float(np.std(pts))
    
    def clean_sheet_rate(tid):
        s = team_stats[tid]
        return safe_pct(s["clean_sheets"], s["gp"])
    
    def btts_rate(tid):
        s = team_stats[tid]
        return safe_pct(s["btts"], s["gp"])
    
    def home_win_rate(tid):
        s = team_stats[tid]
        return safe_pct(s["home_w"], s["home_gp"])
    
    def away_win_rate(tid):
        s = team_stats[tid]
        return safe_pct(s["away_w"], s["away_gp"])
    
    def get_form_trend(tid):
        wf = weighted_form(tid)
        if wf > 0.70: return "excellent"
        if wf > 0.50: return "good"
        if wf < 0.25: return "poor"
        if wf < 0.40: return "inconsistent"
        return "mixed"
    
    def most_likely_scoreline(lam_h, lam_a, max_goals=6):
        best_prob, best = -1, (0, 0)
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = (math.exp(-lam_h) * lam_h**i / math.factorial(i)) * \
                    (math.exp(-lam_a) * lam_a**j / math.factorial(j))
                if p > best_prob:
                    best_prob, best = p, (i, j)
        return best, best_prob
    
    def get_h2h_summary(hid, aid):
        key   = frozenset([hid, aid])
        games = h2h_store.get(key, [])
        if not games:
            return None
        h_wins = sum(
            1 for g in games
            if (g["home"] == hid and g["hg"] > g["ag"]) or
               (g["away"] == hid and g["ag"] > g["hg"])
        )
        a_wins = sum(
            1 for g in games
            if (g["home"] == aid and g["hg"] > g["ag"]) or
               (g["away"] == aid and g["ag"] > g["hg"])
        )
        draws = len(games) - h_wins - a_wins
        return {"games": len(games), "h_wins": h_wins, "a_wins": a_wins, "draws": draws}
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 4. Load predict.py model
    # ─────────────────────────────────────────────────────────────────────────────
    print("Loading prediction model from predict.py...")
    import predict as pred_mod
    
    team_ids = list(id_to_title.keys())
    print(f"  Prediction model ready — {len(team_ids)} teams\n")
    
    def get_prediction(hid, aid):
        try:
            feat  = pred_mod.compute_features(hid) + pred_mod.compute_features(aid) + [1/3]*3
            h_enc = int(pred_mod.team_le.transform([hid])[0])
            a_enc = int(pred_mod.team_le.transform([aid])[0])
            with torch.no_grad():
                lams = pred_mod.model(
                    torch.tensor([h_enc], dtype=torch.long),
                    torch.tensor([a_enc], dtype=torch.long),
                    torch.tensor([feat],  dtype=torch.float32),
                ).squeeze().tolist()
            lam_h, lam_a = lams
            hw, dr, aw = pred_mod.outcome_probs(lam_h, lam_a)
            return lam_h, lam_a, hw, dr, aw
        except Exception:
            return None
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 5. Answer generators
    #    Short, punchy answers — critical for TinyLlama.
    #    Every answer is under ~80 words so it fits in 512 tokens with the prompt.
    # ─────────────────────────────────────────────────────────────────────────────
    
    def prediction_answer(hid, aid):
        result = get_prediction(hid, aid)
        if result is None:
            return None
        lam_h, lam_a, hw, dr, aw = result
        scoreline, sl_prob = most_likely_scoreline(lam_h, lam_a)
        H = id_to_title[hid]
        A = id_to_title[aid]
    
        outcomes = ["a home win", "a draw", "an away win"]
        best     = outcomes[int(np.argmax([hw, dr, aw]))]
        h_trend  = get_form_trend(hid)
        a_trend  = get_form_trend(aid)
        h_streak = team_stats[hid]["current_streak"]
        a_streak = team_stats[aid]["current_streak"]
    
        # Core stats line — always included
        core = (
            f"{H} {hw*100:.0f}% | Draw {dr*100:.0f}% | {A} {aw*100:.0f}%. "
            f"xG: {H} {lam_h:.2f}, {A} {lam_a:.2f}. "
            f"Most likely scoreline: {scoreline[0]}-{scoreline[1]}. "
            f"Leans toward {best}."
        )
    
        # One contextual note — pick most interesting
        extras = []
        if h_streak >= 3:
            extras.append(f"{H} are on a {h_streak}-game winning streak.")
        elif h_streak <= -3:
            extras.append(f"{H} haven't won in {abs(h_streak)} straight.")
        if a_streak >= 3:
            extras.append(f"{A} arrive with {a_streak} wins on the bounce.")
        if h_trend == "excellent" and a_trend == "poor":
            extras.append(f"{H} are flying while {A} are struggling.")
        elif a_trend == "excellent" and h_trend == "poor":
            extras.append(f"{A} are in great form; {H} have been misfiring.")
    
        h2h = get_h2h_summary(hid, aid)
        if h2h and h2h["games"] >= 3:
            if h2h["h_wins"] > h2h["a_wins"] * 2:
                extras.append(
                    f"{H} dominate this fixture historically "
                    f"({h2h['h_wins']} wins in {h2h['games']} meetings)."
                )
            elif h2h["a_wins"] > h2h["h_wins"] * 2:
                extras.append(
                    f"{A} hold a strong H2H edge "
                    f"({h2h['a_wins']} wins in {h2h['games']} games)."
                )
    
        extra = (" " + extras[0]) if extras else ""
        return core + extra
    
    
    def form_answer(tid):
        s      = team_stats[tid]
        T      = id_to_title[tid]
        recent = s["recent"][-5:]
        if not recent:
            return None
    
        gf_avg   = np.mean([r["gf"] for r in recent])
        ga_avg   = np.mean([r["ga"] for r in recent])
        csr      = clean_sheet_rate(tid)
        h_wr     = home_win_rate(tid)
        a_wr     = away_win_rate(tid)
        streak   = s["current_streak"]
        trend    = get_form_trend(tid)
        form_str = "".join(r["result"] for r in recent)
    
        trend_phrases = {
            "excellent":    f"{T} are in excellent form",
            "good":         f"{T} are in solid form",
            "poor":         f"{T} are really struggling",
            "inconsistent": f"{T} have been very inconsistent",
            "mixed":        f"{T}'s form has been mixed",
        }
    
        core = (
            f"{trend_phrases[trend]}. "
            f"Last 5: {form_str} — {gf_avg:.1f} scored, {ga_avg:.1f} conceded per game. "
            f"Season: {s['w']}W {s['d']}D {s['l']}L, {s['pts']} pts, "
            f"GD {s['gf'] - s['ga']:+d}."
        )
    
        extras = []
        if streak >= 3:
            extras.append(f"On a {streak}-game winning streak.")
        elif streak <= -2:
            extras.append(f"No win in {abs(streak)} straight — confidence will be low.")
        if csr > 40:
            extras.append(f"Solid defensively — {csr:.0f}% clean sheet rate.")
        if h_wr > 65:
            extras.append(f"Strong at home: {h_wr:.0f}% win rate.")
        if a_wr > 50:
            extras.append(f"Travel well: {a_wr:.0f}% away win rate.")
    
        extra = (" " + extras[0]) if extras else ""
        return core + extra
    
    
    def h2h_answer(hid, aid):
        H    = id_to_title[hid]
        A    = id_to_title[aid]
        data = get_h2h_summary(hid, aid)
        if not data or data["games"] == 0:
            return f"No head-to-head data available for {H} vs {A} in this dataset."
    
        base = (
            f"In {data['games']} meetings: {H} {data['h_wins']} wins, "
            f"{A} {data['a_wins']} wins, {data['draws']} draws."
        )
        if data["h_wins"] > data["a_wins"] * 2:
            edge = f" {H} have the clear historical edge."
        elif data["a_wins"] > data["h_wins"] * 2:
            edge = f" {A} have dominated this fixture historically."
        elif data["draws"] >= data["games"] // 2:
            edge = " These two draw a lot — very even rivalry."
        else:
            edge = " It's been a balanced fixture overall."
    
        return base + edge
    
    
    def btts_answer(hid, aid):
        H   = id_to_title[hid]
        A   = id_to_title[aid]
        h_r = btts_rate(hid)
        a_r = btts_rate(aid)
        avg = (h_r + a_r) / 2
    
        verdict = "likely" if avg > 60 else ("unlikely" if avg < 40 else "50/50")
        return (
            f"BTTS looks {verdict} here. "
            f"{H} see both teams score in {h_r:.0f}% of games, "
            f"{A} in {a_r:.0f}%. Combined: {avg:.0f}%."
        )
    
    
    def table_answer(rows):
        top3  = rows[:3]
        bot3  = rows[-3:]
        top_s = ", ".join(f"{r['name']} ({r['pts']}pts)" for r in top3)
        bot_s = ", ".join(f"{r['name']} ({r['pts']}pts)" for r in bot3)
        return (
            f"Top 3: {top_s}. "
            f"Bottom 3: {bot_s}. "
            f"{rows[0]['name']} lead on {rows[0]['pts']} points, GD {rows[0]['gd']:+d}."
        )
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 6. Build league table
    # ─────────────────────────────────────────────────────────────────────────────
    table_rows = []
    for tid, s in team_stats.items():
        name = id_to_title.get(tid)
        if name and s["gp"] > 0:
            table_rows.append({
                "name": name,
                "pts":  s["pts"],
                "gd":   s["gf"] - s["ga"],
                "gp":   s["gp"],
                "w":    s["w"],
                "d":    s["d"],
                "l":    s["l"],
            })
    table_rows.sort(key=lambda r: (r["pts"], r["gd"]), reverse=True)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 7. TinyLlama ChatML format
    #
    #    TinyLlama-Chat uses the ChatML template:
    #
    #    <|system|>
    #    system prompt</s>
    #    <|user|>
    #    user message</s>
    #    <|assistant|>
    #    assistant reply</s>
    #
    #    This must match EXACTLY at inference time in chatbot.py.
    # ─────────────────────────────────────────────────────────────────────────────
    
    SYSTEM = (
        "You are an EPL football analyst chatbot. "
        "You give concise, accurate match predictions and team analysis "
        "using real statistics. You can also handle casual conversation naturally."
    )
    
    def chatml(user_msg, assistant_msg, include_system=False):
        """Format one training example in TinyLlama ChatML format."""
        if include_system:
            return (
                f"<|system|>\n{SYSTEM}</s>\n"
                f"<|user|>\n{user_msg}</s>\n"
                f"<|assistant|>\n{assistant_msg}</s>"
            )
        return (
            f"<|user|>\n{user_msg}</s>\n"
            f"<|assistant|>\n{assistant_msg}</s>"
        )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 8. Question templates
    # ─────────────────────────────────────────────────────────────────────────────
    
    PRED_Q = [
        "Who will win {h} vs {a}?",
        "Predict {h} vs {a}.",
        "Give me a prediction for {h} against {a}.",
        "Can {h} beat {a}?",
        "{h} vs {a} — who wins?",
        "What are the odds for {h} vs {a}?",
        "Break down {h} vs {a}.",
        "Will {h} beat {a}?",
        "What chance does {a} have at {h}?",
        "Who's favoured: {h} or {a}?",
    ]
    
    FORM_Q = [
        "How has {t} been playing?",
        "What's {t}'s form like?",
        "How is {t} doing this season?",
        "Is {t} in good form?",
        "Tell me about {t}'s recent results.",
        "What's {t}'s record like?",
        "Are {t} playing well?",
        "How have {t} been performing?",
    ]
    
    H2H_Q = [
        "What's the head-to-head between {h} and {a}?",
        "H2H record: {h} vs {a}?",
        "Who has the better record, {h} or {a}?",
        "Historical results between {h} and {a}?",
    ]
    
    BTTS_Q = [
        "Will both teams score in {h} vs {a}?",
        "BTTS prediction for {h} vs {a}?",
        "Is BTTS likely for {h} against {a}?",
        "Both teams to score: {h} vs {a}?",
    ]
    
    TABLE_Q = [
        "What's the league table?",
        "Who's top of the league?",
        "Show me the standings.",
        "Current league standings?",
        "Who's leading the league?",
        "What does the table look like?",
    ]
    
    EXPLAIN_Q = [
        "How does your prediction model work?",
        "What is xG?",
        "Explain expected goals.",
        "How do you predict matches?",
        "What's the Poisson model?",
        "How do you calculate win probabilities?",
        "What is expected goals and why does it matter?",
    ]
    
    EXPLAIN_A = [
        "I use a Poisson neural network trained on EPL data. Each team gets an expected goals value based on their form, xG history, and opponent quality. I then calculate the probability of every possible scoreline to get win/draw/loss odds.",
        "Expected goals (xG) measures shot quality — distance, angle, assist type. A tap-in might be 0.8 xG; a long-range effort 0.04. Over a season it predicts future results better than raw goals.",
        "Poisson modelling treats goals as rare random events. Given a team's expected rate (λ), I calculate the probability of scoring 0, 1, 2... goals. Combine both teams and you get the full scoreline matrix.",
        "Recent form gets exponentially higher weight — last match counts most, older games fade. I also factor in home/away splits, clean sheet rates, and head-to-head history.",
    ]
    
    TEAMS_Q = [
        "Which teams can you predict?",
        "What teams are available?",
        "List all the teams.",
        "Which clubs do you cover?",
        "What EPL teams do you know?",
    ]
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 9. Conversational examples
    #    Short replies — TinyLlama needs to learn brevity.
    #    Repeated 5x each so the model sees them often enough to stick.
    # ─────────────────────────────────────────────────────────────────────────────
    
    CONV_PAIRS = [
        # Greetings
        ("Hi!", "Hey! Ask me about any EPL fixture or team and I'll give you the stats."),
        ("Hello!", "Hello! I'm your EPL football analyst. What match do you want to look at?"),
        ("Hey!", "Hey! Got a fixture you want me to predict?"),
        ("Hi there!", "Hi! Ask me about predictions, form, or the league table."),
        ("Good morning!", "Morning! Any EPL fixtures you want previewed today?"),
        ("Good evening!", "Evening! Want a prediction or some form analysis?"),
        ("What's up?", "Just crunching EPL stats. What fixture do you want?"),
        ("How are you?", "Good thanks! Ready to analyse some football. What do you need?"),
        ("Hey, how's it going?", "Going well! What can I help with — prediction, form, H2H?"),
    
        # Farewells
        ("Bye!", "See you next match day!"),
        ("Thanks!", "No problem! Let me know if you want anything else."),
        ("Cheers!", "Cheers! Hope the prediction comes good."),
        ("Thanks, goodbye!", "No problem — enjoy the football!"),
        ("That's great, thanks!", "Glad I could help! Ask me anything else you need."),
        ("Nice one!", "Anytime! Good luck with it."),
    
        # What can you do
        ("What can you do?", "I can predict EPL matches, break down team form, show H2H records, calculate BTTS probability, and show the league table. Just ask about any team or fixture!"),
        ("What are you?", "I'm an EPL football prediction bot powered by a Poisson neural network and real match data. Ask me anything!"),
        ("Help", "Try: 'Predict Arsenal vs Chelsea', 'How is Liverpool doing?', 'H2H: Man City vs Man Utd', or 'Show me the table'."),
    
        # Off-topic
        ("What's the weather like?", "No idea about weather! Any fixture you want me to preview instead?"),
        ("What should I have for dinner?", "Outside my expertise! Want me to find a game worth watching tonight?"),
        ("Tell me a joke.", "Why did the footballer bring string? In case he tied the score! Now, any predictions you want?"),
        ("What's the meaning of life?", "Scoring more than the other side, probably. Want a prediction?"),
        ("Are you human?", "Nope — I'm a football analytics bot. Ask me about any EPL fixture!"),
        ("Who's the best player?", "I'll leave that to the pundits! Want to know which team has the best form instead?"),
        ("I'm bored.", "Nothing cures boredom like football stats! Which team do you want to start with?"),
        ("I'm stressed.", "Sorry to hear that. Want me to find a fixture to look forward to?"),
        ("What's in the news?", "I don't follow the news, but I know everything about EPL form. What do you want to know?"),
        ("Can you help me with something else?", "Football is my speciality — predictions, form, stats. What EPL question have you got?"),
        ("What's your favourite team?", "Strictly neutral — every team gets the same statistical treatment from me!"),
        ("Do you watch football?", "I can't watch, but I've processed every EPL match in the dataset. Ask me anything!"),
    
        # Follow-ups
        ("And their away form?", "Tell me which team and I'll break down their away record specifically."),
        ("Is that good?", "Depends on the context — tell me which team or stat and I'll explain."),
        ("What does that mean?", "Which part — the xG, win probability, or form rating? Happy to explain."),
        ("Can you explain more?", "Sure — which bit do you want explained?"),
        ("Who else could win it?", "Give me the two teams and I'll show you the full win/draw/loss breakdown."),
    ]
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 10. Build training corpus
    # ─────────────────────────────────────────────────────────────────────────────
    print("Generating training examples...")
    examples = []
    
    # --- Predictions (~40% of total) ---
    print("  Generating prediction examples...")
    pred_count = 0
    all_pairs  = [(h, a) for h in team_ids for a in team_ids if h != a]
    random.shuffle(all_pairs)
    
    for hid, aid in all_pairs:
        ans = prediction_answer(hid, aid)
        if ans is None:
            continue
        for _ in range(2):
            q = random.choice(PRED_Q).format(h=id_to_title[hid], a=id_to_title[aid])
            examples.append(chatml(q, ans, include_system=random.random() < 0.3))
            pred_count += 1
    print(f"    {pred_count} prediction examples")
    
    # --- Form (~15%) ---
    print("  Generating form examples...")
    form_count = 0
    for tid in team_ids:
        ans = form_answer(tid)
        if ans is None:
            continue
        for _ in range(3):
            q = random.choice(FORM_Q).format(t=id_to_title[tid])
            examples.append(chatml(q, ans, include_system=random.random() < 0.2))
            form_count += 1
    print(f"    {form_count} form examples")
    
    # --- H2H (~10%) ---
    print("  Generating H2H examples...")
    h2h_count = 0
    for hid in team_ids:
        for aid in [t for t in team_ids if t != hid]:
            data = get_h2h_summary(hid, aid)
            if data and data["games"] >= 2:
                ans = h2h_answer(hid, aid)
                q   = random.choice(H2H_Q).format(h=id_to_title[hid], a=id_to_title[aid])
                examples.append(chatml(q, ans))
                h2h_count += 1
    print(f"    {h2h_count} H2H examples")
    
    # --- BTTS (~5%) ---
    print("  Generating BTTS examples...")
    btts_count = 0
    for hid in random.sample(team_ids, min(12, len(team_ids))):
        for aid in random.sample([t for t in team_ids if t != hid], min(6, len(team_ids)-1)):
            ans = btts_answer(hid, aid)
            q   = random.choice(BTTS_Q).format(h=id_to_title[hid], a=id_to_title[aid])
            examples.append(chatml(q, ans))
            btts_count += 1
    print(f"    {btts_count} BTTS examples")
    
    # --- Table ---
    print("  Generating table examples...")
    t_ans = table_answer(table_rows)
    for q in TABLE_Q:
        for _ in range(4):
            examples.append(chatml(q, t_ans, include_system=True))
    print(f"    {len(TABLE_Q) * 4} table examples")
    
    # --- Explain ---
    print("  Generating explain examples...")
    exp_count = 0
    for q in EXPLAIN_Q:
        for ans in EXPLAIN_A:
            examples.append(chatml(q, ans))
            exp_count += 1
    print(f"    {exp_count} explain examples")
    
    # --- Teams list ---
    print("  Generating teams list examples...")
    team_list = ", ".join(sorted(id_to_title[t] for t in team_ids))
    teams_ans = f"I cover these EPL clubs: {team_list}. Ask me about any of them!"
    for q in TEAMS_Q:
        for _ in range(3):
            examples.append(chatml(q, teams_ans))
    print(f"    {len(TEAMS_Q) * 3} teams list examples")
    
    # --- Conversational (~20%) ---
    # Repeated 5x — TinyLlama needs more exposure to conversational
    # patterns or they get drowned out by the football examples
    print("  Generating conversational examples...")
    conv_count = 0
    for q, a in CONV_PAIRS:
        for _ in range(5):
            examples.append(chatml(q, a, include_system=random.random() < 0.3))
            conv_count += 1
    print(f"    {conv_count} conversational examples")
    
    random.shuffle(examples)
    print(f"\n  Total training examples: {len(examples)}\n")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 11. Load TinyLlama in 4-bit
    # ─────────────────────────────────────────────────────────────────────────────
    print("Loading TinyLlama-1.1B-Chat-v1.0...")
    print("(Using CPU-only mode)\n")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map=None,
    ).to("cpu")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 12. Apply LoRA adapters
    # ─────────────────────────────────────────────────────────────────────────────
    print("Applying LoRA adapters (r=8)...")
    
    lora_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = "CAUSAL_LM",
        target_modules = LORA_TARGET_MODULES,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 13. Tokenise dataset
    # ─────────────────────────────────────────────────────────────────────────────
    print("Tokenising dataset...")
    
    def tokenise(example):
        out = tokenizer(
            example["text"],
            truncation     = True,
            max_length     = MAX_SEQ_LENGTH,
            padding        = "max_length",
            return_tensors = "pt",
        )
        labels = out["input_ids"].clone()
        labels[out["attention_mask"] == 0] = -100
        out["labels"] = labels
        return {k: v.squeeze(0) for k, v in out.items()}
    
    raw_dataset = Dataset.from_dict({"text": examples})
    tokenised   = raw_dataset.map(
        tokenise,
        remove_columns = ["text"],
        desc           = "Tokenising",
        num_proc       = 0,
    )
    print(f"  Dataset: {len(tokenised)} examples tokenised\n")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 14. Train
    # ─────────────────────────────────────────────────────────────────────────────
    print("Starting QLoRA fine-tuning...")
    print(f"  Steps:           {TRAIN_STEPS}")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Estimated time:  20-30 mins on RTX 3070\n")
    
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        max_steps                   = TRAIN_STEPS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        warmup_steps                = WARMUP_STEPS,
        weight_decay                = WEIGHT_DECAY,
        lr_scheduler_type           = "cosine",
        max_grad_norm               = 1.0,
        fp16                        = False,
        optim                       = "adamw_torch",
        logging_steps               = 20,
        save_steps                  = 150,
        save_total_limit            = 2,
        report_to                   = "none",
        dataloader_pin_memory       = False,
        dataloader_num_workers      = 0,
        remove_unused_columns       = False,
        #group_by_length             = True,
    )
    
    # Custom training loop to avoid Trainer hang
    from torch.utils.data import DataLoader
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenised,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    
    total_loss = 0
    step = 0
    
    print("\nStarting training loop...\n")
    
    for epoch in range(10):  # Multiple epochs to reach ~300 steps
        for batch_idx, batch in enumerate(dataloader):
            if step >= TRAIN_STEPS:
                break
            
            batch = {k: v.to("cpu") for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            step += 1
            
            if step % 20 == 0:
                avg_loss = total_loss / 20
                print(f"Step {step}/{TRAIN_STEPS}, Loss: {avg_loss:.4f}")
                total_loss = 0
        
        if step >= TRAIN_STEPS:
            break
    
    print(f"\nTraining completed! Final step: {step}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 15. Save LoRA adapters
    # ─────────────────────────────────────────────────────────────────────────────
    print(f"\nSaving LoRA adapters to {OUTPUT_DIR}/...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print()
    print("=" * 60)
    print("  Training complete!")
    print(f"  Adapters saved to: {OUTPUT_DIR}/")
    print(f"  Adapter size:      ~40-60 MB")
    print(f"  Training examples: {len(examples)}")
    print(f"  Steps run:         {TRAIN_STEPS}")
    print()
    print("  Next steps:")
    print("  1. Commit ./lora_adapters/ to your Replit project")
    print("  2. Run chatbot.py")
    print("=" * 60)