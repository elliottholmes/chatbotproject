"""
Fine-tune distilgpt2 on a football prediction corpus generated from data.json.
Run this once to produce the llm_model/ directory used by llm_chat.py.
"""

import json
import random
import math
import os
import numpy as np
import torch
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

random.seed(42)

# ── Load match data ────────────────────────────────────────────────────────────
with open("data.json") as f:
    raw = json.load(f)

matches = [m for m in raw["results"] if m.get("isResult")]
matches.sort(key=lambda m: m["datetime"])

# Build team lookups
title_to_id = {}
id_to_title = {}
for m in matches:
    for side in ("h", "a"):
        title_to_id[m[side]["title"].lower()] = m[side]["id"]
        id_to_title[m[side]["id"]] = m[side]["title"]

all_team_titles = sorted(id_to_title.values())

# Per-team season stats
team_stats = defaultdict(lambda: {"w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0,
                                   "gp": 0, "pts": 0, "recent": []})
for m in matches:
    hid, aid = m["h"]["id"], m["a"]["id"]
    hg, ag   = int(m["goals"]["h"]), int(m["goals"]["a"])
    for tid, gf, ga in [(hid, hg, ag), (aid, ag, hg)]:
        s = team_stats[tid]
        s["gf"] += gf; s["ga"] += ga; s["gp"] += 1
        if gf > ga:  s["w"] += 1; s["pts"] += 3; r = "W"
        elif gf==ga: s["d"] += 1; s["pts"] += 1; r = "D"
        else:        s["l"] += 1;                 r = "L"
        s["recent"].append(f"{r} {gf}-{ga}")

# League table
table_rows = []
for tid, s in team_stats.items():
    gd = s["gf"] - s["ga"]
    table_rows.append((s["pts"], gd, s["gf"], id_to_title[tid], s))
table_rows.sort(reverse=True)

def table_str():
    lines = []
    for i, (pts, gd, gf, name, s) in enumerate(table_rows, 1):
        gd_s = f"+{gd}" if gd >= 0 else str(gd)
        lines.append(f"{i}. {name} — {s['gp']}GP  {pts}pts  GD:{gd_s}  "
                     f"W{s['w']} D{s['d']} L{s['l']}")
    return "\n".join(lines)

def form_str(tid):
    s = team_stats[tid]
    recent = s["recent"][-5:]
    avg_gf = s["gf"] / max(s["gp"], 1)
    avg_ga = s["ga"] / max(s["gp"], 1)
    return (f"{id_to_title[tid]} — W{s['w']} D{s['d']} L{s['l']} | "
            f"GF:{s['gf']} GA:{s['ga']} | Pts:{s['pts']} | "
            f"Recent: {' '.join(recent)} | "
            f"Avg scored:{avg_gf:.1f} conceded:{avg_ga:.1f}")

# ── Training corpus generation ──────────────────────────────────────────────────
SEP    = "\n### Bot: "
END    = "\n<|endoftext|>\n"

PRED_Qs = [
    "Who will win {h} vs {a}?",
    "Predict the result of {h} against {a}.",
    "What does the model say about {h} playing {a}?",
    "Can {h} beat {a}?",
    "What are {h}'s chances against {a}?",
    "Give me a prediction for {h} vs {a}.",
    "Who's favoured in {h} vs {a}?",
    "Break down the {h} vs {a} matchup.",
    "{h} host {a} — what's your prediction?",
    "Will {a} get a result at {h}?",
]

FORM_Qs = [
    "How has {t} been playing recently?",
    "What's {t}'s form like?",
    "How is {t} doing this season?",
    "Give me {t}'s season stats.",
    "Is {t} in good form?",
    "What's {t}'s record this season?",
    "Talk me through {t}'s season so far.",
    "How many points has {t} got?",
]

TABLE_Qs = [
    "Show me the league table.",
    "What does the league table look like?",
    "Who's top of the league?",
    "Who's in the relegation zone?",
    "Where does each team stand in the table?",
    "What are the current standings?",
    "Who's leading the league?",
]

EXPLAIN_Qs = [
    "How does the prediction model work?",
    "Explain how you generate predictions.",
    "What is Poisson regression in football?",
    "How do you calculate expected goals?",
    "What features does the model use?",
    "Why do you use temperature scaling?",
    "How does opponent quality affect the predictions?",
    "What is xG and why does it matter?",
    "How is the model trained?",
    "What does the temperature setting do?",
    "Why use a Poisson distribution for football?",
    "How does recent form factor into predictions?",
]

EXPLAIN_As = [
    ("The model is a PyTorch neural network trained on {n} EPL matches. It predicts λ (expected goals) "
     "for each team using Poisson NLL loss, then sums scoreline probabilities from 0-0 to 10-10 to give "
     "P(home win), P(draw), P(away win). Features include quality-adjusted goals, xG, win rate, opponent "
     "strength, and league table position. Recent matches get a 2.5x weight boost, older ones decay "
     "exponentially. Temperature {temp} flattens the output so it's not overconfident."),

    ("I use a Poisson neural network. Goals in football follow a Poisson distribution — they're rare, "
     "random events. The model learns expected goals (λ) for each team from {n} matches of data. "
     "Features per team: goals scored/conceded (quality-adjusted, decay-weighted), xG for/against, "
     "win/draw rate, opponent quality faced, and live league position. Temperature {temp} softens "
     "the final probabilities since football is inherently unpredictable."),

    ("Poisson regression models goals as random events with a rate parameter λ. The model predicts "
     "λ_home and λ_away, then computes every possible scoreline's probability using the Poisson PMF. "
     "Summing those gives clean W/D/L probabilities. It's been trained on {n} EPL results with "
     "exponential decay weighting — last 5 games get extra emphasis. Opponent quality adjusts each "
     "stat so beating a strong defence counts more than beating a weak one."),

    ("xG (expected goals) measures the quality of chances, not just the outcome. A team that creates "
     "high-xG chances consistently is likely to outperform their actual goal tally over time. "
     "The model uses xG alongside actual goals, both quality-adjusted by opponent strength, "
     "as core training features. This makes it more reliable than using raw goals alone."),
]

GENERAL_Qs = [
    ("What is this chatbot?",
     "I'm a football prediction chatbot powered by a fine-tuned language model and a Poisson neural network "
     "trained on EPL match data. Ask me to predict any match, check team form, show the league table, "
     "or explain how the model works."),
    ("What can you do?",
     "I can predict EPL match outcomes, show current league standings, analyse team form, "
     "and explain the Poisson model that powers my predictions."),
    ("What league do you cover?",
     "I'm trained on English Premier League (EPL) data for the current season."),
    ("How accurate are your predictions?",
     "Football is inherently unpredictable — even the best models top out around 55-60% accuracy on "
     "win/draw/loss classification. The model is honest about uncertainty via temperature scaling, "
     "so you'll rarely see extreme probabilities."),
    ("Who's the best team in the league?",
     f"Based on the data, Arsenal are top with the most points and best goal difference. "
     f"Manchester City are second. The table tells the full story — ask me to show it."),
]

def pred_answer(h_title, a_title, hw, dr, aw, lam_h, lam_a):
    outcomes = ["Home Win", "Draw", "Away Win"]
    best = outcomes[int(np.argmax([hw, dr, aw]))]
    openers = [
        f"The model gives {h_title} a {hw*100:.1f}% chance of winning at home, "
        f"a {dr*100:.1f}% chance of a draw, and {a_title} a {aw*100:.1f}% chance of winning.",

        f"Expected goals: {h_title} {lam_h:.2f} — {a_title} {lam_a:.2f}. "
        f"That translates to {hw*100:.1f}% home win, {dr*100:.1f}% draw, {aw*100:.1f}% away win.",

        f"Poisson model output — {h_title}: {lam_h:.2f} xG, {a_title}: {lam_a:.2f} xG. "
        f"Win probabilities: {h_title} {hw*100:.1f}%, Draw {dr*100:.1f}%, {a_title} {aw*100:.1f}%.",

        f"Looking at both teams' form, stats, and league position, the model predicts "
        f"{hw*100:.1f}% {h_title} win / {dr*100:.1f}% draw / {aw*100:.1f}% {a_title} win.",
    ]
    suffix_map = {
        "Home Win":  [f"Slight edge to {h_title} at home.",
                      f"{h_title} are the marginal favourites here.",
                      f"Home advantage tips it toward {h_title}."],
        "Draw":      ["It's a very open match — a draw looks most likely.",
                      "Hard to split these two — the draw is the most probable outcome.",
                      "Neither side looks dominant enough to guarantee a win."],
        "Away Win":  [f"{a_title} look capable of taking something here.",
                      f"Slight lean toward {a_title} despite playing away.",
                      f"The model tips {a_title} to get the result on the road."],
    }
    return random.choice(openers) + " " + random.choice(suffix_map[best])

# Build prediction examples using actual model outputs
print("Loading predict module for training data generation...")
import predict as pred_mod
print("Generating training corpus...\n")

examples = []

# 1. Prediction examples for all team pairs
team_ids = list(id_to_title.keys())
pairs_used = 0
for hid in team_ids:
    for aid in team_ids:
        if hid == aid:
            continue
        h_title = id_to_title[hid]
        a_title = id_to_title[aid]
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
            probs = pred_mod.outcome_probs(lam_h, lam_a)
            hw, dr, aw = probs

            q = random.choice(PRED_Qs).format(h=h_title, a=a_title)
            a = pred_answer(h_title, a_title, hw, dr, aw, lam_h, lam_a)
            examples.append(f"### User: {q}{SEP}{a}{END}")
            pairs_used += 1
        except Exception:
            pass

print(f"  Prediction pairs: {pairs_used}")

# 2. Form examples
for tid, s in team_stats.items():
    name = id_to_title[tid]
    recent = s["recent"][-5:]
    wr = s["w"] / max(s["gp"], 1)
    avg_gf = s["gf"] / max(s["gp"], 1)
    avg_ga = s["ga"] / max(s["gp"], 1)

    form_desc = [
        f"{name} have played {s['gp']} games this season, winning {s['w']}, drawing {s['d']}, "
        f"and losing {s['l']}. They've scored {s['gf']} and conceded {s['ga']} goals. "
        f"Recent results: {' | '.join(recent)}.",

        f"Season record for {name}: W{s['w']} D{s['d']} L{s['l']}, {s['pts']} points. "
        f"Averaging {avg_gf:.1f} goals scored and {avg_ga:.1f} conceded per game. "
        f"Last 5: {' '.join(recent)}.",

        f"{name} sit on {s['pts']} points from {s['gp']} matches. "
        f"Their goal difference is {s['gf']-s['ga']:+d}. "
        f"Recent form: {' '.join(recent)}. Win rate: {wr*100:.0f}%.",
    ]

    for q_tmpl in random.sample(FORM_Qs, min(3, len(FORM_Qs))):
        q = q_tmpl.format(t=name)
        a = random.choice(form_desc)
        examples.append(f"### User: {q}{SEP}{a}{END}")

print(f"  Form examples added. Total so far: {len(examples)}")

# 3. Table examples
table_text = table_str()
for q in TABLE_Qs:
    a = f"Here are the current standings:\n{table_text}"
    examples.append(f"### User: {q}{SEP}{a}{END}")

# 4. Explanation examples
n_matches = len(matches)
for q in EXPLAIN_Qs:
    a_tmpl = random.choice(EXPLAIN_As)
    a = a_tmpl.format(n=n_matches, temp=pred_mod.TEMPERATURE)
    examples.append(f"### User: {q}{SEP}{a}{END}")

# 5. General Q&As
for q, a in GENERAL_Qs:
    examples.append(f"### User: {q}{SEP}{a}{END}")

random.shuffle(examples)
print(f"  Total training examples: {len(examples)}\n")

# ── Tokenise & fine-tune ───────────────────────────────────────────────────────
MODEL_NAME = "distilgpt2"
SAVE_DIR   = "llm_model"

print(f"Loading base model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

corpus = "\n".join(examples)
tokenized = tokenizer(
    corpus,
    return_tensors="pt",
    truncation=False,
    padding=False,
)

# Split into chunks of 64 tokens — shorter = faster on CPU
CHUNK = 64
input_ids = tokenized["input_ids"][0]
chunks    = [input_ids[i:i+CHUNK] for i in range(0, len(input_ids) - CHUNK, CHUNK)]
# Cap at 300 chunks so training finishes in reasonable time
chunks = chunks[:300]
print(f"Training chunks: {len(chunks)}  (chunk size: {CHUNK})\n")

dataset = Dataset.from_dict({"input_ids": [c.tolist() for c in chunks]})

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    max_steps=40,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    warmup_steps=5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no",
    fp16=False,
    report_to="none",
    use_cpu=not torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("Fine-tuning distilgpt2 on football corpus...")
trainer.train()

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nFine-tuned model saved to {SAVE_DIR}/")
