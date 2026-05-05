"""
Football chatbot powered by a fine-tuned distilgpt2 LLM.

Architecture:
  - predict.py computes all factual data (λ, probabilities, form, table).
  - The LLM receives a concise, grounded prompt anchored with real numbers
    and generates a natural-language continuation — preventing hallucination.
  - For structured outputs (table, teams list) the LLM is bypassed entirely.
"""

import os
import re
import math
import random
import torch
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

import predict as pred

MODEL_DIR = "llm_model"

# ── Load fine-tuned LLM ────────────────────────────────────────────────────────
print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token
llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
llm.eval()
print("Model ready.\n")

# ── Team name fuzzy matching ───────────────────────────────────────────────────
TEAMS = sorted(pred.title_to_id.keys())   # lowercase titles

def fuzzy_match(phrase, threshold=0.55):
    phrase = phrase.lower().strip()
    best, best_score = None, 0.0
    for c in TEAMS:
        score = SequenceMatcher(None, phrase, c).ratio()
        if phrase in c or c in phrase:
            score = max(score, 0.70)
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= threshold else None

_CONNECTORS = {"vs", "v", "versus", "against", "and", "host", "hosting",
               "play", "playing", "beats", "beat", "face", "faces"}

def extract_teams(text):
    text_clean = re.sub(r"[^a-z\s]", "", text.lower())
    words = text_clean.split()
    found = []
    for length in range(5, 0, -1):
        for start in range(len(words) - length + 1):
            phrase_words = words[start : start + length]
            # Skip any phrase that contains a connector word
            if any(w in _CONNECTORS for w in phrase_words):
                continue
            phrase = " ".join(phrase_words)
            m = fuzzy_match(phrase)
            if m and m not in found:
                found.append(m)
            if len(found) == 2:
                return found
    return found

# ── Intent detection ───────────────────────────────────────────────────────────
TABLE_WORDS  = ["table", "standings", "standing", "league table", "ranked", "top of"]
FORM_WORDS   = ["form", "recent", "how is", "how are", "doing", "results", "record",
                "performing", "season", "last few"]
EXPLAIN_WORDS= ["explain", "how does", "how do you", "why", "poisson", "neural",
                "algorithm", "temperature", "methodology", "model work", "trained"]

def detect_intent(text):
    t = text.lower()
    teams = extract_teams(t)
    if any(w in t for w in TABLE_WORDS):
        return "table"
    if any(w in t for w in FORM_WORDS) and len(teams) <= 1:
        return "form"
    if any(w in t for w in EXPLAIN_WORDS):
        return "explain"
    if any(w in t for w in ["list teams", "which teams", "all teams",
                              "what teams", "available teams", "all clubs"]):
        return "teams"
    if len(teams) >= 2:
        return "predict"
    if any(w in t for w in ["predict", "vs", "versus", "against", "win",
                              "match", "beat", "result", "odds", "chance"]):
        return "predict"
    return "general"

# ── Prediction helpers ────────────────────────────────────────────────────────
def run_prediction(home, away):
    hid = pred.title_to_id.get(home)
    aid = pred.title_to_id.get(away)
    if not hid or not aid:
        return None
    feat  = pred.compute_features(hid) + pred.compute_features(aid) + [1/3]*3
    h_enc = int(pred.team_le.transform([hid])[0])
    a_enc = int(pred.team_le.transform([aid])[0])
    with torch.no_grad():
        lams = pred.model(
            torch.tensor([h_enc], dtype=torch.long),
            torch.tensor([a_enc], dtype=torch.long),
            torch.tensor([feat],  dtype=torch.float32),
        ).squeeze().tolist()
    lam_h, lam_a = lams
    probs = pred.outcome_probs(lam_h, lam_a)
    return lam_h, lam_a, probs

# ── Form helper ────────────────────────────────────────────────────────────────
def get_form(team):
    tid = pred.title_to_id.get(team)
    if not tid:
        return None
    hist = list(pred.team_history.get(tid, []))
    if not hist:
        return None
    recent = hist[-5:]
    results = []
    for h in recent:
        r = "W" if h["won"] else ("D" if h["drew"] else "L")
        results.append(f"{r}({h['gf']}-{h['ga']})")
    gf_avg = np.mean([h["gf"] for h in recent])
    ga_avg = np.mean([h["ga"] for h in recent])
    t = pred.final_table.get(tid, {"pts": 0, "gd": 0, "gp": 0, "w": 0})
    wins = t.get("w", sum(1 for h in hist if h["won"]))
    total = t.get("gp", len(hist))
    wr = wins / max(total, 1)
    return {
        "name":    team.title(),
        "results": results,
        "gf_avg":  gf_avg,
        "ga_avg":  ga_avg,
        "pts":     t["pts"],
        "gd":      t["gd"],
        "win_pct": wr * 100,
    }

# ── Table helper ───────────────────────────────────────────────────────────────
def build_table():
    table = pred.final_table
    rows  = []
    for tid, t in table.items():
        name = None
        for title, i in pred.title_to_id.items():
            if i == tid:
                name = title.title()
                break
        if name and t["gp"] > 0:
            rows.append((t["pts"], t["gd"], t["gp"], name,
                         t.get("w", 0), t.get("d", 0), t.get("l", 0)))
    rows.sort(reverse=True)
    lines = []
    for pos, (pts, gd, gp, name, w, d, l) in enumerate(rows, 1):
        gd_str = f"+{gd}" if gd >= 0 else str(gd)
        lines.append(
            f"{pos:>2}. {name:<28} {gp:>2}GP  {pts:>3}pts  GD:{gd_str}"
        )
    return "\n".join(lines)

# ── LLM generation (completion mode) ──────────────────────────────────────────
def generate_completion(anchor, max_new_tokens=60, temperature=0.80,
                        top_p=0.90, top_k=50):
    """
    Generate a continuation of `anchor`. The anchor is the start of the bot's
    reply, already containing all factual data — the LLM only adds commentary.
    """
    inputs = tokenizer(
        anchor, return_tensors="pt",
        truncation=True, max_length=400
    )
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = llm.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            repetition_penalty=1.35,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = tokenizer.decode(
        output[0][input_len:], skip_special_tokens=True
    )
    # Cut at stop markers
    for stop in ["### User:", "### Bot:", "<|endoftext|>", "\n\n"]:
        if stop in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(stop)]

    # Keep only the first two complete sentences
    sentences = re.split(r"(?<=[.!?])\s+", new_tokens.strip())
    clean = " ".join(sentences[:2]).strip()
    return clean


_JUNK_PATTERNS = [
    r"https?://\S+",           # URLs
    r"@\w+",                   # @mentions
    r"#\w+",                   # hashtags
    r"\(\s*@\w+\s*\)",         # (@user)
    r"pic[,\.]?\s*twitter\S*", # pic.twitter links
    r"[-+]?[A-Za-z]?=\d+",    # code-like tokens e.g. P=3, GWS
    r"\d{1,2}\s+\d{1,2}\s+\d{1,2}",  # sequences of bare numbers
    r"\b[A-Z][a-z]+ \d{1,2},\s*\d{4}\b",  # dates
    r"github\S*|gitlab\S*",    # code repo links
    r"@?\S+@\S+\.\S+",         # email-like strings
]
_JUNK_RE = re.compile("|".join(_JUNK_PATTERNS), re.IGNORECASE)

def filter_commentary(text, allowed_teams=None):
    """
    Clean and filter LLM commentary:
    - Remove URLs, hashtags, @mentions, code fragments, dates.
    - Drop sentences mentioning teams not in `allowed_teams`.
    - Require minimum sentence length for inclusion.
    """
    text = _JUNK_RE.sub("", text).strip()
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:          # too short — likely a fragment
            continue
        if re.search(r"[{}\[\]\\|<>]", s):  # code-like chars
            continue
        s_lower = s.lower()
        if allowed_teams is not None:
            mentions_wrong = any(
                t in s_lower and t not in allowed_teams
                for t in TEAMS
            )
            if mentions_wrong:
                continue
        kept.append(s)

    return " ".join(kept[:2]).strip()

# ── Response builders ─────────────────────────────────────────────────────────
def prediction_response(home, away):
    result = run_prediction(home, away)
    if result is None:
        return f"Couldn't find both teams. Type 'list teams' to see what's available."

    lam_h, lam_a, (hw, dr, aw) = result
    H = home.title()
    A = away.title()

    outcomes  = ["Home Win", "Draw", "Away Win"]
    best_idx  = int(np.argmax([hw, dr, aw]))
    best      = outcomes[best_idx]
    leaders   = [H, "Either side", A]
    leader    = leaders[best_idx]

    bar = lambda p: "█" * int(p * 30) + f"  {p*100:.1f}%"

    # Fixed factual block
    factual = (
        f"{H} vs {A}\n"
        f"  Expected goals: {H} {lam_h:.2f} — {A} {lam_a:.2f}\n"
        f"\n"
        f"  Home Win  {bar(hw)}\n"
        f"  Draw      {bar(dr)}\n"
        f"  Away Win  {bar(aw)}\n"
    )

    # LLM anchored commentary — start the sentence for it
    anchor_templates = [
        f"### User: Predict {H} vs {A}.\n### Bot: The model gives {H} a {hw*100:.1f}% win chance versus {aw*100:.1f}% for {A}.",
        f"### User: Who will win {H} vs {A}?\n### Bot: Expected goals of {lam_h:.2f} for {H} and {lam_a:.2f} for {A} —",
        f"### User: {H} host {A} — prediction?\n### Bot: With {H} at {lam_h:.2f} xG and {A} at {lam_a:.2f} xG,",
    ]
    anchor = random.choice(anchor_templates)
    raw = generate_completion(anchor, max_new_tokens=50)
    commentary = filter_commentary(raw, allowed_teams=[home, away])

    return f"{factual}\n{commentary}"


def form_response(team):
    f = get_form(team)
    if f is None:
        return f"I don't have form data for {team.title()}."

    results_str = "  ".join(f["results"])
    factual = (
        f"{f['name']} — recent form:\n"
        f"  {results_str}\n"
        f"  Avg scored: {f['gf_avg']:.1f}  |  Avg conceded: {f['ga_avg']:.1f}\n"
        f"  Points: {f['pts']}  |  GD: {f['gd']:+d}  |  Win rate: {f['win_pct']:.0f}%\n"
    )

    trend = "good" if f["win_pct"] >= 60 else ("poor" if f["win_pct"] < 35 else "mixed")
    anchor = (
        f"### User: How has {f['name']} been doing recently?\n"
        f"### Bot: {f['name']} have shown {trend} form recently —"
    )
    raw = generate_completion(anchor, max_new_tokens=45)
    commentary = filter_commentary(raw, allowed_teams=[team])
    return f"{factual}\n{commentary}"


def explain_response():
    T = pred.TEMPERATURE

    anchor_options = [
        (f"### User: How does the prediction model work?\n"
         f"### Bot: The model is a Poisson neural network."),

        (f"### User: Explain how you generate predictions.\n"
         f"### Bot: Goals in football follow a Poisson distribution — rare, independent events."),

        (f"### User: What does temperature scaling do?\n"
         f"### Bot: Temperature {T} divides the logits before the softmax,"),
    ]
    anchor = random.choice(anchor_options)
    raw = generate_completion(anchor, max_new_tokens=60)
    # For explain, reject any sentence that names a specific team
    llm_part = filter_commentary(raw, allowed_teams=[])

    fixed = (
        f"Key facts:\n"
        f"  • Trained on 349 EPL matches with exponential decay weighting\n"
        f"  • Features: goals, xG, win rate, opponent quality, league position\n"
        f"  • Recent 5 matches get a 2.5× boost; older matches decay at rate 0.12\n"
        f"  • Temperature {T} softens probabilities — football is inherently unpredictable\n"
        f"  • Poisson grid (0–0 to 10–10) gives P(home win), P(draw), P(away win)"
    )
    intro = llm_part if llm_part else f"The Poisson model predicts expected goals (λ) for each team, then maps every scoreline's probability to a clean W/D/L distribution."
    return f"{intro}\n\n{fixed}"


def table_response():
    return (
        "Current league standings (from season data):\n\n"
        f"{build_table()}"
    )


def teams_response():
    team_list = "  ".join(t.title() for t in sorted(TEAMS))
    return f"Available clubs:\n{team_list}"

# ── Main chat entry point ──────────────────────────────────────────────────────
def chat(user_input):
    intent = detect_intent(user_input)
    teams  = extract_teams(user_input.lower())

    if intent == "table":
        return table_response()

    if intent == "teams":
        return teams_response()

    if intent == "predict":
        if len(teams) < 2:
            return "I need two team names to make a prediction — who's playing?"
        return prediction_response(teams[0], teams[1])

    if intent == "form":
        if not teams:
            return "Which team's form would you like to check?"
        return form_response(teams[0])

    if intent == "explain":
        return explain_response()

    # General / fallback — LLM free generation
    anchor = f"### User: {user_input}\n### Bot:"
    response = generate_completion(anchor, max_new_tokens=80)
    if not response or len(response) < 15:
        return ("I'm a football prediction chatbot. Ask me to predict a match, "
                "check team form, show the league table, or explain how I work.")
    return response

# ── Interactive loop ───────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  EPL Football Chatbot  |  Fine-tuned LLM + Poisson Model")
    print("  Ask: predict matches · check form · standings · explain")
    print("  Type 'quit' to exit.")
    print("=" * 60)
    print()

    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in ("quit", "exit", "bye", "q"):
                print("Bot: Cheers. See you next match day.")
                break
            response = chat(user)
            print(f"\nBot: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye.")
            break

if __name__ == "__main__":
    run()
