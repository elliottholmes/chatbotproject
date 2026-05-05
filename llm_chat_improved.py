"""
Football chatbot powered by a fine-tuned distilgpt2 LLM.
IMPROVED VERSION: Better reasoning, stronger grounding, less hallucination.

Architecture:
  - predict.py computes all factual data (λ, probabilities, form, table).
  - The LLM receives a rich, grounded prompt with real numbers and stats.
  - For predictions, we GENERATE reasoning about WHY a team will win.
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

# ── Load fine-tuned LLM ───────────────────────────────────────────────────────
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

# ── Intent detection ────────────────────────────────────────────────────────
TABLE_WORDS  = ["table", "standings", "standing", "league table", "ranked", "top of"]
FORM_WORDS   = ["form", "recent", "how is", "how are", "doing", "results", "record",
                "performing", "season", "last few"]
EXPLAIN_WORDS= ["explain", "how does", "how do you", "why", "poisson", "neural",
                "algorithm", "temperature", "methodology", "model work", "trained"]
WHY_WIN_WORDS= ["why", "explain", "reason", "will win", "win", "beat"]

def detect_intent(text):
    t = text.lower()
    teams = extract_teams(t)
    if any(w in t for w in TABLE_WORDS):
        return "table"
    if any(w in t for w in FORM_WORDS) and len(teams) <= 1:
        return "form"
    if any(w in t for w in EXPLAIN_WORDS) and "win" not in t:
        return "explain"
    if len(teams) >= 2:
        return "predict"
    if any(w in t for w in ["list teams", "which teams", "all teams",
                              "what teams", "available teams", "all clubs"]):
        return "teams"
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

# ── Form helper ──────────────────────────────────────────────────────────
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

# ── Table helper ─────────────────────────────────────────────────────────
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

# ── LLM generation (completion mode) with STRICTER constraints ──────────────────
def generate_completion(anchor, max_new_tokens=50, temperature=0.65,
                        top_p=0.85, top_k=40):
    """
    Generate a continuation of `anchor`. The anchor is the start of the bot's
    reply, already containing all factual data — the LLM only adds commentary.
    
    IMPROVED: Lower temperature, stricter top_k/top_p, better stopping.
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
            repetition_penalty=1.5,  # Increased: punish repetition more
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
    r"\b(user|admin|system|root|bot)\b",  # system user strings
    r"(?:http|ftp|smtp|ssh)://",  # protocol markers
]
_JUNK_RE = re.compile("|".join(_JUNK_PATTERNS), re.IGNORECASE)

def filter_commentary(text, allowed_teams=None, min_length=20):
    """
    Clean and filter LLM commentary:
    - Remove URLs, hashtags, @mentions, code fragments, dates.
    - Drop sentences mentioning teams not in `allowed_teams`.
    - Require minimum sentence length for inclusion.
    - IMPROVED: Stricter filtering, reject more edge cases.
    """
    text = _JUNK_RE.sub("", text).strip()
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    if not text or len(text) < min_length:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        s = s.strip()
        if len(s) < min_length:
            continue
        if re.search(r"[{}\[\]\\|<>]", s):  # code-like chars
            continue
        # Reject all-caps or all-lowercase fragments
        if s.isupper() or (s.islower() and len(s) < 50):
            continue
        # Reject if mostly numbers
        if sum(c.isdigit() for c in s) / len(s) > 0.3:
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

# ── STRUCTURED REASONING for "why" predictions ──────────────────────────────────
def explain_team_advantage(team, opp, stats, prob):
    """
    Generate a STRUCTURED reason why `team` will beat `opp`.
    Uses actual stats — xG, form, position — not free-form LLM.
    """
    gf_team = stats["gf"]
    ga_team = stats["ga"]
    gf_opp = stats["opp_gf"]
    ga_opp = stats["opp_ga"]
    form_team = stats["form"]
    form_opp = stats["opp_form"]
    
    reasons = []
    
    # Attack-based reasons
    if gf_team > gf_opp * 1.1:
        reasons.append(f"{team.title()} averages {gf_team:.1f} goals per game, "
                       f"compared to {opp.title()}'s {gf_opp:.1f}.")
    
    # Defense-based reasons
    if ga_team < ga_opp * 0.9:
        reasons.append(f"Defensively, {team.title()} concedes just {ga_team:.1f} per game "
                       f"versus {opp.title()}'s {ga_opp:.1f}.")
    
    # Form-based reasons
    if form_team > form_opp:
        reasons.append(f"{team.title()} have won {form_team*100:.0f}% of recent matches, "
                       f"while {opp.title()} sit at {form_opp*100:.0f}%.")
    
    # Overall likelihood
    if prob > 0.60:
        reasons.append(f"The model gives {team.title()} a {prob*100:.0f}% win probability.")
    elif prob > 0.50:
        reasons.append(f"{team.title()} have a slight edge at {prob*100:.0f}% likelihood.")
    
    return reasons if reasons else [f"{team.title()} are favoured for this match."]

# ── Response builders ────────────────────────────────────────────────────────
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
    raw = generate_completion(anchor, max_new_tokens=50, temperature=0.60)
    commentary = filter_commentary(raw, allowed_teams=[home, away], min_length=20)

    return f"{factual}\n{commentary}"


def prediction_why_response(home, away):
    """
    IMPROVED: When user asks WHY a team will win, give structured reasoning.
    """
    result = run_prediction(home, away)
    if result is None:
        return f"Couldn't find both teams. Type 'list teams' to see what's available."

    lam_h, lam_a, (hw, dr, aw) = result
    H = home.title()
    A = away.title()

    # Determine winner
    if hw > max(dr, aw):
        winner = home
        prob = hw
        loser = away
    elif aw > max(hw, dr):
        winner = away
        prob = aw
        loser = home
    else:
        return prediction_response(home, away)  # Draw case, use default response

    # Get stats for reasoning
    home_form = get_form(home)
    away_form = get_form(away)
    if not home_form or not away_form:
        return prediction_response(home, away)

    stats = {
        "gf": home_form["gf_avg"],
        "ga": home_form["ga_avg"],
        "opp_gf": away_form["gf_avg"],
        "opp_ga": away_form["ga_avg"],
        "form": home_form["win_pct"] / 100,
        "opp_form": away_form["win_pct"] / 100,
    }

    reasons = explain_team_advantage(winner.title(), loser.title(), stats, prob)
    reason_text = " ".join(reasons[:2])  # Use top 2 reasons

    bar = lambda p: "█" * int(p * 30) + f"  {p*100:.1f}%"

    return (
        f"{H} vs {A}\n"
        f"  Expected goals: {H} {lam_h:.2f} — {A} {lam_a:.2f}\n\n"
        f"  Home Win  {bar(hw)}\n"
        f"  Draw      {bar(dr)}\n"
        f"  Away Win  {bar(aw)}\n\n"
        f"Why {winner.title()} will likely win:\n"
        f"  {reason_text}"
    )


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
    raw = generate_completion(anchor, max_new_tokens=45, temperature=0.60)
    commentary = filter_commentary(raw, allowed_teams=[team], min_length=15)
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
    raw = generate_completion(anchor, max_new_tokens=60, temperature=0.60)
    # For explain, reject any sentence that names a specific team
    llm_part = filter_commentary(raw, allowed_teams=[], min_length=15)

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
    
    # Check if this is a "why X will win Y" query
    is_why_query = any(w in user_input.lower() for w in ["why", "explain"]) and len(teams) >= 2

    if intent == "table":
        return table_response()

    if intent == "teams":
        return teams_response()

    if intent == "predict":
        if len(teams) < 2:
            return "I need two team names to make a prediction — who's playing?"
        # Use improved "why" response if appropriate
        if is_why_query:
            return prediction_why_response(teams[0], teams[1])
        return prediction_response(teams[0], teams[1])

    if intent == "form":
        if not teams:
            return "Which team's form would you like to check?"
        return form_response(teams[0])

    if intent == "explain":
        return explain_response()

    # General / fallback — return helpful guidance instead of free-form LLM
    return ("I'm a football prediction chatbot. I can:\n"
            "  • Predict matches: 'Liverpool vs Arsenal'\n"
            "  • Explain why a team will win: 'Why will Man City beat Brighton?'\n"
            "  • Check form: 'How is Liverpool doing?'\n"
            "  • Show standings: 'League table'\n"
            "  • Explain the model: 'How does the prediction work?'")

# ── Interactive loop ────────────────────────────────────────────────────────
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
