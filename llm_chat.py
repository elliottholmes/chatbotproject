"""
Football chatbot — fully LLM-driven responses.

Uses base TinyLlama-1.1B-Chat-v1.0 with NO fine-tuning.

Real data (xG, probabilities, form, table) is injected into the prompt as
grounding context, then the LLM writes the entire reply in natural language.
The model can't hallucinate numbers because every fact is pinned in the prompt.
"""

import os
import re
import math
import random
import torch
import numpy as np
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM

import predict as pred

print('Imported')

MODEL_DIR = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ── Load model ──────────────────────────────────────────────────────────
print(f"Loading base model from {MODEL_DIR}...")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token
llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)
llm.eval()
print(f"Model ready on {device}.\n")

TEAMS = sorted(pred.title_to_id.keys())

TEAM_ALIASES = {
    "wolves": "wolverhampton wanderers",
    "spurs": "tottenham hotspur",
    "man u": "manchester united",
    "man city": "manchester city",
    "newcastle": "newcastle united",
    "west ham": "west ham united",
    "brighton": "brighton & hove albion",
    "palace": "crystal palace",
    "bournemouth": "afc bournemouth",
    "fulham": "fulham",
    "liverpool": "liverpool",
    "chelsea": "chelsea",
    "arsenal": "arsenal",
    "everton": "everton",
    "burnley": "burnley",
}

# ── Hard system prefix injected into every prompt ───────────────────────────
SYSTEM_PREFIX = (
    "### System: You are a football analysis engine. "
    "You MUST ONLY use the DATA provided. "
    "Do NOT add any external knowledge, history, or context. "
    "Do NOT mention teams not in the DATA. "
    "Do NOT infer anything not explicitly stated. "
    "If it is not in the DATA, you cannot say it.\n"
    "Always include:\n"
    "- Winner\n"
    "- Scoreline in format X-Y\n"
    "Be direct and deterministic.\n"
    """Explain ONLY using the provided numbers.
Do not mention any teams outside this match.
Do not add historical or external context."""
)

# ── Blocked phrases — any sentence containing these is dropped ───────────────
_BLOCK_PHRASES = [
    "bet365", "william hill", "betfair", "paddy power", "ladbrokes",
    "sky bet", "coral", "unibet", "draftkings", "fanduel",
    "sportsbook", "bookmaker", "bookie", "betting shop", "betting site",
    "betting platform", "betting exchange", "betting app", "betting odds",
    "place a bet", "place your bet", "have a bet", "make a bet",
    "accumulator", "each-way", "each way", "ante-post",
    "i'm not sure", "i am not sure", "i'm not certain", "i cannot be sure",
    "i don't have", "i do not have", "i lack", "without more data",
    "i cannot guarantee", "i can't guarantee", "no guarantees",
    "for entertainment", "not financial advice", "not a financial",
    "consult a professional", "disclaimer", "please note that",
    "it's worth noting", "it is worth noting",
    "i would recommend checking", "you might want to check",
    "you should check", "you could check", "check out",
    "visit ", "browse ", "explore other",
]

_HEDGE_RE = re.compile(
    r"\b(I'?m not (sure|certain|able|confident)|"
    r"I (don'?t|cannot|can'?t) (say|tell|know|confirm|predict)|"
    r"hard to (say|predict|tell)|"
    r"difficult to (say|predict|tell)|"
    r"take (this|it) with|"
    r"grain of salt|"
    r"just (a |my )?(opinion|guess|estimate))\b",
    re.IGNORECASE,
)

# ── Fuzzy team matching ───────────────────────────────────────────────────────
def fuzzy_match(phrase, threshold=0.55):
    phrase = phrase.lower().strip()
    if phrase in TEAM_ALIASES:
        return TEAM_ALIASES[phrase]
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
            phrase_words = words[start: start + length]
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
TABLE_WORDS   = ["table", "standings", "standing", "league table", "ranked", "top of", "who's top"]
FORM_WORDS    = ["form", "recent", "how is", "how are", "doing", "results", "record",
                 "performing", "season", "last few", "run of"]
EXPLAIN_WORDS = ["explain", "how does", "how do you", "why", "poisson", "neural",
                 "algorithm", "temperature", "methodology", "model work", "trained",
                 "xg", "expected goals", "what is xg"]
TEAMS_WORDS   = ["list teams", "which teams", "all teams", "what teams",
                 "available teams", "all clubs", "what clubs"]
BTTS_WORDS    = ["both teams score", "btts", "both score", "both net"]
H2H_WORDS     = ["head to head", "h2h", "history between", "historical record",
                 "past meetings", "past results between"]

def detect_intent(text):
    t = text.lower()
    teams = extract_teams(t)
    
    # 🔥 FORCE table detection first (important)
    if any(w in t for w in ["table", "standings", "league table", "epl table", "premier league table"]):
        return "table"

    if any(w in t for w in TEAMS_WORDS):
        return "teams"

    if any(w in t for w in EXPLAIN_WORDS):
        return "explain"

    if any(w in t for w in H2H_WORDS) and len(teams) == 2:
        return "h2h"

    if any(w in t for w in BTTS_WORDS) and len(teams) == 2:
        return "btts"

    if any(w in t for w in FORM_WORDS) and len(teams) <= 1:
        return "form"

    if len(teams) >= 2:
        return "predict"

    if any(w in t for w in ["predict", "vs", "versus", "against", "win", "match"]):
        return "predict"

    return "general"

# ── Data gathering ─────────────────────────────────────────────────────────

def get_prediction_data(home, away):
    result = pred.predict(home, away)
    if result is None:
        return None

    lam_h = result["lam_h"]
    lam_a = result["lam_a"]
    hw, dr, aw = result["probs"]

    return lam_h, lam_a, hw, dr, aw

def most_likely_scoreline(lam_h, lam_a, max_g=6):
    best_p, best = -1, (0, 0)
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = (math.exp(-lam_h) * lam_h**i / math.factorial(i)) * \
                (math.exp(-lam_a) * lam_a**j / math.factorial(j))
            if p > best_p:
                best_p, best = p, (i, j)
    return best, best_p

def get_team_data(team):
    tid = pred.title_to_id.get(team)
    if not tid:
        return None
    hist   = list(pred.team_history.get(tid, []))
    recent = hist[-5:]
    if not recent:
        return None
    t      = pred.final_table.get(tid, {"pts": 0, "gd": 0, "gp": 0, "w": 0, "d": 0, "l": 0})
    wins   = t.get("w", sum(1 for h in hist if h["won"]))
    total  = t.get("gp", len(hist))
    wr     = wins / max(total, 1)
    gfpg   = sum(h["gf"] for h in recent) / len(recent)
    gapg   = sum(h["ga"] for h in recent) / len(recent)
    form_s = "".join("W" if h["won"] else ("D" if h["drew"] else "L") for h in recent)
    streak_val = 0
    for h in reversed(hist):
        r = "W" if h["won"] else ("D" if h["drew"] else "L")
        if streak_val == 0:
            streak_val = 1 if r == "W" else (-1 if r == "L" else 0)
        elif streak_val > 0 and r == "W":
            streak_val += 1
        elif streak_val < 0 and r == "L":
            streak_val -= 1
        else:
            break
    return {
        "name":    team.title(),
        "form_s":  form_s,
        "gfpg":    gfpg,
        "gapg":    gapg,
        "pts":     t["pts"],
        "gd":      t["gd"],
        "gp":      total,
        "w":       wins,
        "d":       t.get("d", 0),
        "l":       t.get("l", 0),
        "win_pct": wr * 100,
        "streak":  streak_val,
    }

def get_table_data():
    rows = []
    for tid, t in pred.final_table.items():
        name = next((title.title() for title, i in pred.title_to_id.items() if i == tid), None)
        if name and t["gp"] > 0:
            rows.append({
                "name": name, "pts": t["pts"], "gd": t["gd"],
                "gp": t["gp"], "w": t.get("w", 0),
                "d": t.get("d", 0), "l": t.get("l", 0),
            })
    rows.sort(key=lambda r: (r["pts"], r["gd"]), reverse=True)
    return rows

def get_h2h_data(home, away):
    hid = pred.title_to_id.get(home)
    aid = pred.title_to_id.get(away)
    if not hid or not aid:
        return None
    games = []
    for m in getattr(pred, "_all_matches", []):
        mhid = m["h"]["id"]
        maid = m["a"]["id"]
        if (mhid == hid and maid == aid) or (mhid == aid and maid == hid):
            games.append(m)
    if not games:
        return None
    h_wins = sum(1 for g in games if
                 (g["h"]["id"] == hid and int(g["goals"]["h"]) > int(g["goals"]["a"])) or
                 (g["a"]["id"] == hid and int(g["goals"]["a"]) > int(g["goals"]["h"])))
    a_wins = sum(1 for g in games if
                 (g["h"]["id"] == aid and int(g["goals"]["h"]) > int(g["goals"]["a"])) or
                 (g["a"]["id"] == aid and int(g["goals"]["a"]) > int(g["goals"]["h"])))
    draws  = len(games) - h_wins - a_wins
    avg_hg = np.mean([int(g["goals"]["h"]) + int(g["goals"]["a"]) for g in games])
    return {
        "games": len(games), "h_wins": h_wins,
        "a_wins": a_wins,    "draws": draws,
        "avg_total_goals": avg_hg,
    }

# ── Output sanitisation ────────────────────────────────────────────────────

_JUNK_RE = re.compile(
    r"https?://\S+|@\w+|#\w+|\(\s*@\w+\s*\)|pic[,\.]?\s*twitter\S*"
    r"|[-+]?[A-Za-z]?=\d+|\d{1,2}\s+\d{1,2}\s+\d{1,2}"
    r"|\b[A-Z][a-z]+ \d{1,2},\s*\d{4}\b|github\S*|gitlab\S*|@?\S+@\S+\.\S+",
    re.IGNORECASE,
)

def _sentence_is_clean(s, allowed_teams=None):
    """Return False if the sentence should be dropped."""
    low = s.lower()

    # Block any gambling / hedging phrase
    if any(p in low for p in _BLOCK_PHRASES):
        return False

    # Block hedging patterns via regex
    if _HEDGE_RE.search(s):
        return False

    # Block sentences that smuggle in non-allowed team names
    if allowed_teams is not None:
        for t in TEAMS:
            if t in low and t not in allowed_teams:
                return False

    # Drop very short or symbol-heavy sentences
    if len(s) < 20:
        return False
    if re.search(r"[{}\[\]\\|<>]", s):
        return False

    return True

def clean_output(text, allowed_teams=None):
    text = _JUNK_RE.sub("", text).strip()
    text = re.sub(r" {2,}", " ", text)
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s.strip() for s in sentences if _sentence_is_clean(s.strip(), allowed_teams)]
    text = " ".join(kept[:2]).strip()

    # hard kill any remaining weird phrasing
    banned = ["example", "for instance", "such as", "in conclusion"]
    for b in banned:
        if b in text.lower():
            return ""

    return text

# ── LLM generation ─────────────────────────────────────────────────────────

def llm_respond(prompt, max_new_tokens=120, temperature=0.75,
                top_p=0.90, top_k=50, allowed_teams=None):
    """
    Feed a grounded prompt to the base LLM and return clean output.
    The system prefix steers the model away from gambling / hedging.
    Lower temperature (0.75 vs 0.82) keeps outputs more decisive.
    """
    full_prompt = SYSTEM_PREFIX + prompt

    inputs = tokenizer(full_prompt, return_tensors="pt",
                       truncation=True, max_length=600).to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = llm.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            repetition_penalty=1.35,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    for stop in ["### User:", "### Bot:", "### System:", "<|endoftext|>", "\n\n"]:
        if stop in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(stop)]

    return clean_output(new_tokens, allowed_teams=allowed_teams)

# ── Prompt builders ───────────────────────────────────────────────────────────

def prediction_prompt(home, away):
    d = get_prediction_data(home, away)
    if d is None:
        return None, None, None
    lam_h, lam_a, hw, dr, aw = d
    scoreline, sl_prob = most_likely_scoreline(lam_h, lam_a)
    outcomes   = ["home win", "draw", "away win"]
    best       = outcomes[int(np.argmax([hw, dr, aw]))]
    H, A       = home.title(), away.title()

    hd = get_team_data(home)
    ad = get_team_data(away)

    home_context = ""
    away_context = ""
    if hd:
        home_context = (
            f"{H} recent form: {hd['form_s']}, averaging {hd['gfpg']:.1f} scored "
            f"and {hd['gapg']:.1f} conceded per game. Win rate {hd['win_pct']:.0f}%."
        )
    if ad:
        away_context = (
            f"{A} recent form: {ad['form_s']}, averaging {ad['gfpg']:.1f} scored "
            f"and {ad['gapg']:.1f} conceded per game. Win rate {ad['win_pct']:.0f}%."
        )

    streak_note = ""
    if hd and hd["streak"] >= 3:
        streak_note = f"{H} are on a {hd['streak']}-game winning run."
    elif hd and hd["streak"] <= -2:
        streak_note = f"{H} have lost their last {abs(hd['streak'])} games."
    elif ad and ad["streak"] >= 3:
        streak_note = f"{A} arrive on a {ad['streak']}-match winning streak."

    grounding = (
        f"DATA:\n"
        f"{H} vs {A}\n"
        f"xG: {lam_h:.2f} vs {lam_a:.2f}\n"
        f"Win probabilities: {hw*100:.1f}% / {dr*100:.1f}% / {aw*100:.1f}%\n"
        f"Most likely scoreline: {scoreline[0]}-{scoreline[1]}\n"
        f"Verdict: {best}\n"
    )

    prompt = (
        f"### User: Explain this prediction.\n"
        f"### Bot: {grounding}"
        f"Explain WHY this result is likely using ONLY this data. "
        f"Do not mention any teams outside this match. "
        f"Do not invent history. "
        f"Do not repeat the numbers. "
        f"Do not say 'the data shows' or 'the prediction says'. "
        f"Directly explain the football reasoning.\n"
    )

    # Prompts open mid-confident-sentence so TinyLlama continues in that register
    templates = [
         f"""### User: Who will win {H} vs {A}?
        ### Bot: {grounding}

        Final prediction:
        Winner: {best}
        Scoreline: {scoreline[0]}-{scoreline[1]}

        Analysis:""",
         f"""### User: Who will win {H} vs {A}?
        ### Bot:
        {grounding}

        Output:
        Winner: {best}
        Scoreline: {scoreline[0]}-{scoreline[1]}
        Reason:"""
    ]
    prompt = random.choice(templates)
    return prompt, [home, away], (lam_h, lam_a, hw, dr, aw, scoreline)

def form_prompt(team):
    d = get_team_data(team)
    if d is None:
        return None, None
    T = team.title()
    trend = "excellent" if d["win_pct"] >= 65 else \
            "poor"      if d["win_pct"] < 35 else "mixed"
    streak_note = ""
    if d["streak"] >= 3:
        streak_note = f"They are on a {d['streak']}-game winning streak."
    elif d["streak"] <= -2:
        streak_note = f"They have lost their last {abs(d['streak'])} matches."

    grounding = (
        f"DATA: {T} form. Last 5 results: {d['form_s']}. "
        f"Goals scored per game: {d['gfpg']:.1f}. Goals conceded per game: {d['gapg']:.1f}. "
        f"Season: {d['w']}W {d['d']}D {d['l']}L, {d['pts']} points, GD {d['gd']:+d}. "
        f"Win rate: {d['win_pct']:.0f}%. Form trend: {trend}. {streak_note}"
    ).strip()

    templates = [
        f"### User: How has {T} been playing?\n### Bot: {grounding} {T} are in {trend} form —",
        f"### User: What's {T}'s form like recently?\n### Bot: {grounding} The data shows {T} with a {d['win_pct']:.0f}% win rate, and",
        f"### User: Tell me about {T}'s season.\n### Bot: {grounding} {T} have recorded {d['w']} wins from {d['gp']} games, which",
        f"### User: Is {T} in good form?\n### Bot: {grounding} Looking at {d['form_s']}, {T} are clearly",
    ]
    return random.choice(templates), [team]

def table_prompt():
    rows = get_table_data()
    top5  = rows[:5]
    bot5  = rows[-5:]
    top_s = ", ".join(f"{r['name']} ({r['pts']}pts)" for r in top5)
    bot_s = ", ".join(f"{r['name']} ({r['pts']}pts)" for r in bot5)
    leader = rows[0]
    grounding = (
        f"DATA: Current league table. "
        f"Top 5: {top_s}. "
        f"Bottom 5: {bot_s}. "
        f"League leader: {leader['name']} with {leader['pts']} points, GD {leader['gd']:+d}."
    )
    templates = [
        f"### User: Show me the league table.\n### Bot: {grounding} {leader['name']} lead the table with {leader['pts']} points, followed by",
        f"### User: Who's top of the league?\n### Bot: {grounding} {leader['name']} top the table on {leader['pts']} points, and",
        f"### User: Give me the current standings.\n### Bot: {grounding} The top five reads {top_s}, with",
    ]
    return random.choice(templates)

def explain_prompt():
    T = pred.TEMPERATURE
    grounding = (
        f"DATA: Prediction methodology. "
        f"Model type: Poisson neural network. "
        f"Training data: 349 EPL matches. "
        f"Features: goals, xG, win rate, opponent quality, league position. "
        f"Form weighting: last 5 matches get 2.5x boost, older matches decay at 0.12 per game. "
        f"Temperature: {T} (softens probability distributions). "
        f"Output: expected goals (λ) for each team, then Poisson grid gives W/D/L probabilities. "
        f"xG measures shot quality — distance, angle, assist type."
    )
    templates = [
        f"### User: How does the prediction model work?\n### Bot: {grounding} The model generates expected goals (xG) for each side, then",
        f"### User: Explain xG and the Poisson model.\n### Bot: {grounding} xG quantifies shot quality, and the Poisson network uses it to",
        f"### User: Why do you use expected goals?\n### Bot: {grounding} xG captures shot quality far better than raw goals, which is why",
    ]
    return random.choice(templates)

def h2h_prompt(home, away):
    d = get_h2h_data(home, away)
    H, A = home.title(), away.title()
    if d is None:
        grounding = f"DATA: No historical head-to-head data found for {H} vs {A} in this dataset."
        fallback_opener = f"This fixture has no recorded meetings in the dataset, but"
    else:
        edge = H if d["h_wins"] > d["a_wins"] else (A if d["a_wins"] > d["h_wins"] else "neither side")
        grounding = (
            f"DATA: Head-to-head {H} vs {A}. "
            f"Total meetings: {d['games']}. "
            f"{H} wins: {d['h_wins']}. {A} wins: {d['a_wins']}. Draws: {d['draws']}. "
            f"Average total goals per game: {d['avg_total_goals']:.1f}. "
            f"Historical edge: {edge}."
        )
        fallback_opener = f"Across {d['games']} meetings, {edge} hold the historical edge, and"
    templates = [
        f"### User: What's the head-to-head record between {H} and {A}?\n### Bot: {grounding} {fallback_opener}",
        f"### User: H2H history for {H} vs {A}?\n### Bot: {grounding} The record shows",
    ]
    return random.choice(templates), [home, away]

def btts_prompt(home, away):
    hd = get_team_data(home)
    ad = get_team_data(away)
    H, A = home.title(), away.title()
    h_btts = sum(1 for h in list(pred.team_history.get(pred.title_to_id.get(home), []))[-10:]
                 if h["gf"] > 0 and h["ga"] > 0)
    a_btts = sum(1 for h in list(pred.team_history.get(pred.title_to_id.get(away), []))[-10:]
                 if h["gf"] > 0 and h["ga"] > 0)
    h_btts_pct = h_btts / 10 * 100
    a_btts_pct = a_btts / 10 * 100
    avg = (h_btts_pct + a_btts_pct) / 2
    likely = "likely" if avg > 60 else ("unlikely" if avg < 40 else "50/50")

    grounding = (
        f"DATA: Both-teams-to-score (BTTS) for {H} vs {A}. "
        f"{H} BTTS rate (last 10): {h_btts_pct:.0f}%. "
        f"{A} BTTS rate (last 10): {a_btts_pct:.0f}%. "
        f"Combined average: {avg:.0f}%. BTTS verdict: {likely}."
    )
    templates = [
        f"### User: Will both teams score when {H} play {A}?\n### Bot: {grounding} With a combined BTTS rate of {avg:.0f}%, this is",
        f"### User: BTTS for {H} vs {A}?\n### Bot: {grounding} Both-teams-to-score is {likely} here —",
    ]
    return random.choice(templates), [home, away]

def teams_prompt():
    team_list = ", ".join(t.title() for t in sorted(TEAMS))
    grounding = f"DATA: Available EPL clubs in this dataset: {team_list}."
    return f"### User: Which teams are available?\n### Bot: {grounding} The clubs covered are"

# ── Hardcoded fallbacks (used when LLM output is empty after sanitisation) ───

def _fallback_prediction(home, away, stats):
    lam_h, lam_a, hw, dr, aw, scoreline = stats
    H, A = home.title(), away.title()
    outcomes = ["home win", "draw", "away win"]
    best = outcomes[int(np.argmax([hw, dr, aw]))]
    return (
        f"{H} {hw*100:.1f}% / Draw {dr*100:.1f}% / {A} {aw*100:.1f}%. "
        f"Most likely scoreline: {scoreline[0]}-{scoreline[1]}. "
        f"Model calls a {best}."
    )

def _fallback_form(team, d):
    T = team.title()
    trend = "excellent" if d["win_pct"] >= 65 else ("poor" if d["win_pct"] < 35 else "mixed")
    return (
        f"{T} are in {trend} form. Last 5: {d['form_s']}. "
        f"Season record: {d['w']}W {d['d']}D {d['l']}L, {d['pts']} pts, GD {d['gd']:+d}."
    )

# ── Main chat function ───────────────────────────────────────────────────────

def ensure_scoreline(text, scoreline):
    if re.search(r"\b\d{1,2}-\d{1,2}\b", text):
        return text

    # If missing → inject it
    return text.strip() + f" Predicted scoreline: {scoreline[0]}-{scoreline[1]}."

    # (KEEP EVERYTHING ABOVE EXACTLY THE SAME UNTIL chat())

def chat(user_input):
        intent = detect_intent(user_input)
        teams  = extract_teams(user_input.lower())

        if intent == "predict":
            if len(teams) < 2:
                return "I need two team names to make a prediction."

            output = pred.predict_with_output(teams[0], teams[1])

            if output is None:
                return "Couldn't find one or both teams."

            formatted, result = output
            if formatted is None:
                return "Couldn't find one or both teams."

            # ── MUCH STRONGER EXPLANATION PROMPT ──
            explanation_prompt = f"""
    ### User: Explain this match outcome.

    DATA:
    Home xG: {result['lam_h']:.2f}
    Away xG: {result['lam_a']:.2f}
    Home win: {result['probs'][0]*100:.1f}%
    Draw: {result['probs'][1]*100:.1f}%
    Away win: {result['probs'][2]*100:.1f}%
    Prediction: {result['prediction']}

    ### Bot:
    Explain the result using football reasoning.

    Rules:
    - Do NOT repeat the numbers
    - Do NOT say "the data shows" or "the model says"
    - Do NOT mention probabilities or percentages
    - Focus on WHY one side is stronger
    - Talk about attack vs defence using the xG difference
    - Be confident and direct
    """

            explanation = llm_respond(
                explanation_prompt,
                max_new_tokens=60,     # ↓ shorter = less nonsense
                temperature=0.4,       # ↓ much more deterministic
                top_p=0.8,
                top_k=30,
                allowed_teams=teams
            )

            # fallback explanation (very important)
            if not explanation:
                if result['lam_h'] > result['lam_a']:
                    explanation = "The home side creates better chances and should have the edge in attack."
                elif result['lam_h'] < result['lam_a']:
                    explanation = "The away side creates more dangerous chances and are likely to be more clinical."
                else:
                    explanation = "Both sides create similar chances, making this a very even matchup."

            return formatted + "\n\n" + explanation

        # ─────────────────────────────────────────────

        if intent == "form":
            if not teams:
                return "Which team's form would you like?"
            d = get_team_data(teams[0])
            if d is None:
                return f"No form data available for {teams[0].title()}."
            prompt, allowed = form_prompt(teams[0])
            resp = llm_respond(prompt, allowed_teams=allowed, max_new_tokens=110)
            return resp or _fallback_form(teams[0], d)

        if intent == "table":
               
                
                rows = get_table_data()

                if not rows:
                    return "No table data available."

                # column widths (dynamic so it always aligns)
                pos_w  = len(str(len(rows)))          # width for position (1–20)
                name_w = max(len(r['name']) for r in rows)
                pts_w  = max(len(str(r['pts'])) for r in rows)
                gd_w   = max(len(f"{r['gd']:+d}") for r in rows)

                # header
                header = (
                    f"{'':>{pos_w}}  "
                    f"{'TEAM':<{name_w}} | "
                    f"{'PTS':>{pts_w}} | "
                    f"{'GD':>{gd_w}}"
                )

                lines = [header]

                # rows
                for i, r in enumerate(rows, start=1):
                    gd = f"{r['gd']:+d}"
                    line = (
                        f"{i:>{pos_w}}. "
                        f"{r['name']:<{name_w}} | "
                        f"{r['pts']:>{pts_w}} | "
                        f"{gd:>{gd_w}}"
                    )
                    lines.append(line)

                return "\n" + "\n".join(lines)

        if intent == "explain":
            prompt = explain_prompt()
            resp   = llm_respond(prompt, allowed_teams=[], max_new_tokens=130)
            return resp or (
                "The model generates expected goals for each team and converts them into match outcome probabilities."
            )

        if intent == "h2h":
            prompt, allowed = h2h_prompt(teams[0], teams[1])
            resp = llm_respond(prompt, allowed_teams=allowed, max_new_tokens=100)
            if resp:
                return resp
            d = get_h2h_data(teams[0], teams[1])
            H, A = teams[0].title(), teams[1].title()
            if d:
                return (f"{H} vs {A}: {d['games']} meetings — "
                        f"{H} {d['h_wins']}W / {d['draws']}D / {A} {d['a_wins']}W.")
            return f"No head-to-head records found for {H} vs {A}."

        if intent == "btts":
            prompt, allowed = btts_prompt(teams[0], teams[1])
            resp = llm_respond(prompt, allowed_teams=allowed, max_new_tokens=80)
            if resp:
                return resp
            return "BTTS estimate unavailable."

        if intent == "teams":
            prompt = teams_prompt()
            resp   = llm_respond(prompt, max_new_tokens=100)
            return resp or ", ".join(t.title() for t in sorted(TEAMS))

        # greeting
        greetings = {"hello", "hi", "hey", "sup", "yo"}
        if any(g in user_input.lower() for g in greetings):
            return "Hi — ask me about match predictions, form, or stats."

        return "Ask me about EPL predictions, form, or stats."

# ── Interactive loop ────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  EPL Football Chatbot")
    print("  Try: predictions · form · table · h2h · btts · explain")
    print("  Type 'quit' to exit.")
    print("=" * 60)
    print()
    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in ("quit", "exit", "bye", "q"):
                print("Bot: See you next match day.")
                break
            resp = chat(user)
            print(f"\nBot: {resp}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye.")
            break

if __name__ == "__main__":
    run()