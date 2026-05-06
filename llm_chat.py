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

MODEL_DIR = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Base model, not fine-tuned

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
    "leicester": "leicester city",
    "southampton": "southampton",
    "norwich": "norwich city",
    "watford": "watford",
    "bournemouth": "afc bournemouth",
    "cardiff": "cardiff city",
    "fulham": "fulham",
    "huddersfield": "huddersfield town",
    "sheffield united": "sheffield united",
    "liverpool": "liverpool",
    "chelsea": "chelsea",
    "arsenal": "arsenal",
    "everton": "everton",
    "burnley": "burnley",
    "west brom": "west bromwich albion",
}

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
    if any(w in t for w in TEAMS_WORDS):   return "teams"
    if any(w in t for w in TABLE_WORDS):   return "table"
    if any(w in t for w in EXPLAIN_WORDS): return "explain"
    if any(w in t for w in H2H_WORDS) and len(teams) == 2: return "h2h"
    if any(w in t for w in BTTS_WORDS) and len(teams) == 2: return "btts"
    if any(w in t for w in FORM_WORDS) and len(teams) <= 1: return "form"
    if len(teams) >= 2: return "predict"
    if any(w in t for w in ["predict", "vs", "versus", "against", "win",
                             "match", "beat", "result", "odds", "chance",
                             "who will", "who wins", "scoreline"]): return "predict"
    return "general"

# ── Data gathering ─────────────────────────────────────────────────────────

def get_prediction_data(home, away):
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
    hw, dr, aw = pred.outcome_probs(lam_h, lam_a)
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
    from collections import defaultdict
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

# ── LLM generation ─────────────────────────────────────────────────────────

_JUNK_RE = re.compile(
    r"https?://\S+|@\w+|#\w+|\(\s*@\w+\s*\)|pic[,\.]?\s*twitter\S*"
    r"|[-+]?[A-Za-z]?=\d+|\d{1,2}\s+\d{1,2}\s+\d{1,2}"
    r"|\b[A-Z][a-z]+ \d{1,2},\s*\d{4}\b|github\S*|gitlab\S*|@?\S+@\S+\.\S+",
    re.IGNORECASE,
)

def clean_output(text, allowed_teams=None):
    text = _JUNK_RE.sub("", text).strip()
    text = re.sub(r" {2,}", " ", text)
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        if re.search(r"[{}\[\]\\|<>]", s):
            continue
        if allowed_teams is not None:
            if any(t in s.lower() and t not in allowed_teams for t in TEAMS):
                continue
        kept.append(s)
    return " ".join(kept[:3]).strip()

def llm_respond(prompt, max_new_tokens=120, temperature=0.82,
                top_p=0.91, top_k=50, allowed_teams=None):
    """
    Feed a grounded prompt to the base LLM and return clean output.
    The prompt already contains all real numbers — the LLM writes
    the natural language around them.
    """
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(device)
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
    for stop in ["### User:", "### Bot:", "<|endoftext|>", "\n\n"]:
        if stop in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(stop)]

    return clean_output(new_tokens, allowed_teams=allowed_teams)

# ── Prompt builders (data → grounded prompt → LLM) ───────────────────────────

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

    # Streak flavour
    streak_note = ""
    if hd and hd["streak"] >= 3:
        streak_note = f"{H} are on a {hd['streak']}-game winning run."
    elif hd and hd["streak"] <= -2:
        streak_note = f"{H} have lost their last {abs(hd['streak'])} games."
    elif ad and ad["streak"] >= 3:
        streak_note = f"{A} arrive on a {ad['streak']}-match winning streak."

    grounding = (
        f"DATA: {H} vs {A}. "
        f"Expected goals: {H} {lam_h:.2f}, {A} {lam_a:.2f}. "
        f"Win probabilities: {H} {hw*100:.1f}%, Draw {dr*100:.1f}%, {A} {aw*100:.1f}%. "
        f"Most likely scoreline: {scoreline[0]}-{scoreline[1]} ({sl_prob*100:.1f}%). "
        f"Model leans toward a {best}. "
        f"{home_context} {away_context} {streak_note}"
    ).strip()

    templates = [
        f"### User: Who will win {H} vs {A}?\n### Bot: {grounding} Based on this,",
        f"### User: Predict {H} against {A}.\n### Bot: {grounding} In summary,",
        f"### User: {H} vs {A} — give me the full breakdown.\n### Bot: {grounding} Overall,",
        f"### User: Can {A} get a result at {H}?\n### Bot: {grounding} Looking at this,",
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
        streak_note = f"They are currently on a {d['streak']}-game winning streak."
    elif d["streak"] <= -2:
        streak_note = f"They have lost their last {abs(d['streak'])} matches."

    grounding = (
        f"DATA: {T} form. Last 5 results: {d['form_s']}. "
        f"Goals scored per game: {d['gfpg']:.1f}. Goals conceded per game: {d['gapg']:.1f}. "
        f"Season: {d['w']}W {d['d']}D {d['l']}L, {d['pts']} points, GD {d['gd']:+d}. "
        f"Win rate: {d['win_pct']:.0f}%. Form trend: {trend}. {streak_note}"
    ).strip()

    templates = [
        f"### User: How has {T} been playing?\n### Bot: {grounding} To summarise,",
        f"### User: What's {T}'s form like recently?\n### Bot: {grounding} Overall,",
        f"### User: Tell me about {T}'s season.\n### Bot: {grounding} In short,",
        f"### User: Is {T} in good form?\n### Bot: {grounding} Looking at the numbers,",
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
        f"### User: Show me the league table.\n### Bot: {grounding} Here's how it looks:",
        f"### User: Who's top of the league?\n### Bot: {grounding} In terms of the standings,",
        f"### User: Give me the current standings.\n### Bot: {grounding} Summarising the table,",
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
        f"### User: How does the prediction model work?\n### Bot: {grounding} In plain terms,",
        f"### User: Explain xG and the Poisson model.\n### Bot: {grounding} To explain this simply,",
        f"### User: Why do you use expected goals?\n### Bot: {grounding} The reason is,",
    ]
    return random.choice(templates)

def h2h_prompt(home, away):
    d = get_h2h_data(home, away)
    H, A = home.title(), away.title()
    if d is None:
        grounding = f"DATA: No historical head-to-head data found for {H} vs {A}."
    else:
        edge = H if d["h_wins"] > d["a_wins"] else (A if d["a_wins"] > d["h_wins"] else "neither side")
        grounding = (
            f"DATA: Head-to-head {H} vs {A}. "
            f"Total meetings: {d['games']}. "
            f"{H} wins: {d['h_wins']}. {A} wins: {d['a_wins']}. Draws: {d['draws']}. "
            f"Average total goals per game: {d['avg_total_goals']:.1f}. "
            f"Historical edge: {edge}."
        )
    templates = [
        f"### User: What's the head-to-head record between {H} and {A}?\n### Bot: {grounding} Looking at this,",
        f"### User: H2H history for {H} vs {A}?\n### Bot: {grounding} In summary,",
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
        f"### User: Will both teams score when {H} play {A}?\n### Bot: {grounding} Based on this,",
        f"### User: BTTS for {H} vs {A}?\n### Bot: {grounding} Looking at the numbers,",
    ]
    return random.choice(templates), [home, away]

def teams_prompt():
    team_list = ", ".join(t.title() for t in sorted(TEAMS))
    grounding = f"DATA: Available EPL clubs in this dataset: {team_list}."
    return f"### User: Which teams are available?\n### Bot: {grounding} The clubs I cover are"

def general_prompt(user_input):
    return (
        f"### User: {user_input}\n"
        f"### Bot: DATA: This query does not appear to be about football, EPL teams, matches, predictions, form, table, or methodology. "
        f"As a specialized football prediction bot, I only provide information on those topics. "
        f"Politely explain that and suggest asking about football instead."
    )

# ── Main chat function ───────────────────────────────────────────────────────

def chat(user_input):
    intent = detect_intent(user_input)
    teams  = extract_teams(user_input.lower())

    if intent == "predict":
        if len(teams) < 2:
            prompt = (
                f"### User: {user_input}\n"
                f"### Bot: DATA: User asked for a prediction but only named one or no teams. "
                f"I need two team names to run a prediction. In response,"
            )
            resp = llm_respond(prompt, max_new_tokens=50)
            return resp or "I need two team names to make a prediction — who's playing who?"

        prompt, allowed, stats = prediction_prompt(teams[0], teams[1])
        if prompt is None:
            return llm_respond(
                f"### User: {user_input}\n### Bot: DATA: One or both teams not found in dataset.",
                max_new_tokens=40
            ) or "Couldn't find one or both teams. Type 'teams' to see what's available."
        resp = llm_respond(prompt, allowed_teams=allowed, max_new_tokens=130)
        return resp or f"Model projects {teams[0].title()} {stats[2]*100:.1f}% / Draw {stats[3]*100:.1f}% / {teams[1].title()} {stats[4]*100:.1f}%."

    if intent == "form":
        if not teams:
            prompt = (
                f"### User: {user_input}\n"
                f"### Bot: DATA: User asked about form but didn't name a team. In response,"
            )
            resp = llm_respond(prompt, max_new_tokens=40)
            return resp or "Which team's form would you like?"
        prompt, allowed = form_prompt(teams[0])
        if prompt is None:
            return llm_respond(
                f"### User: {user_input}\n### Bot: DATA: No form data found for {teams[0].title()}.",
                max_new_tokens=40
            ) or f"No form data available for {teams[0].title()}."
        resp = llm_respond(prompt, allowed_teams=allowed, max_new_tokens=110)
        return resp or f"No detailed commentary available for {teams[0].title()}."

    if intent == "table":
        prompt = table_prompt()
        resp   = llm_respond(prompt, max_new_tokens=130)
        return resp or "League table data is available — try asking about specific teams."

    if intent == "explain":
        prompt = explain_prompt()
        resp   = llm_respond(prompt, allowed_teams=[], max_new_tokens=130)
        return resp or "I use a Poisson neural network trained on EPL data to predict match outcomes."

    if intent == "h2h":
        prompt, allowed = h2h_prompt(teams[0], teams[1])
        resp = llm_respond(prompt, allowed_teams=allowed, max_new_tokens=100)
        return resp or "Head-to-head data not available for that pair."

    if intent == "btts":
        prompt, allowed = btts_prompt(teams[0], teams[1])
        resp = llm_respond(prompt, allowed_teams=allowed, max_new_tokens=80)
        return resp or "BTTS data unavailable for that fixture."

    if intent == "teams":
        prompt = teams_prompt()
        resp   = llm_respond(prompt, max_new_tokens=100)
        return resp or ", ".join(t.title() for t in sorted(TEAMS))

    # General / off-topic — fixed response to avoid hallucination
    greetings = ["hello", "hi", "hey", "good morning", "good evening", "sup", "what's up", "yo"]
    if any(g in user_input.lower() for g in greetings):
        return "Hi there! I'm a football prediction bot. Ask me about EPL matches, team form, the league table, or how my model works!"
    else:
        return "I'm a football prediction bot. Ask me about EPL matches, team form, the league table, or how my model works!"

# ── Interactive loop ────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  EPL Football Chatbot  |  Base TinyLlama · Poisson grounded")
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
                print("Bot: " + (llm_respond(
                    "### User: Goodbye!\n### Bot:", max_new_tokens=20
                ) or "Cheers. See you next match day."))
                break
            resp = chat(user)
            print(f"\nBot: {resp}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye.")
            break

if __name__ == "__main__":
    run()
