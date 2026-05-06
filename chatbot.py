"""
Self-contained football prediction chatbot.
- Inference-only: no fine-tuning or training happens here
- Detects intent from natural language
- Fuzzy-matches team names
- Calls the Poisson model for predictions
- Generates varied, non-deterministic explanations
"""

import re
import math
import random
import numpy as np
import torch
from difflib import SequenceMatcher

import predict as pred

# ── Team name index ─────────────────────────────────────────────────────────────
TEAMS = sorted(pred.title_to_id.keys())   # lowercase

def fuzzy_match(query, candidates, threshold=0.6):
    """Return best-matching candidate for query string, or None."""
    query = query.lower().strip()
    best, best_score = None, 0.0
    for c in candidates:
        score = SequenceMatcher(None, query, c).ratio()
        if score > best_score:
            best, best_score = c, score
    # Also try substring containment
    for c in candidates:
        if query in c or c in query:
            if SequenceMatcher(None, query, c).ratio() > best_score * 0.8:
                best, best_score = c, max(best_score, 0.75)
    return best if best_score >= threshold else None

def extract_teams(text):
    """Pull up to two team names out of a user message."""
    text_lower = text.lower()
    found = []
    # Try every substring of 2-5 words against the team list
    words = re.sub(r"[^a-z\s]", "", text_lower).split()
    for length in range(5, 0, -1):
        for start in range(len(words) - length + 1):
            phrase = " ".join(words[start:start + length])
            match = fuzzy_match(phrase, TEAMS)
            if match and match not in found:
                found.append(match)
            if len(found) == 2:
                return found
    return found

# ── Intent detection ────────────────────────────────────────────────────────────
PREDICT_WORDS  = ["predict", "who will win", "who wins", "vs", "versus", "against",
                  "match", "game", "beat", "result", "score",
                  "chance", "odds", "likely", "favourite", "winner"]
EXPLAIN_WORDS  = ["explain", "why does", "how does", "how do", "how is it",
                  "reason", "because", "poisson", "temperature", "neural",
                  "training", "algorithm", "methodology", "calculate", "feature"]
FORM_WORDS     = ["form", "recent", "last few", "performing", "doing", "how are",
                  "how is", "run", "streak", "record", "results"]
TABLE_WORDS    = ["table", "standing", "standings", "position", "ranked",
                  "league table", "top of", "bottom of", "points tally"]
TEAMS_WORDS    = ["list teams", "list clubs", "which teams", "what teams",
                  "available teams", "all teams", "all clubs"]
GREET_WORDS    = ["hello", "hi", "hey", "howdy", "what's up", "sup"]

def detect_intent(text):
    t = text.lower()
    if any(w in t for w in GREET_WORDS) and len(t.split()) < 6:
        return "greet"
    # Table and form checked before explain to avoid false matches
    if any(w in t for w in TABLE_WORDS):
        return "table"
    if any(w in t for w in FORM_WORDS) and len(extract_teams(t)) <= 1:
        return "form"
    if any(w in t for w in EXPLAIN_WORDS):
        return "explain"
    if any(w in t for w in TEAMS_WORDS):
        return "teams"
    if any(w in t for w in PREDICT_WORDS) or len(extract_teams(t)) >= 2:
        return "predict"
    return "unknown"

# ── Response generation ─────────────────────────────────────────────────────────
OPENERS = [
    "Right,", "Sure.", "Good question.", "Let me look at that.",
    "Interesting matchup.", "Let's see.", "Here's what the model says.",
    "Based on the data,", "Crunching the numbers —", "Alright,",
]

CONFIDENCE_HIGH  = ["clearly favours", "strongly backs", "points firmly to",
                    "leans heavily toward", "makes a strong case for"]
CONFIDENCE_MED   = ["leans toward", "slightly favours", "gives the edge to",
                    "marginally prefers", "tips in favour of"]
CONFIDENCE_LOW   = ["sees it as very open between", "can't separate",
                    "calls it almost even between", "struggles to split"]

XGOAL_COMMENTS = [
    "The model expects {h} to create {lh:.1f} goals worth of chances and {a} to create {la:.1f}.",
    "Expected goals: {h} {lh:.2f} — {a} {la:.2f}.",
    "{h} are projected to average {lh:.1f} xG, while {a} come in at {la:.1f}.",
    "In terms of quality chances, the model puts {h} at {lh:.2f} and {a} at {la:.2f}.",
]

TEMP_COMMENTS = [
    "With a temperature of {t}, predictions are deliberately spread out to reflect football's inherent unpredictability.",
    "Temperature {t} keeps the model from being overconfident — football is volatile.",
    "The temperature setting ({t}) softens these probabilities, since upsets happen all the time.",
]

OUTCOME_FLAVOUR = {
    "Home Win":  ["a home victory", "the hosts to win", "a win for the home side", "{team} to take all three points at home"],
    "Draw":      ["a draw", "the points to be shared", "a stalemate", "both sides to cancel each other out"],
    "Away Win":  ["an away win", "the visitors to take it", "a result for {team}", "{team} to win on the road"],
}

def p_description(p):
    if p > 0.50: return "strong"
    if p > 0.40: return "decent"
    if p > 0.33: return "moderate"
    return "slim"

def confidence_phrase(probs):
    hw, dr, aw = probs
    spread = max(probs) - sorted(probs)[-2]
    if spread > 0.12:
        return random.choice(CONFIDENCE_HIGH)
    if spread > 0.05:
        return random.choice(CONFIDENCE_MED)
    return random.choice(CONFIDENCE_LOW)

def winner_name(probs, home, away):
    idx = int(np.argmax(probs))
    if idx == 0: return home.title()
    if idx == 2: return away.title()
    return None

def format_prediction(home, away, probs, lam_h, lam_a):
    hw, dr, aw = probs
    outcome_idx = int(np.argmax(probs))
    outcomes    = ["Home Win", "Draw", "Away Win"]
    pred_label  = outcomes[outcome_idx]
    w_name      = winner_name(probs, home, away)

    opener = random.choice(OPENERS)
    conf   = confidence_phrase(probs)

    # Winner sentence
    if pred_label == "Draw":
        win_str = f"the model {conf} {home.title()} and {away.title()}"
    else:
        win_str = f"the model {conf} {w_name}"

    # xG sentence
    xg_tmpl = random.choice(XGOAL_COMMENTS)
    xg_str  = xg_tmpl.format(h=home.title(), a=away.title(), lh=lam_h, la=lam_a)

    # Probability breakdown
    prob_lines = [
        f"  Home Win  {hw*100:4.1f}%  {'█' * int(hw*28)}",
        f"  Draw      {dr*100:4.1f}%  {'█' * int(dr*28)}",
        f"  Away Win  {aw*100:4.1f}%  {'█' * int(aw*28)}",
    ]

    temp_str = random.choice(TEMP_COMMENTS).format(t=pred.TEMPERATURE)

    # Outcome flavour
    flavour_opts = OUTCOME_FLAVOUR[pred_label]
    flavour = random.choice(flavour_opts).format(team=w_name or "")

    lines = [
        f"{opener} {win_str}.",
        "",
        xg_str,
        "",
        "\n".join(prob_lines),
        "",
        f"That points to {flavour}.",
        temp_str,
    ]
    return "\n".join(lines)

# ── Explanation responses ───────────────────────────────────────────────────────
EXPLAIN_RESPONSES = [
    """\
The model is a Poisson neural network. It takes two teams and outputs λ (expected goals) \
for each side. It was trained using Poisson negative log-likelihood loss on {n} real EPL \
matches, so it learns that goals are count data — rare, random events.

Features per team:
  • Quality-adjusted goals scored/conceded (whole season, decay-weighted)
  • xG for and against (same weighting)
  • Win and draw rates (decay-weighted)
  • Average opponent attack and defense quality faced
  • League points per game, goal difference per game, and normalised position

Recent matches get a {boost}x weight boost over older ones, and exponential decay \
(rate {decay}) fades out results further back. Opponent quality adjusts stats so that \
scoring against a tough defense counts more than scoring against a weak one.

The temperature ({temp}) flattens the final W/D/L probabilities so the model isn't \
overconfident — football is chaotic and upsets happen.\
""",
    """\
Here's how it works under the hood:

1. Each team gets an embedding (a learned identity vector) that captures overall quality.
2. Rolling season stats are computed with exponential decay — last 5 matches get a {boost}x \
boost, older games fade at rate {decay}.
3. Every stat is adjusted for opponent quality: beating a tough defense counts more.
4. League position, points per game and goal difference per game are added as direct signals \
of season-wide standing.
5. The neural net predicts λ_home and λ_away — expected goals for each side.
6. Poisson probabilities for every scoreline (0–0 to 10–10) are summed to get \
P(home win), P(draw), P(away win).
7. Temperature scaling ({temp}) softens those probabilities before showing them to you.\
""",
]

def explain_response():
    from collections import defaultdict
    import json
    with open("data.json") as f:
        data = json.load(f)
    n = len([m for m in data["results"] if m.get("isResult")])
    tmpl = random.choice(EXPLAIN_RESPONSES)
    return tmpl.format(
        n=n,
        boost=pred.RECENT_BOOST,
        decay=pred.DECAY_RATE,
        temp=pred.TEMPERATURE,
    )

# ── Form response ───────────────────────────────────────────────────────────────
def form_response(team_name):
    tid  = pred.title_to_id.get(team_name.lower())
    if not tid:
        return f"I don't have data on {team_name.title()}."
    hist = list(pred.team_history.get(tid, []))
    if not hist:
        return f"No match history found for {team_name.title()} yet."

    recent = hist[-5:]
    results = []
    for h in recent:
        r = "W" if h["won"] else ("D" if h["drew"] else "L")
        results.append(f"{r} ({h['gf']}–{h['ga']})")

    gf_avg = np.mean([h["gf"] for h in recent])
    ga_avg = np.mean([h["ga"] for h in recent])
    wr     = sum(h["won"] for h in recent) / len(recent)

    openers = [
        f"{team_name.title()}'s last {len(recent)} results:",
        f"Here's {team_name.title()}'s recent form:",
        f"Looking at {team_name.title()}'s last few games:",
    ]
    return (
        f"{random.choice(openers)}\n\n"
        f"  {' | '.join(results)}\n\n"
        f"  Avg goals scored: {gf_avg:.1f}  |  Avg goals conceded: {ga_avg:.1f}\n"
        f"  Win rate (last {len(recent)}): {wr*100:.0f}%"
    )

# ── League table response ───────────────────────────────────────────────────────
def table_response():
    table = pred.final_table
    rows  = []
    for tid, t in table.items():
        name = None
        for title, i in pred.title_to_id.items():
            if i == tid:
                name = title.title()
                break
        if name and t["gp"] > 0:
            rows.append((t["pts"], t["gd"], t["gp"], name))

    rows.sort(reverse=True)
    lines = [f"{'Pos':<4} {'Team':<28} {'GP':<4} {'Pts':<5} {'GD'}"]
    lines.append("─" * 48)
    for pos, (pts, gd, gp, name) in enumerate(rows, 1):
        gd_str = f"+{gd}" if gd >= 0 else str(gd)
        lines.append(f"{pos:<4} {name:<28} {gp:<4} {pts:<5} {gd_str}")

    openers = [
        "Here's the current league table based on the season data:",
        "Based on all results in the dataset, here's the table:",
        "The league standings from the training data:",
    ]
    return f"{random.choice(openers)}\n\n" + "\n".join(lines)

# ── Teams list ──────────────────────────────────────────────────────────────────
def teams_response():
    teams = sorted(t.title() for t in TEAMS)
    cols  = [teams[i::3] for i in range(3)]
    rows  = []
    for r in zip(*cols):
        rows.append("  ".join(f"{x:<28}" for x in r))
    openers = [
        "Here are all the teams I have data on:",
        "These are the clubs in the dataset:",
        "I can predict matches between any of these teams:",
    ]
    return f"{random.choice(openers)}\n\n" + "\n".join(rows)

# ── Greeting ────────────────────────────────────────────────────────────────────
GREETINGS = [
    "Hey! Ask me to predict any match, explain how the model works, or check a team's form.",
    "Hi there. I can predict EPL match results, show league standings, or explain my reasoning. What do you want to know?",
    "Hello! Give me two teams and I'll predict the result. You can also ask me about form, the table, or how I work.",
]

# ── Unknown ─────────────────────────────────────────────────────────────────────
UNKNOWNS = [
    "I'm not sure what you're asking. Try asking me to predict a match (e.g. 'Liverpool vs Arsenal'), explain the model, or check a team's form.",
    "Could you rephrase that? I can predict matches, show the league table, explain my methodology, or check recent form.",
    "Not quite sure — you can ask for a prediction, a form check, the standings, or an explanation of how I work.",
]

# ── Main chatbot function ───────────────────────────────────────────────────────
def chat(user_input):
    intent = detect_intent(user_input)
    teams  = extract_teams(user_input)

    if intent == "greet":
        return random.choice(GREETINGS)

    if intent == "explain":
        return explain_response()

    if intent == "table":
        return table_response()

    if intent == "form":
        if teams:
            return form_response(teams[0])
        return "Which team's form would you like to see?"

    if intent == "teams":
        return teams_response()

    if intent == "predict" or len(teams) >= 2:
        if len(teams) < 2:
            return "I need two team names to make a prediction. Who's playing?"
        home, away = teams[0], teams[1]
        hid = pred.title_to_id.get(home)
        aid = pred.title_to_id.get(away)
        if not hid or not aid:
            return "Couldn't find both teams. Type 'list teams' to see available clubs."
        feat  = pred.compute_features(hid) + pred.compute_features(aid) + [1/3, 1/3, 1/3]
        h_enc = int(pred.team_le.transform([hid])[0])
        a_enc = int(pred.team_le.transform([aid])[0])
        with torch.no_grad():
            lambdas = pred.model(
                torch.tensor([h_enc], dtype=torch.long),
                torch.tensor([a_enc], dtype=torch.long),
                torch.tensor([feat],  dtype=torch.float32),
            ).squeeze().tolist()
        lam_h, lam_a = lambdas
        probs = pred.outcome_probs(lam_h, lam_a)
        return format_prediction(home, away, probs, lam_h, lam_a)

    # If we found one team with no clear intent, try form
    if len(teams) == 1:
        return form_response(teams[0])

    return random.choice(UNKNOWNS)

# ── Run loop ────────────────────────────────────────────────────────────────────
def run():
    print("=" * 55)
    print("  Football Prediction Chatbot")
    print("  Ask me anything — predictions, form, standings,")
    print("  or how the model works. Type 'quit' to exit.")
    print("=" * 55)
    print()

    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in ("quit", "exit", "bye"):
                print("Bot: See you later.")
                break
            response = chat(user)
            print(f"\nBot: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBot: See you later.")
            break

if __name__ == "__main__":
    run()
