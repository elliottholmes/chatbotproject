"""
STRICT FOOTBALL CHATBOT
Fast + grounded + low hallucination

Model:
Qwen/Qwen2.5-1.5B-Instruct

Features:
- predictions
- form
- table
- BTTS
- H2H
- explain methodology
- list teams

Major improvements:
- deterministic outputs
- strict grounding
- no model rambling
- no hallucinated football history
- no invented statistics
- much faster than DeepSeek 7B
"""

import os
import re
import math
import torch
import numpy as np
import predict as pred
import datascript

from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=os.environ.get("HF_TOKEN"),
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    token=os.environ.get("HF_TOKEN"),
)

model.eval()

print("Model loaded.\n")

# ─────────────────────────────────────────────
# TEAMS
# ─────────────────────────────────────────────

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
}

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PREFIX = """
You are a STRICT football analysis engine.

RULES:
- ONLY use supplied DATA
- NEVER invent statistics
- NEVER invent football history
- NEVER mention injuries
- NEVER mention managers or players
- NEVER mention betting
- NEVER explain the AI model unless asked
- NEVER use outside football knowledge
- NEVER hedge
- Keep answers short and factual

Prediction explanations must ONLY discuss:
- attacking threat
- defensive weakness
- chance creation
- control of the game
"""

# ─────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────

_BLOCK_PHRASES = [
    "bet365",
    "william hill",
    "betfair",
    "paddy power",
    "ladbrokes",
    "bookmaker",
    "betting",
    "odds",
    "historically",
    "traditionally",
    "manager",
    "injury",
    "according to the model",
    "machine learning",
    "neural network",
    "poisson",
]

_HEDGE_RE = re.compile(
    r"\b(might|could|possibly|perhaps|maybe|not sure|hard to say)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────
# TEAM MATCHING
# ─────────────────────────────────────────────

def fuzzy_match(phrase, threshold=0.55):

    phrase = phrase.lower().strip()

    if phrase in TEAM_ALIASES:
        return TEAM_ALIASES[phrase]

    best = None
    best_score = 0.0

    for t in TEAMS:

        score = SequenceMatcher(None, phrase, t).ratio()

        if phrase in t or t in phrase:
            score = max(score, 0.70)

        if score > best_score:
            best = t
            best_score = score

    return best if best_score >= threshold else None


_CONNECTORS = {
    "vs",
    "v",
    "versus",
    "against",
    "play",
    "playing",
    "face",
    "faces",
}


def extract_teams(text):

    text = re.sub(r"[^a-z\s]", "", text.lower())

    words = text.split()

    found = []

    for length in range(5, 0, -1):

        for start in range(len(words) - length + 1):

            phrase_words = words[start:start + length]

            if any(w in _CONNECTORS for w in phrase_words):
                continue

            phrase = " ".join(phrase_words)

            match = fuzzy_match(phrase)

            if match and match not in found:
                found.append(match)

            if len(found) == 2:
                return found

    return found

# ─────────────────────────────────────────────
# INTENTS
# ─────────────────────────────────────────────

def detect_intent(text):

    t = text.lower()

    # greetings
    if any(x in t for x in [
        "hello",
        "hi",
        "hey",
        "yo",
        "sup",
    ]):
        return "greeting"

    # non football
    if any(x in t for x in [
        "recipe",
        "cook",
        "pancake",
        "weather",
        "movie",
        "music",
        "your name",
        "my name",
    ]):
        return "general"

    if any(x in t for x in ["table", "standings"]):
        return "table"

    if any(x in t for x in [
        "form",
        "recent",
        "season",
    ]):
        return "form"

    if any(x in t for x in [
        "btts",
        "both teams score",
        "both score",
    ]):
        return "btts"

    if any(x in t for x in [
        "head to head",
        "h2h",
    ]):
        return "h2h"

    if any(x in t for x in [
        "how does",
        "xg",
        "explain model",
    ]):
        return "explain"

    if any(x in t for x in [
        "teams",
        "clubs",
    ]):
        return "teams"

    teams = extract_teams(t)

    if len(teams) >= 2:
        return "predict"

    return "general"
# ─────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────

def sentence_clean(s, allowed_teams=None):

    low = s.lower()

    if any(p in low for p in _BLOCK_PHRASES):
        return False

    if _HEDGE_RE.search(low):
        return False

    if allowed_teams:

        for t in TEAMS:

            if t in low and t not in allowed_teams:
                return False

    return len(s) >= 15


def clean_output(text, allowed_teams=None):

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)

    kept = []

    for s in sentences:

        if sentence_clean(s, allowed_teams):
            kept.append(s)

    return " ".join(kept[:3])

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

def llm_respond(prompt, max_new_tokens=80, allowed_teams=None):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PREFIX,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            # IMPORTANT
            do_sample=False,

            max_new_tokens=max_new_tokens,

            repetition_penalty=1.12,

            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return clean_output(
        generated,
        allowed_teams=allowed_teams,
    )

def general_llm_response(user_input):

    system = """
    You are a conversational football chatbot.

    You answer ALL questions normally and naturally.

    However:
    - your personality is football-obsessed
    - you naturally relate things back to football when appropriate
    - you often use football metaphors, comparisons, or analogies
    - football should feel like your favourite topic

    IMPORTANT:
    - still answer the user's actual question
    - do NOT refuse harmless non-football questions
    - do NOT force football references unnaturally
    - keep replies concise and conversational
    - avoid sounding robotic
    - avoid generic AI assistant phrasing

    Examples of desired behavior:

    User: "how do I make pancakes?"
    Good response:
    "Pancakes are usually just flour, eggs, milk, and butter. Easier to put together than a relegation defence."

    User: "what's the weather?"
    Good response:
    "I can't check live weather, but hopefully it's less unpredictable than football results."

    User: "what is the meaning of life?"
    Good response:
    "Probably scoring in stoppage time."

    User: "hello"
    Good response:
    "Hey — ready for football talk?"

    Bad behavior:
    - refusing the question
    """

    messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            do_sample=True,
            temperature=0.8,
            top_p=0.9,

            max_new_tokens=200,

            repetition_penalty=1.1,

            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    generated = re.sub(
        r"<think>.*?</think>",
        "",
        generated,
        flags=re.DOTALL,
    )

    return generated.strip()

# ─────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────

def most_likely_scoreline(lam_h, lam_a, max_g=6):

    best_p = -1
    best = (0, 0)

    for i in range(max_g + 1):

        for j in range(max_g + 1):

            p = (
                (math.exp(-lam_h) * lam_h**i / math.factorial(i))
                *
                (math.exp(-lam_a) * lam_a**j / math.factorial(j))
            )

            if p > best_p:
                best_p = p
                best = (i, j)

    return best


def get_team_data(team):

    tid = pred.title_to_id.get(team)

    if not tid:
        return None

    hist = list(pred.team_history.get(tid, []))

    recent = hist[-5:]

    if not recent:
        return None

    table = pred.final_table.get(
        tid,
        {"pts": 0, "gd": 0, "w": 0, "d": 0, "l": 0},
    )

    gfpg = sum(h["gf"] for h in recent) / len(recent)
    gapg = sum(h["ga"] for h in recent) / len(recent)

    form = "".join(
        "W" if h["won"] else "D" if h["drew"] else "L"
        for h in recent
    )

    return {
        "form": form,
        "gfpg": gfpg,
        "gapg": gapg,
        "pts": table["pts"],
        "gd": table["gd"],
        "w": table.get("w", 0),
        "d": table.get("d", 0),
        "l": table.get("l", 0),
    }

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────
def prediction_prompt(home, away):

    result = pred.predict(home, away)

    if result is None:
        return None

    lam_h = result["lam_h"]
    lam_a = result["lam_a"]

    hw, dr, aw = result["probs"]

    scoreline = most_likely_scoreline(lam_h, lam_a)

    H = home.title()
    A = away.title()

    # determine actual winner correctly
    if hw >= dr and hw >= aw:
        result_type = "home win"
        winner_name = H

    elif aw >= hw and aw >= dr:
        result_type = "away win"
        winner_name = A

    else:
        result_type = "draw"
        winner_name = "Draw"

    # explanation direction
    if result_type == "home win":

        explanation_direction = (
            f"{H} generate the stronger attacking threat "
            f"and should create the better chances."
        )

    elif result_type == "away win":

        explanation_direction = (
            f"{A} carry more attacking danger "
            f"and should control the more dangerous moments."
        )

    else:

        explanation_direction = (
            "Both teams appear evenly matched "
            "with similar attacking output."
        )

    prompt = f"""
MATCH:
{H} vs {A}

OFFICIAL RESULT:
- Winner: {winner_name}
- Result type: {result_type}
- Scoreline: {scoreline[0]}-{scoreline[1]}

MATCH DATA:
- {H} xG: {lam_h:.2f}
- {A} xG: {lam_a:.2f}

EXPLANATION DIRECTION:
{explanation_direction}

TASK:
Explain why this result is likely.

IMPORTANT RULES:
- NEVER change the winner
- NEVER contradict the official result
- NEVER say the opposite team should win
- NEVER invent league positions
- NEVER invent form
- NEVER invent injuries
- NEVER invent statistics
- ONLY use supplied data
- ONLY discuss attack, defence, control, and chance quality
- DO NOT explain the AI model
- Keep the answer concise
"""

    return {
        "prompt": prompt,
        "winner": winner_name,
        "scoreline": scoreline,
        "allowed_teams": [home, away],
    }

def form_prompt(team):

    d = get_team_data(team)

    if d is None:
        return None

    T = team.title()

    return f"""
TEAM:
{T}

DATA:
- Recent form: {d['form']}
- Goals scored per game: {d['gfpg']:.1f}
- Goals conceded per game: {d['gapg']:.1f}
- Record: {d['w']}W {d['d']}D {d['l']}L
- Points: {d['pts']}
- Goal difference: {d['gd']}

TASK:
Explain the team's current form briefly.
"""


def btts_prompt(home, away):

    hid = pred.title_to_id.get(home)
    aid = pred.title_to_id.get(away)

    hh = list(pred.team_history.get(hid, []))[-10:]
    ah = list(pred.team_history.get(aid, []))[-10:]

    h_btts = sum(1 for h in hh if h["gf"] > 0 and h["ga"] > 0)
    a_btts = sum(1 for h in ah if h["gf"] > 0 and h["ga"] > 0)

    avg = ((h_btts + a_btts) / 20) * 100

    verdict = (
        "likely"
        if avg >= 60
        else "unlikely"
        if avg <= 40
        else "balanced"
    )

    return f"""
MATCH:
{home.title()} vs {away.title()}

DATA:
- Home BTTS rate: {h_btts}/10
- Away BTTS rate: {a_btts}/10
- Verdict: {verdict}

TASK:
Explain whether both teams are likely to score.
"""

# ─────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────

def chat(user_input):

    intent = detect_intent(user_input)

    teams = extract_teams(user_input)
    
    if intent == "general" or intent == 'greeting':
        return general_llm_response(user_input)
    
    # ───────── PREDICT ─────────

    # ───────── PREDICT ─────────

    if intent == "predict":

        if len(teams) < 2:
            return "I need two team names."

        home, away = teams[0], teams[1]

        data = prediction_prompt(home, away)

        if data is None:
            return "Prediction unavailable."

        response = llm_respond(
            data["prompt"],
            max_new_tokens=90,
            allowed_teams=data["allowed_teams"],
        )

        if not response:

            response = (
                f"{data['winner']} should create the better attacking chances "
                f"and control more of the dangerous play."
            )

        return (
            f"Winner: {data['winner']}\n"
            f"Scoreline: {data['scoreline'][0]}-{data['scoreline'][1]}\n\n"
            f"{response}"
        )
    # ───────── FORM ─────────

    if intent == "form":

        if not teams:
            return "Which team?"

        prompt = form_prompt(teams[0])

        if prompt is None:
            return "No form data available."

        return llm_respond(
            prompt,
            max_new_tokens=70,
            allowed_teams=[teams[0]],
        )

    # ───────── BTTS ─────────

    if intent == "btts":

        if len(teams) < 2:
            return "I need two teams."

        prompt = btts_prompt(teams[0], teams[1])

        return llm_respond(
            prompt,
            max_new_tokens=60,
            allowed_teams=teams,
        )

    # ───────── TABLE ─────────

    if intent == "table":

        rows = []

        for tid, t in pred.final_table.items():

            name = next(
                (
                    title.title()
                    for title, i in pred.title_to_id.items()
                    if i == tid
                ),
                None,
            )

            if name:
                rows.append({
                    "name": name,
                    "pts": t["pts"],
                    "gd": t["gd"],
                })

        rows.sort(
            key=lambda r: (r["pts"], r["gd"]),
            reverse=True,
        )

        lines = []

        for i, r in enumerate(rows, start=1):

            lines.append(
                f"{i}. {r['name']} - {r['pts']} pts (GD {r['gd']:+d})"
            )

        return "\n".join(lines)

    # ───────── EXPLAIN ─────────

    if intent == "explain":

        return (
            "The model estimates expected goals for each team "
            "using form, attack, defence, and opponent strength. "
            "Those expected goals are converted into match outcome probabilities."
        )

    # ───────── TEAMS ─────────

    if intent == "teams":

        return ", ".join(t.title() for t in TEAMS)

    # ───────── H2H ─────────

    if intent == "h2h":

        if len(teams) < 2:
            return "I need two teams."

    return datascript.headtohead(teams)

    return "Ask about predictions, form, BTTS, H2H, or standings."

# ─────────────────────────────────────────────
# LOOP
# ─────────────────────────────────────────────

def run():

    print("=" * 60)
    print("STRICT FOOTBALL CHATBOT")
    print("=" * 60)

    while True:

        try:

            user = input("\nYou: ").strip()

            if not user:
                continue

            if user.lower() in ["quit", "exit", "q"]:

                print("\nBot: Goodbye.")
                break

            response = chat(user)

            print(f"\nBot:\n{response}")

        except KeyboardInterrupt:

            print("\nBot: Goodbye.")
            break

        except Exception as e:

            print("\nERROR:", e)

# ─────────────────────────────────────────────

if __name__ == "__main__":
    run()