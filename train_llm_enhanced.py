"""
Utility script for generating grounding context for the LLM chatbot.

This script analyzes EPL match data and generates contextual information
(predictions, form analysis, H2H records, etc.) that is used to ground
the LLM responses with real statistics.

The base TinyLlama model is NOT fine-tuned. Instead, structured prompts
containing real data are fed to the base model, which writes natural language
around the facts.

Hardware: CPU or GPU (runs fine on both)
"""

import json
import math
import random
import os
import numpy as np
import torch
from collections import defaultdict

if __name__ == "__main__":
    
    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # ── Config ────────────────────────────────────────────────────────────
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DATA_FILE  = "data.json"
    
    # ───────────────────────────────────────────────────────────────
    # 1. Load match data
    # ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  EPL Football Chatbot — Context Generation")
    print("  Base model: TinyLlama-1.1B-Chat-v1.0 (NOT fine-tuned)")
    print("=" * 60)
    print()
    print("Loading match data...")
    
    with open(DATA_FILE) as f:
        raw = json.load(f)
    
    matches = [m for m in raw["results"] if m.get("isResult")]
    matches.sort(key=lambda m: m["datetime"])
    print(f"  Loaded {len(matches)} completed matches\n")
    
    # ───────────────────────────────────────────────────────────────
    # 2. Build team statistics
    # ───────────────────────────────────────────────────────────────
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
    
    # ───────────────────────────────────────────────────────────────
    # 3. Stat helpers
    # ───────────────────────────────────────────────────────────────
    
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
    
    # ───────────────────────────────────────────────────────────────
    # 4. Load predict.py model
    # ───────────────────────────────────────────────────────────────
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
    
    # ───────────────────────────────────────────────────────────────
    # 5. Answer generators
    #    Short, punchy answers — TinyLlama works best with focused context.
    # ───────────────────────────────────────────────────────────────
    
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
    
    
    # ───────────────────────────────────────────────────────────────
    # 6. Build league table
    # ───────────────────────────────────────────────────────────────
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
    
    print("=" * 60)
    print("  Context generation complete!")
    print(f"  Teams: {len(team_ids)}")
    print(f"  Matches: {len(matches)}")
    print()
    print("  This script has generated statistics for grounding prompts.")
    print("  The base TinyLlama model is NOT fine-tuned.")
    print("  Instead, real data is injected into prompts to guide responses.")
    print()
    print("  Run: python main.py")
    print("=" * 60)
