# EPL Football Prediction Chatbot

## Overview
A self-contained English Premier League match prediction chatbot. No external LLM APIs required. Runs entirely locally using a fine-tuned distilgpt2 model and a custom Poisson neural network trained on real EPL match data.

## Architecture

### Prediction Engine (`predict.py`)
- PyTorch neural network predicting λ_home / λ_away (expected goals)
- Trained on 349 completed EPL matches (`data.json`)
- 25 features per match: quality-adjusted goals, xG, win rate, opponent quality, league position
- Exponential decay weighting (rate=0.12), recent 5 matches get 2.5× boost
- Poisson scoreline grid (0–0 to 10–10) → P(home win), P(draw), P(away win)
- **Temperature = 1.5** (softened probabilities)
- Saved to `model.pt`

### Language Model (`llm_model/`)
- Base: `distilgpt2` (82M params, HuggingFace)
- Fine-tuned via `finetune.py` on a football corpus generated from match data
- Corpus includes: 380 team-pair prediction Q&As, 60 form Q&As, table/explain/general examples
- 40 gradient steps, batch size 8, 64-token chunks, loss 4.0 → 2.2
- Hybrid RAG approach: factual data always comes from predict.py; LLM generates conversational commentary anchored to real numbers

### Chatbot (`llm_chat.py`)
- Intent detection: predict · form · table · explain · teams · general
- Fuzzy team name matching via `difflib.SequenceMatcher`
- `extract_teams()` skips connector words (vs, against, versus) to preserve home/away order
- `filter_commentary()` strips hallucinated team names, URLs, dates from LLM output
- Factual block always computed from predict.py; LLM adds commentary via completion anchoring

## Files
| File | Purpose |
|------|---------|
| `main.py` | Entry point — loads llm_chat if model exists, falls back to chatbot |
| `llm_chat.py` | Fine-tuned LLM chatbot (primary) |
| `chatbot.py` | Rule-based chatbot (fallback) |
| `predict.py` | Model loader and Poisson prediction logic |
| `train.py` | Full training pipeline, saves model.pt |
| `finetune.py` | Generates football corpus and fine-tunes distilgpt2 → llm_model/ |
| `data.json` | 349 completed EPL match results |
| `model.pt` | Trained Poisson neural network checkpoint |
| `llm_model/` | Fine-tuned distilgpt2 weights and tokenizer |

## Running
```
python main.py        # Start chatbot
python train.py       # Retrain Poisson model
python finetune.py    # Regenerate corpus and fine-tune LLM (takes ~90s)
```

## Dependencies
- torch, numpy, scikit-learn (Poisson model)
- transformers, accelerate, datasets (LLM fine-tuning and inference)
- difflib (fuzzy team matching, stdlib)
