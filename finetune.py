"""
DEPRECATED: Fine-tuning is no longer used.

This file is kept for reference only. The chatbot now uses the base
TinyLlama-1.1B-Chat-v1.0 model WITHOUT any fine-tuning or LoRA adapters.

Instead of fine-tuning, real data is injected into prompts to ground
the LLM responses with actual football statistics.

See llm_chat.py for the current implementation.
"""

print("""
╔════════════════════════════════════════════════════════════════╗
║                       DEPRECATED FILE                         ║
║                                                                ║
║  finetune.py is no longer used. The chatbot has been          ║
║  simplified to use the base TinyLlama model with prompt        ║
║  grounding instead of fine-tuning.                            ║
║                                                                ║
║  To run the chatbot:                                           ║
║    python main.py                                              ║
║                                                                ║
║  Model: TinyLlama-1.1B-Chat-v1.0 (base, not fine-tuned)       ║
╚════════════════════════════════════════════════════════════════╝
""")
