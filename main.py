import os
import sys

# Use fine-tuned LLM chatbot if model exists, otherwise prompt to run finetune.py
if os.path.exists("llm_model"):
    import llm_chat
    llm_chat.run()
else:
    print("Fine-tuned model not found. Run 'python finetune.py' first to train it.")
    print("Falling back to rule-based chatbot...\n")
    import chatbot
    chatbot.run()
