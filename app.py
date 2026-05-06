from flask import Flask, request, jsonify, render_template
import llm_chat

app = Flask(__name__)

@app.route("/endpoint/input/", methods=['POST'])
def endpoint():
    data = request.get_json()

    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    response = llm_chat.chat(user_input)  # 🔥 your chatbot function

    return jsonify({
        "response": response
    })

@app.route("/")
def index():
    return render_template("index.html")

app.run()