from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from flask_cors import CORS
import os
import json

load_dotenv()

app = Flask(__name__, template_folder='templates')
CORS(app)

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.5)

with open("faq.json", "r") as f:
    faq_data = json.load(f)

system_prompt = (
    "You are a helpful AI assistant for an electronics company. Respond politely and clearly to customer questions "
    "about smartphones, laptops, order tracking, returns, payments, and warranties. If you're unsure, recommend contacting support."
)

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip().lower()

    if not user_input:
        return jsonify({"response": "Please enter a valid message."})

    if "smartphone" in user_input and "spec" in user_input:
        return jsonify({"response": faq_data["smartphones"]["specs"]})
    elif "laptop" in user_input and "spec" in user_input:
        return jsonify({"response": faq_data["laptops"]["specs"]})
    elif "track" in user_input and "order" in user_input:
        return jsonify({"response": faq_data["orders"]["tracking"]})
    elif "return" in user_input:
        return jsonify({"response": faq_data["orders"]["return_policy"]})
    elif "payment" in user_input or "pay" in user_input:
        return jsonify({"response": faq_data["orders"]["payment_methods"]})
    elif "warranty" in user_input:
        if "smartphone" in user_input:
            return jsonify({"response": faq_data["smartphones"]["warranty"]})
        elif "laptop" in user_input:
            return jsonify({"response": faq_data["laptops"]["warranty"]})
        else:
            return jsonify({"response": "All products come with at least a 1-year standard warranty."})

    try:
        response = conversation.predict(input=f"{system_prompt}\nUser: {user_input}")
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"})

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)


