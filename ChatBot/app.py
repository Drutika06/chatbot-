from flask import Flask, request, jsonify, render_template
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__, template_folder='templates')

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.5)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"response": "Please enter a valid message."})
    
    response = conversation.predict(input=user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
