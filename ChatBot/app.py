from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os
from flask_cors import CORS


# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Get OpenAI API Key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the model and memory
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

    # Basic validation for empty message
    if not user_input.strip():
        return jsonify({"response": "Please enter a valid message."})

    # Get the bot's response based on the conversation
    try:
        response = conversation.predict(input=user_input)
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"})

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)

