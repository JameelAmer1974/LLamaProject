from flask import Flask, render_template, request
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # You can also use "llama3:instruct" or another model

@app.route("/", methods=["GET", "POST"])
def index():
    response_text = ""
    prompt = ""

    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        try:
            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                }
            )
            res.raise_for_status()
            response_text = res.json().get("response", "لم يتم العثور على نتيجة.")
        except Exception as e:
            response_text = f"حدث خطأ: {str(e)}"

    return render_template("index.html", result=response_text, prompt=prompt)
