from flask import Flask, render_template, request, jsonify
from chat import get_response

app = Flask(__name__)

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')  # File harus ada di folder templates/

# Route untuk halaman chatbot (nama file: aaa.html)
@app.route('/chatbot')
def chatbot():
    return render_template('aaa.html')  # Sesuaikan dengan nama file Anda

@app.post("/get")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    responses = get_response(text)
    message = {"answer": responses}
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
