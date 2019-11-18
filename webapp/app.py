from flask import Flask
import config

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello Naj and Dan! This is where Book Covers live!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
