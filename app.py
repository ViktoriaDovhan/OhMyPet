from flask import Flask, render_template
from auth import auth

app = Flask(__name__)
app.secret_key = "secret-key"

app.register_blueprint(auth)


@app.route('/')
def main():
    return render_template("main.html")


@app.route('/adopt')
def adopt():
    return render_template("adopt.html")


@app.route("/auth")
def auth_page():
    return render_template("auth.html")


if __name__ == "__main__":
    app.run(debug=True)