from flask import Blueprint, render_template, request, redirect, url_for, session
import bcrypt
from database.db import get_connection

auth = Blueprint("auth", __name__)

@auth.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        password_hash = bcrypt.hashpw(
            password.encode(),
            bcrypt.gensalt()
        ).decode()

        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO users (email, password_hash, role)
            VALUES (%s, %s, 'USER')
            """, (email, password_hash)
        )

        conn.commit()
        cur.close()
        conn.close()

        return redirect(url_for("auth.login"))

    return render_template("register.html")


@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, password_hash, role
            FROM users
            WHERE email = %s
            """, (email,)
        )

        user = cur.fetchone()

        cur.close()
        conn.close()

        if user and bcrypt.checkpw(password.encode(), user[1].encode()):
            session["user_id"] = user[0]
            session["role"] = user[2]
            return redirect(url_for("main"))

        return "Invalid email or password"

    return render_template("login.html")


@auth.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("main"))



