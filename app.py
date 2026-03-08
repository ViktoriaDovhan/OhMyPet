from flask import Flask, render_template, request, abort, redirect, url_for, session
from psycopg2.extras import RealDictCursor
from auth import auth
from database.db import get_connection
from functools import wraps

app = Flask(__name__)
app.secret_key = "secret-key"

app.register_blueprint(auth)

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("auth.login"))
        if session.get("role") != "ADMIN":
            abort(403)
        return view(*args, **kwargs)
    return wrapped


@app.route('/')
def main():
    return render_template("main.html")


@app.route('/adopt')
def adopt():
    selected_types = request.args.getlist("type")
    selected_sexes = request.args.getlist("sex")
    selected_ages = request.args.getlist("age")
    selected_sizes = request.args.getlist("size")
    selected_characters = request.args.getlist("character")

    sterilized = request.args.get("sterilized") == "true"
    urgent = request.args.get("urgent") == "true"

    query = """
        SELECT
            a.id,
            a.name,
            a.animal_type,
            a.breed,
            a.sex,
            a.age_months,
            a.size,
            a.character,
            a.color,
            COALESCE(a.sterilized, FALSE) AS sterilized,
            COALESCE(a.urgent, FALSE) AS urgent,
            COALESCE(a.description, '') AS description,
            COALESCE(
                (
                    SELECT ap.photo_url
                    FROM animal_photos ap
                    WHERE ap.animal_id = a.id
                    ORDER BY ap.is_main DESC, ap.id ASC
                    LIMIT 1
                ),
                'images/no-image.png'
            ) AS photo_url
        FROM animals a
        WHERE COALESCE(a.is_adopted, FALSE) = FALSE
    """

    params = []

    if selected_types:
        query += " AND a.animal_type = ANY(%s)"
        params.append(selected_types)

    if selected_sexes:
        query += " AND a.sex = ANY(%s)"
        params.append(selected_sexes)

    if selected_sizes:
        query += " AND a.size = ANY(%s)"
        params.append(selected_sizes)

    if selected_characters:
        query += " AND a.character = ANY(%s)"
        params.append(selected_characters)

    if sterilized:
        query += " AND COALESCE(a.sterilized, FALSE) = TRUE"

    if urgent:
        query += " AND COALESCE(a.urgent, FALSE) = TRUE"

    if selected_ages:
        age_conditions = []

        if "baby" in selected_ages:
            age_conditions.append("a.age_months <= 12")
        if "young" in selected_ages:
            age_conditions.append("(a.age_months > 12 AND a.age_months <= 36)")
        if "adult" in selected_ages:
            age_conditions.append("(a.age_months > 36)")


        if age_conditions:
            query += " AND (" + " OR ".join(age_conditions) + ")"

    query += " ORDER BY a.id"

    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(query, params)
    animals = cur.fetchall()
    cur.close()
    conn.close()

    filters = {
        "types": selected_types,
        "sexes": selected_sexes,
        "ages": selected_ages,
        "sizes": selected_sizes,
        "characters": selected_characters,
        "sterilized": sterilized,
        "urgent": urgent
    }

    return render_template("adopt.html", animals=animals, filters=filters)


@app.route("/animal/<int:animal_id>")
def animal_details(animal_id):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT
            a.id,
            a.name,
            a.animal_type,
            a.breed,
            a.sex,
            a.age_months,
            a.size,
            a.character,
            a.color,
            a.sterilized,
            a.urgent,
            a.vaccinated,
            a.health_status,
            a.description,
            s.name AS shelter_name,
            s.city AS shelter_city,
            s.phone AS shelter_phone,
            s.email AS shelter_email
        FROM animals a
        LEFT JOIN shelters s ON s.id = a.shelter_id
        WHERE a.id = %s
    """, (animal_id,))

    animal = cur.fetchone()

    if not animal:
        cur.close()
        conn.close()
        abort(404)

    cur.execute("""
        SELECT photo_url, is_main
        FROM animal_photos
        WHERE animal_id = %s
        ORDER BY is_main DESC, id ASC
    """, (animal_id,))

    photos = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("animal.html", animal=animal, photos=photos)


@app.route("/animal/<int:animal_id>/request", methods=["POST"])
@login_required
def create_adoption_request(animal_id):
    message = request.form.get("message", "").strip()

    if not message:
        return redirect(url_for("animal_details", animal_id=animal_id))

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO adoption_requests (user_id, animal_id, message, status)
        VALUES (%s, %s, %s, 'NEW')
    """, (session["user_id"], animal_id, message))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("my_requests"))


@app.route("/my-requests")
@login_required
def my_requests():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT 
            r.id,
            r.message,
            r.status,
            r.created_at,
            a.name AS animal_name
        FROM adoption_requests r
        JOIN animals a ON a.id = r.animal_id
        WHERE r.user_id = %s
        ORDER BY r.created_at DESC
    """, (session["user_id"],))

    requests_list = cur.fetchall()
    cur.close()
    conn.close()

    return render_template("my_requests.html", requests_list=requests_list)


@app.route("/shelter/requests")
@admin_required
def shelter_requests():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT 
            r.id,
            r.message,
            r.status,
            r.created_at,
            u.email AS user_email,
            a.name AS animal_name
        FROM adoption_requests r
        JOIN users u ON u.id = r.user_id
        JOIN animals a ON a.id = r.animal_id
        WHERE a.shelter_id = %s
        ORDER BY r.created_at DESC
    """, (session["shelter_id"],))

    requests_list = cur.fetchall()
    cur.close()
    conn.close()

    return render_template("shelter_requests.html", requests_list=requests_list)


@app.route("/shelter/requests/<int:request_id>/status", methods=["POST"])
@admin_required
def update_request_status(request_id):
    new_status = request.form.get("status")

    if new_status not in ["NEW", "IN_REVIEW", "APPROVED", "REJECTED"]:
        abort(400)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE adoption_requests
        SET status = %s
        WHERE id = %s
    """, (new_status, request_id))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("shelter_requests"))

@app.route("/auth")
def auth_page():
    return redirect(url_for("auth.login"))


if __name__ == "__main__":
    app.run(debug=True)