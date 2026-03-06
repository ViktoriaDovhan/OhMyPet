from flask import Flask, render_template, request, abort
from psycopg2.extras import RealDictCursor
from auth import auth
from database.db import get_connection

app = Flask(__name__)
app.secret_key = "secret-key"

app.register_blueprint(auth)


@app.route('/')
def main():
    return render_template("main.html")


@app.route('/adopt')
def adopt():
    animal_type = request.args.get("type", "").strip()
    sex = request.args.get("sex", "").strip()
    age = request.args.get("age", "").strip()
    size = request.args.get("size", "").strip()
    sterilized = request.args.get("sterilized", "").strip()
    urgent = request.args.get("urgent", "").strip()

    query = """
        SELECT
            a.id,
            a.name,
            a.animal_type,
            a.breed,
            a.sex,
            a.age_months,
            a.size,
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

    if animal_type:
        query += " AND a.animal_type = %s"
        params.append(animal_type)

    if sex:
        query += " AND a.sex = %s"
        params.append(sex)

    if size:
        query += " AND a.size = %s"
        params.append(size)

    if sterilized == "true":
        query += " AND COALESCE(a.sterilized, FALSE) = TRUE"

    if urgent == "true":
        query += " AND COALESCE(a.urgent, FALSE) = TRUE"

    if age == "baby":
        query += " AND a.age_months <= 12"
    elif age == "young":
        query += " AND a.age_months > 12 AND a.age_months <= 36"
    elif age == "adult":
        query += " AND a.age_months > 36"

    query += " ORDER BY a.id"

    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(query, params)
    animals = cur.fetchall()

    print("ANIMALS FROM DB:", animals)

    cur.close()
    conn.close()

    filters = {
        "type": animal_type,
        "sex": sex,
        "age": age,
        "size": size,
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
    """, (animal_id))

    animal = cur.fetchone()

    if not animal:
        cur.close()
        conn.close()
        abort(404)

    photos = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("animal.html", animal=animal, photos=photos)


@app.route("/auth")
def auth_page():
    return render_template("auth.html")


if __name__ == "__main__":
    app.run(debug=True)