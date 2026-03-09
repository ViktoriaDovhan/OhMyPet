from flask import Flask, render_template, request, abort, redirect, url_for, session
from psycopg2.extras import RealDictCursor
from auth import auth
from database.db import get_connection
from functools import wraps
import os, uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "secret-key"

app.register_blueprint(auth)

ANIMAL_UPLOAD_FOLDER = os.path.join(app.static_folder, "images", "animals")
os.makedirs(ANIMAL_UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def delete_static_file(photo_url):
    if not photo_url:
        return

    relative_path = photo_url.replace("/", os.sep)
    file_path = os.path.join(app.static_folder, relative_path)

    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)


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


@app.route("/auth")
def auth_page():
    return redirect(url_for("auth.login"))


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
          AND COALESCE(a.is_active, TRUE) = TRUE
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

    cur.execute("""
        SELECT DISTINCT character
        FROM animals
        WHERE character IS NOT NULL
          AND TRIM(character) <> ''
          AND COALESCE(is_adopted, FALSE) = FALSE
          AND COALESCE(is_active, TRUE) = TRUE
        ORDER BY character
    """)
    character_rows = cur.fetchall()
    available_characters = [row["character"] for row in character_rows]

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

    return render_template("adopt.html", animals=animals, filters=filters, available_characters=available_characters)


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
            a.shelter_id,
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

    return redirect(url_for("user_profile", section="requests"))


@app.route("/shelter/requests/<int:request_id>/status", methods=["POST"])
@admin_required
def update_request_status(request_id):
    new_status = request.form.get("status")
    animal_id = request.form.get("animal_id", type=int)

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

    return redirect(url_for("shelter_profile", section="requests", animal_id=animal_id))


@app.route("/profile")
@login_required
def profile():
    role = session.get("role")

    if role == "USER":
        return redirect(url_for("user_profile"))

    if role == "ADMIN":
        return redirect(url_for("shelter_profile"))

    if role == "SUPERADMIN":
        return redirect(url_for("superadmin_profile"))

    abort(403)


@app.route("/profile/user")
@login_required
def user_profile():
    if session.get("role") != "USER":
        abort(403)

    section = request.args.get("section", "info")

    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
           SELECT id, first_name, last_name, email, phone, city, role
           FROM users
           WHERE id = %s
       """, (session["user_id"],))
    user = cur.fetchone()

    cur.execute("""
           SELECT r.id,
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

    return render_template("user.html", section=section, user=user, requests_list=requests_list)


@app.route("/profile/user/update", methods=["POST"])
@login_required
def update_user_profile():
    if session.get("role") != "USER":
        abort(403)

    first_name = request.form.get("first_name", "").strip()
    last_name = request.form.get("last_name", "").strip()
    email = request.form.get("email", "").strip()
    phone = request.form.get("phone", "").strip()
    city = request.form.get("city", "").strip()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE users
        SET first_name = %s,
            last_name = %s,
            email = %s,
            phone = %s,
            city = %s
        WHERE id = %s
    """, (first_name, last_name, email, phone, city, session["user_id"]))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("user_profile", section="info"))


@app.route("/profile/shelter")
@login_required
def shelter_profile():
    if session.get("role") != "ADMIN":
        abort(403)

    section = request.args.get("section", "requests")
    animal_id = request.args.get("animal_id", type=int)

    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    query = """
            SELECT r.id,
                   r.message,
                   r.status,
                   r.created_at,
                   u.email AS user_email,
                   a.id AS animal_id,
                   a.name AS animal_name
            FROM adoption_requests r
            JOIN users u ON u.id = r.user_id
            JOIN animals a ON a.id = r.animal_id
            WHERE a.shelter_id = %s
        """
    params = [session["shelter_id"]]

    if animal_id:
        query += " AND a.id = %s"
        params.append(animal_id)

    query += " ORDER BY r.created_at DESC"

    cur.execute(query,params)
    requests_list = cur.fetchall()

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
            COALESCE(a.is_active, TRUE) AS is_active,
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
        WHERE a.shelter_id = %s
        ORDER BY a.id DESC
    """, (session["shelter_id"],))

    animals_list = cur.fetchall()

    cur.execute("""
            SELECT id, name, city, phone, email
            FROM shelters
            WHERE id = %s
        """, (session["shelter_id"],))

    shelter = cur.fetchone()

    cur.close()
    conn.close()

    return render_template("shelter.html", section=section, requests_list=requests_list, animals_list=animals_list, shelter=shelter, selected_animal_id=animal_id)


@app.route("/profile/superadmin")
@login_required
def superadmin_profile():
    if session.get("role") != "SUPERADMIN":
        abort(403)

    section = request.args.get("section", "users")

    return render_template("superadmin.html", section=section)


@app.route("/profile/shelter/animal/add", methods=["POST"])
@admin_required
def add_shelter_animal():
    shelter_id = session.get("shelter_id")
    if not shelter_id:
        abort(403)

    name = request.form.get("name", "").strip()
    animal_type = request.form.get("animal_type", "").strip()
    breed = request.form.get("breed", "").strip()
    sex = request.form.get("sex", "").strip()
    age_months = request.form.get("age_months", "").strip()
    size = request.form.get("size", "").strip()
    color = request.form.get("color", "").strip()
    health_status = request.form.get("health_status", "").strip()
    description = request.form.get("description", "").strip()
    sterilized = request.form.get("sterilized") == "on"
    urgent = request.form.get("urgent") == "on"
    vaccinated = request.form.get("vaccinated") == "on"
    is_active = request.form.get("is_active") == "on"
    character_select = request.form.get("character_select", "").strip()
    character_custom = request.form.get("character_custom", "").strip()

    if character_select == "__other__":
        character = character_custom
    else:
        character = character_select

    if not name or not animal_type:
        return redirect(url_for("shelter_profile", section="add_animal"))

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO animals (
            shelter_id,
            name,
            animal_type,
            breed,
            sex,
            age_months,
            size,
            character,
            color,
            sterilized,
            urgent,
            vaccinated,
            health_status,
            description,
            is_active
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id 
    """, (
        shelter_id,
        name,
        animal_type,
        breed,
        sex,
        age_months,
        size,
        character,
        color,
        sterilized,
        urgent,
        vaccinated,
        health_status,
        description,
        is_active
    ))

    animal_id = cur.fetchone()[0]

    photos = request.files.getlist("photos")

    is_first_photo = True

    for photo in photos:
        if photo and photo.filename and allowed_file(photo.filename):
            ext = os.path.splitext(secure_filename(photo.filename))[1].lower()
            filename = f"{uuid.uuid4().hex}{ext}"
            save_path = os.path.join(ANIMAL_UPLOAD_FOLDER, filename)
            photo.save(save_path)

            photo_url = f"images/animals/{filename}"

            cur.execute("""
                INSERT INTO animal_photos (animal_id, photo_url, is_main)
                VALUES (%s, %s, %s)
            """, (animal_id, photo_url, is_first_photo))

            is_first_photo = False

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("shelter_profile", section="animals"))


@app.route("/profile/shelter/animal/<int:animal_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_shelter_animal(animal_id):
    shelter_id = session.get("shelter_id")
    if not shelter_id:
        abort(403)

    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT *
        FROM animals
        WHERE id = %s AND shelter_id = %s
    """, (animal_id, shelter_id))
    animal = cur.fetchone()

    if not animal:
        cur.close()
        conn.close()
        abort(404)

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        animal_type = request.form.get("animal_type", "").strip()
        breed = request.form.get("breed", "").strip()
        sex = request.form.get("sex", "").strip()
        age_months = request.form.get("age_months", type=int)
        size = request.form.get("size", "").strip()
        color = request.form.get("color", "").strip()
        health_status = request.form.get("health_status", "").strip()
        description = request.form.get("description", "").strip()
        sterilized = request.form.get("sterilized") == "on"
        urgent = request.form.get("urgent") == "on"
        vaccinated = request.form.get("vaccinated") == "on"
        is_active = request.form.get("is_active") == "on"
        character_select = request.form.get("character_select", "").strip()
        character_custom = request.form.get("character_custom", "").strip()

        if character_select == "__other__":
            character = character_custom.strip()
        else:
            character = character_select.strip()

        if not name or not animal_type:
            cur.close()
            conn.close()
            return redirect(url_for("edit_shelter_animal", animal_id=animal_id))

        cur2 = conn.cursor()

        cur2.execute("""
            UPDATE animals
            SET name = %s,
                animal_type = %s,
                breed = %s,
                sex = %s,
                age_months = %s,
                size = %s,
                character = %s,
                color = %s,
                sterilized = %s,
                urgent = %s,
                vaccinated = %s,
                health_status = %s,
                description = %s,
                is_active = %s
            WHERE id = %s AND shelter_id = %s
        """, (
            name,
            animal_type,
            breed or None,
            sex or None,
            age_months,
            size or None,
            character or None,
            color or None,
            sterilized,
            urgent,
            vaccinated,
            health_status or None,
            description or None,
            is_active,
            animal_id,
            shelter_id
        ))

        photos = request.files.getlist("photos")

        cur2.execute("""
            SELECT COUNT(*)
            FROM animal_photos
            WHERE animal_id = %s
        """, (animal_id,))
        existing_photos_count = cur2.fetchone()[0]

        for index, photo in enumerate(photos):
            if photo and photo.filename and allowed_file(photo.filename):
                ext = os.path.splitext(secure_filename(photo.filename))[1].lower()
                filename = f"{uuid.uuid4().hex}{ext}"
                save_path = os.path.join(ANIMAL_UPLOAD_FOLDER, filename)
                photo.save(save_path)

                photo_url = f"images/animals/{filename}"

                is_main = existing_photos_count == 0 and index == 0

                cur2.execute("""
                    INSERT INTO animal_photos (animal_id, photo_url, is_main)
                    VALUES (%s, %s, %s)
                """, (animal_id, photo_url, is_main))


        conn.commit()
        cur2.close()
        cur.close()
        conn.close()

        return redirect(url_for("shelter_profile", section="animals"))

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
            COALESCE(a.is_active, TRUE) AS is_active,
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
        WHERE a.shelter_id = %s
        ORDER BY a.id DESC
    """, (shelter_id,))
    animals_list = cur.fetchall()

    cur.execute("""
        SELECT
            r.id,
            r.message,
            r.status,
            r.created_at,
            u.email AS user_email,
            a.id AS animal_id,
            a.name AS animal_name
        FROM adoption_requests r
        JOIN users u ON u.id = r.user_id
        JOIN animals a ON a.id = r.animal_id
        WHERE a.shelter_id = %s
        ORDER BY r.created_at DESC
    """, (shelter_id,))
    requests_list = cur.fetchall()

    cur.execute("""
        SELECT id, name, city, phone, email
        FROM shelters
        WHERE id = %s
    """, (shelter_id,))
    shelter = cur.fetchone()

    cur.execute("""
        SELECT id, photo_url, is_main
        FROM animal_photos
        WHERE animal_id = %s
        ORDER BY is_main DESC, id ASC
    """, (animal_id,))
    animal_photos = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("shelter.html", section="edit_animal", animal_to_edit=animal, animal_photos=animal_photos, animals_list=animals_list, requests_list=requests_list, shelter=shelter, selected_animal_id=None)


@app.route("/profile/shelter/animal/<int:animal_id>/photo/<int:photo_id>/delete", methods=["POST"])
@admin_required
def delete_animal_photo(animal_id, photo_id):
    shelter_id = session.get("shelter_id")
    if not shelter_id:
        abort(403)

    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT ap.id, ap.photo_url, ap.is_main
        FROM animal_photos ap
        JOIN animals a ON a.id = ap.animal_id
        WHERE ap.id = %s
          AND ap.animal_id = %s
          AND a.shelter_id = %s
    """, (photo_id, animal_id, shelter_id))
    photo = cur.fetchone()

    if not photo:
        cur.close()
        conn.close()
        abort(404)

    cur2 = conn.cursor()

    cur2.execute("""
        DELETE FROM animal_photos
        WHERE id = %s
    """, (photo_id,))

    delete_static_file(photo["photo_url"])

    if photo["is_main"]:
        cur2.execute("""
            SELECT id
            FROM animal_photos
            WHERE animal_id = %s
            ORDER BY id ASC
            LIMIT 1
        """, (animal_id,))
        next_photo = cur2.fetchone()

        if next_photo:
            cur2.execute("""
                UPDATE animal_photos
                SET is_main = TRUE
                WHERE id = %s
            """, (next_photo[0],))

    conn.commit()
    cur2.close()
    cur.close()
    conn.close()

    return redirect(url_for("edit_shelter_animal", animal_id=animal_id))


@app.route("/profile/shelter/animal/<int:animal_id>/delete", methods=["POST"])
@admin_required
def delete_shelter_animal(animal_id):
    shelter_id = session.get("shelter_id")
    if not shelter_id:
        abort(403)

    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT id
        FROM animals
        WHERE id = %s AND shelter_id = %s
    """, (animal_id, shelter_id))
    animal = cur.fetchone()

    if not animal:
        cur.close()
        conn.close()
        abort(404)

    cur.execute("""
        SELECT photo_url
        FROM animal_photos
        WHERE animal_id = %s
    """, (animal_id,))
    photos = cur.fetchall()

    cur2 = conn.cursor()

    cur2.execute("""
        DELETE FROM animal_photos
        WHERE animal_id = %s
    """, (animal_id,))

    cur2.execute("""
        DELETE FROM animals
        WHERE id = %s AND shelter_id = %s
    """, (animal_id, shelter_id))

    conn.commit()

    for photo in photos:
        delete_static_file(photo["photo_url"])

    cur2.close()
    cur.close()
    conn.close()

    return redirect(url_for("shelter_profile", section="animals"))


@app.route("/profile/shelter/animal/<int:animal_id>/toggle", methods=["POST"])
@admin_required
def toggle_animal_active(animal_id):
    shelter_id = session.get("shelter_id")
    if not shelter_id:
        abort(403)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE animals
        SET is_active = NOT COALESCE(is_active, TRUE)
        WHERE id = %s AND shelter_id = %s
    """, (animal_id, shelter_id))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("shelter_profile", section="animals"))


if __name__ == "__main__":
    app.run(debug=True)