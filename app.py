from flask import Flask, render_template, request, abort, redirect, url_for, session, jsonify
from psycopg2.extras import RealDictCursor
from auth import auth
from database.db import get_connection
from functools import wraps
import os, uuid, re
from werkzeug.utils import secure_filename
from datetime import timedelta
from cv.predictor import predict_animal_fields as predict_fields_from_photo
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
app.secret_key = "secret-key"

app.register_blueprint(auth)

ANIMAL_UPLOAD_FOLDER = os.path.join(app.static_folder, "images", "animals")
os.makedirs(ANIMAL_UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

LIBERTA_MODEL_NAME = "Goader/liberta-large"

LIBERTA_LABEL_PROTOTYPES = {
    "allergy_risk": "Нотатка про алергію, алерген, небажаний інгредієнт у кормі або алергічну реакцію тварини.",
    "stock_warning": "Нотатка про те, що корм закінчується, запас малий, потрібно поповнення або нове замовлення.",
    "urgent_attention": "Нотатка про термінову, критичну або невідкладну потребу.",
    "medical_diet": "Нотатка про лікувальний корм, спеціальну дієту, ветеринарне харчування або особливі потреби.",
    "appetite_problem": "Нотатка про поганий апетит, відмову від їжі або проблему з годуванням.",
    "friendly_behavior": "Нотатка про спокійну, лагідну, дружню або контактну поведінку тварини.",
    "aggressive_behavior": "Нотатка про агресивну поведінку, гарчання, укуси або проблемну реакцію тварини.",
    "kids_compatibility": "Нотатка про сумісність тварини з дітьми.",
    "animal_compatibility": "Нотатка про сумісність тварини з іншими тваринами.",
    "storage_or_expiry": "Нотатка про термін придатності, умови зберігання або псування корму.",
    "general_note": "Звичайна інформаційна нотатка без явно критичних ознак."
}

FLAG_LABELS = {
    "allergy_risk": "ризик алергії",
    "stock_warning": "закінчується запас",
    "urgent_attention": "потрібна термінова увага",
    "medical_diet": "лікувальна дієта",
    "appetite_problem": "проблема з апетитом",
    "friendly_behavior": "дружня поведінка",
    "aggressive_behavior": "агресивна поведінка",
    "kids_compatibility": "сумісність з дітьми",
    "animal_compatibility": "сумісність з іншими тваринами",
    "storage_or_expiry": "термін придатності / зберігання",
    "general_note": "загальна нотатка"
}

KEYWORD_HINTS = {
    "allergy_risk": [
        "алергія", "алерген", "курка", "корм", "реакція", "небажаний",
        "чутливість", "чутливий", "почервоніння", "чухання", "свербіж", "інгредієнт"
    ],
    "stock_warning": [
        "закінчується", "запас", "замовлення", "мало", "залишилось", "поповнення"
    ],
    "urgent_attention": [
        "терміново", "негайно", "критично", "увага", "невідкладно"
    ],
    "medical_diet": [
        "лікувальний", "дієта", "ветеринарний", "спеціальний корм", "раціон"
    ],
    "appetite_problem": [
        "не їсть", "апетит", "відмовляється", "годування",
        "не доїдає", "менш стабільний", "без особливого інтересу"
    ],
    "friendly_behavior": [
        "спокійний", "лагідний", "дружній", "контактний", "не проявляє агресії"
    ],
    "aggressive_behavior": [
        "агресивний", "гарчить", "кусається", "нападає"
    ],
    "kids_compatibility": [
        "діти", "дитина", "із дітьми", "з дітьми"
    ],
    "animal_compatibility": [
        "інші тварини", "коти", "собаки", "тваринами", "з іншими котами"
    ],
    "storage_or_expiry": [
        "термін придатності", "зберігати", "зберігання", "псується", "упаковка"
    ],
    "general_note": ["нотатка"]
}

LABEL_THRESHOLDS = {
    "allergy_risk": 0.58,
    "stock_warning": 0.60,
    "urgent_attention": 0.62,
    "medical_diet": 0.58,
    "appetite_problem": 0.56,
    "friendly_behavior": 0.60,
    "aggressive_behavior": 0.60,
    "kids_compatibility": 0.55,
    "animal_compatibility": 0.55,
    "storage_or_expiry": 0.58,
    "general_note": 0.72
}

ANIMAL_LABEL_KEYS = [
    "allergy_risk",
    "appetite_problem",
    "friendly_behavior",
    "aggressive_behavior",
    "kids_compatibility",
    "animal_compatibility",
    "general_note"
]

FOOD_LABEL_KEYS = [
    "allergy_risk",
    "stock_warning",
    "urgent_attention",
    "medical_diet",
    "storage_or_expiry",
    "general_note"
]

_liberta_tokenizer = None
_liberta_model = None
_label_embeddings = None
_label_keys = list(LIBERTA_LABEL_PROTOTYPES.keys())


def get_active_label_keys(entity_type):
    if entity_type == "ANIMAL":
        return ANIMAL_LABEL_KEYS
    return FOOD_LABEL_KEYS


def normalize_nlp_text(text):
    text = text.lower().strip()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def split_uk_sentences(text):
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def get_liberta():
    global _liberta_tokenizer, _liberta_model

    if _liberta_tokenizer is None:
        _liberta_tokenizer = AutoTokenizer.from_pretrained(
            LIBERTA_MODEL_NAME,
            trust_remote_code=True
        )

    if _liberta_model is None:
        _liberta_model = AutoModel.from_pretrained(LIBERTA_MODEL_NAME)
        _liberta_model.eval()

    return _liberta_tokenizer, _liberta_model


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)


def encode_texts_liberta(texts):
    tokenizer, model = get_liberta()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**encoded)
        embeddings = mean_pooling(output, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def get_label_embeddings():
    global _label_embeddings

    if _label_embeddings is None:
        prototype_texts = [LIBERTA_LABEL_PROTOTYPES[key] for key in _label_keys]
        _label_embeddings = encode_texts_liberta(prototype_texts)

    return _label_embeddings


def extract_keywords_from_text(normalized_text, selected_labels):
    found = []

    for label in selected_labels:
        hints = KEYWORD_HINTS.get(label, [])
        for hint in hints:
            if hint in normalized_text and hint not in found:
                found.append(hint)

    if not found:
        found.append("немає явних ключових слів")

    return ", ".join(found)


def score_sentence_for_label(sentence, label_key):
    normalized_sentence = normalize_nlp_text(sentence)
    hints = KEYWORD_HINTS.get(label_key, [])

    hint_score = 0
    for hint in hints:
        if hint in normalized_sentence:
            hint_score += 1

    sentence_embedding = encode_texts_liberta([normalized_sentence])
    label_embeddings = get_label_embeddings()
    label_index = _label_keys.index(label_key)

    similarity = torch.mm(sentence_embedding, label_embeddings[label_index].unsqueeze(0).T).item()

    return similarity + hint_score * 0.25


def find_best_evidence_for_label(sentences, normalized_sentences, sentence_embeddings, label_key):
    label_embeddings = get_label_embeddings()
    label_index = _label_keys.index(label_key)
    label_vector = label_embeddings[label_index]

    hints = KEYWORD_HINTS.get(label_key, [])
    sim_scores = torch.mv(sentence_embeddings, label_vector)

    best_idx = 0
    best_score = -999

    for i, sentence in enumerate(normalized_sentences):
        hint_score = 0
        for hint in hints:
            if hint in sentence:
                hint_score += 0.25

        total_score = float(sim_scores[i].item()) + hint_score

        if total_score > best_score:
            best_score = total_score
            best_idx = i

    if label_key == "allergy_risk":
        current = normalized_sentences[best_idx]

        symptom_terms = ["почервон", "чухан", "сверб", "реакц"]
        strong_cause_terms = ["курк", "інгредієнт", "небажан", "чутлив", "алергі", "білков"]

        has_symptoms = any(x in current for x in symptom_terms)
        has_strong_cause = any(x in current for x in strong_cause_terms)

        if has_symptoms and not has_strong_cause:
            if best_idx > 0:
                prev_sentence = normalized_sentences[best_idx - 1]
                if any(x in prev_sentence for x in strong_cause_terms) or "корм" in prev_sentence:
                    return sentences[best_idx - 1].strip() + " " + sentences[best_idx].strip()

            if best_idx + 1 < len(sentences):
                next_sentence = normalized_sentences[best_idx + 1]
                if any(x in next_sentence for x in strong_cause_terms):
                    return sentences[best_idx].strip() + " " + sentences[best_idx + 1].strip()

    return sentences[best_idx].strip()


def find_evidence_fragment(note_text, selected_labels):
    sentences = split_uk_sentences(note_text)

    if not sentences:
        return note_text[:180]

    normalized_sentences = [normalize_nlp_text(s) for s in sentences]
    sentence_embeddings = encode_texts_liberta(normalized_sentences)

    selected_sentences = []

    for label_key in selected_labels:
        evidence = find_best_evidence_for_label(
            sentences,
            normalized_sentences,
            sentence_embeddings,
            label_key
        )

        if evidence and evidence not in selected_sentences:
            selected_sentences.append(evidence)

    return " | ".join(selected_sentences[:3])


def analyze_note_with_liberta(note_text, entity_type):
    normalized_text = normalize_nlp_text(note_text)

    text_embedding = encode_texts_liberta([normalized_text])
    all_label_embeddings = get_label_embeddings()

    active_label_keys = get_active_label_keys(entity_type)
    active_indices = [_label_keys.index(label) for label in active_label_keys]
    active_label_embeddings = all_label_embeddings[active_indices]

    similarities = torch.mm(text_embedding, active_label_embeddings.T).squeeze(0)
    probs = torch.sigmoid(similarities * 3.5)

    base_scores = {}
    for i, label_key in enumerate(active_label_keys):
        base_scores[label_key] = float(probs[i].item())

    label_scores = dict(base_scores)
    label_scores, forced_labels = apply_domain_rules(normalized_text, label_scores, entity_type)

    ranked = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)

    selected_labels = []
    for label_key, score in ranked:
        threshold = LABEL_THRESHOLDS.get(label_key, 0.60)
        if label_key in forced_labels or score >= threshold:
            selected_labels.append(label_key)

    if not selected_labels and ranked:
        selected_labels = [ranked[0][0]]

    if len(selected_labels) > 1 and "general_note" in selected_labels:
        selected_labels = [label for label in selected_labels if label != "general_note"]

    selected_labels = selected_labels[:5]

    confidence_scores = [round(base_scores[label], 4) for label in selected_labels]

    keywords = extract_keywords_from_text(normalized_text, selected_labels)
    flags = ", ".join(FLAG_LABELS[label] for label in selected_labels)
    predicted_labels = ", ".join(selected_labels)
    confidence_text = ", ".join(str(score) for score in confidence_scores)
    evidence_fragment = find_evidence_fragment(note_text, selected_labels)

    return {
        "normalized_text": normalized_text,
        "keywords": keywords,
        "flags": flags,
        "predicted_labels": predicted_labels,
        "confidence_scores": confidence_text,
        "evidence_fragment": evidence_fragment
    }


def apply_domain_rules(normalized_text, label_scores, entity_type):
    forced_labels = set()

    if entity_type == "ANIMAL":
        no_aggression_patterns = [
            r"\bне\s+проявля\w*\s+агрес",
            r"\bне\s+агресив",
            r"\bбез\s+агрес",
            r"\bагрес\w*\s+не\s+проявля\w*",
            r"\bагрес\w*\s+нема",
            r"\bагрес\w*\s+відсут"
        ]

        has_no_aggression = any(re.search(pattern, normalized_text) for pattern in no_aggression_patterns)

        if has_no_aggression:
            if "aggressive_behavior" in label_scores:
                label_scores["aggressive_behavior"] = 0.01
            label_scores["friendly_behavior"] = label_scores.get("friendly_behavior", 0) + 0.15

        if any(word in normalized_text for word in ["спокійний", "лагідний", "дружній", "контактний", "дозволяє гладити", "сидить на руках"]):
            label_scores["friendly_behavior"] = label_scores.get("friendly_behavior", 0) + 0.12

        appetite_hits = sum(
            1 for word in [
                "апетит",
                "не доїдає",
                "залишати частину корму",
                "залишає частину корму",
                "без особливого інтересу",
                "відмовляється від їжі"
            ]
            if word in normalized_text
        )
        if appetite_hits > 0:
            label_scores["appetite_problem"] = label_scores.get("appetite_problem", 0) + 0.18
            forced_labels.add("appetite_problem")
        else:
            label_scores["appetite_problem"] = min(label_scores.get("appetite_problem", 0), 0.05)

        allergy_hits = sum(
            1 for word in [
                "алергі",
                "чутлив",
                "почервоніння",
                "чухання",
                "реакція",
                "небажан",
                "курка",
                "інгредієнт"
            ]
            if word in normalized_text
        )
        if allergy_hits >= 2:
            label_scores["allergy_risk"] = label_scores.get("allergy_risk", 0) + 0.22
            forced_labels.add("allergy_risk")
        else:
            label_scores["allergy_risk"] = min(label_scores.get("allergy_risk", 0), 0.05)

        if any(word in normalized_text for word in ["дітьми", "діти", "дитина", "із дітьми", "з дітьми"]):
            label_scores["kids_compatibility"] = label_scores.get("kids_compatibility", 0) + 0.12
            forced_labels.add("kids_compatibility")

        if any(word in normalized_text for word in [
            "іншими котами",
            "іншими тваринами",
            "з іншими котами",
            "з іншими тваринами"
        ]):
            label_scores["animal_compatibility"] = label_scores.get("animal_compatibility", 0) + 0.10
            forced_labels.add("animal_compatibility")

    elif entity_type == "FOOD":
        stock_hits = sum(
            1 for word in [
                "закінчується",
                "запас",
                "замовлення",
                "залишилося",
                "залишилось",
                "не вистачить"
            ]
            if word in normalized_text
        )
        if stock_hits > 0:
            label_scores["stock_warning"] = label_scores.get("stock_warning", 0) + 0.18
            forced_labels.add("stock_warning")
        else:
            label_scores["stock_warning"] = min(label_scores.get("stock_warning", 0), 0.05)

        medical_hits = sum(
            1 for word in ["лікувальний", "спеціальний", "ветеринарний", "дієта"]
            if word in normalized_text
        )
        if medical_hits > 0:
            label_scores["medical_diet"] = label_scores.get("medical_diet", 0) + 0.16
            forced_labels.add("medical_diet")
        else:
            label_scores["medical_diet"] = min(label_scores.get("medical_diet", 0), 0.05)

        storage_hits = sum(
            1 for word in [
                "термін придатності",
                "зберігати",
                "зберігання",
                "вологість",
                "псується",
                "упаковки"
            ]
            if word in normalized_text
        )
        if storage_hits > 0:
            label_scores["storage_or_expiry"] = label_scores.get("storage_or_expiry", 0) + 0.16
            forced_labels.add("storage_or_expiry")
        else:
            label_scores["storage_or_expiry"] = min(label_scores.get("storage_or_expiry", 0), 0.05)

        food_allergy_hits = sum(
            1 for word in [
                "алергі",
                "алерген",
                "курка",
                "інгредієнт",
                "почервоніння",
                "чухання",
                "реакція",
                "чутлив"
            ]
            if word in normalized_text
        )
        if food_allergy_hits >= 2:
            label_scores["allergy_risk"] = label_scores.get("allergy_risk", 0) + 0.18
            forced_labels.add("allergy_risk")
        else:
            label_scores["allergy_risk"] = min(label_scores.get("allergy_risk", 0), 0.05)

        urgent_hits = sum(
            1 for word in ["терміново", "негайно", "критично", "потребує додаткової уваги"]
            if word in normalized_text
        )
        if urgent_hits > 0:
            label_scores["urgent_attention"] = label_scores.get("urgent_attention", 0) + 0.20
            forced_labels.add("urgent_attention")
        else:
            label_scores["urgent_attention"] = min(label_scores.get("urgent_attention", 0), 0.05)

    return label_scores, forced_labels


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


def build_age_group(age_months):
    if age_months is None:
        return "unknown"
    if age_months <= 12:
        return "baby"
    if age_months <= 36:
        return "young"
    return "adult"


def contains_any(text, keywords):
    text = (text or "").lower()

    for keyword in keywords:
        pattern = r'(^|[^а-яіїєґa-z])' + re.escape(keyword.lower()) + r'([^а-яіїєґa-z]|$)'
        if re.search(pattern, text):
            return True

    return False

def unique_values(values):
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def derive_filters_from_preferences(preferences):
    derived_types = []
    derived_sizes = []
    derived_ages = []

    preferred_type = preferences.get("preferred_type")
    home_type = preferences.get("home_type")
    housing_size = preferences.get("housing_size")
    daily_time = preferences.get("daily_time")
    experience = preferences.get("experience")
    preferred_age = preferences.get("preferred_age")

    if preferred_type in ["Кіт", "Пес"]:
        derived_types.append(preferred_type)

    if preferred_age in ["baby", "young", "adult"]:
        derived_ages.append(preferred_age)

    if home_type == "apartment":
        if housing_size == "small":
            derived_sizes.extend(["Маленький"])
        elif housing_size == "medium":
            derived_sizes.extend(["Маленький", "Середній"])
        else:
            derived_sizes.extend(["Середній", "Маленький"])

    if home_type == "house":
        if housing_size == "small":
            derived_sizes.extend(["Середній"])
        elif housing_size == "medium":
            derived_sizes.extend(["Середній", "Великий"])
        else:
            derived_sizes.extend(["Великий", "Середній"])

    if daily_time == "low" and preferred_age not in ["baby", "young", "adult"]:
        derived_ages.append("adult")

    if experience == "none" and preferred_age not in ["baby", "young", "adult"]:
        derived_ages.append("adult")

    return {
        "types": unique_values(derived_types),
        "sizes": unique_values(derived_sizes),
        "ages": unique_values(derived_ages)
    }


def calculate_match(animal, preferences):
    score = 0
    reasons = []

    def add(points, reason=None):
        nonlocal score
        score += points
        if reason and reason not in reasons:
            reasons.append(reason)

    animal_type = (animal.get("animal_type") or "").strip()
    size = (animal.get("size") or "").strip()
    character = (animal.get("character") or "").strip()
    age_group = build_age_group(animal.get("age_months"))
    description = (animal.get("description") or "").lower()

    preferred_type = preferences.get("preferred_type")
    home_type = preferences.get("home_type")
    housing_size = preferences.get("housing_size")
    daily_time = preferences.get("daily_time")
    activity_preference = preferences.get("activity_preference")
    experience = preferences.get("experience")
    has_children = preferences.get("has_children")
    has_other_animals = preferences.get("has_other_animals")
    preferred_age = preferences.get("preferred_age")

    if preferred_type in ["Кіт", "Пес"] and animal_type == preferred_type:
        add(16, f"Підходить за видом: {animal_type.lower()}")

    if home_type == "apartment":
        if size == "Маленький":
            add(18, "Зручний для квартири")
        elif size == "Середній":
            add(12, "Може підійти для квартири")
        elif size == "Великий":
            add(3)

        if housing_size == "small" and size == "Маленький":
            add(8, "Комфортний для невеликого житла")
        elif housing_size == "medium" and size in ["Маленький", "Середній"]:
            add(6)
        elif housing_size == "large" and size in ["Середній", "Великий"]:
            add(6)

    if home_type == "house":
        if size == "Великий":
            add(18, "Добре підійде для будинку")
        elif size == "Середній":
            add(14, "Комфортний для будинку")
        elif size == "Маленький":
            add(10)

    if daily_time == "low":
        if character in ["Спокійний", "Лагідний"]:
            add(16, "Підійде, якщо вдома небагато часу")
        elif age_group == "adult":
            add(10, "Доросла тварина часто простіша в побуті")
        else:
            add(4)

    if daily_time == "medium":
        if character in ["Спокійний", "Лагідний", "Активний"]:
            add(10)
        if age_group in ["young", "adult"]:
            add(6)

    if daily_time == "high":
        if character == "Активний":
            add(16, "Підійде для активного щоденного контакту")
        elif age_group in ["baby", "young"]:
            add(10, "Потребує більше часу та уваги")
        else:
            add(7)

    if activity_preference == "calm":
        if character in ["Спокійний", "Лагідний"]:
            add(16, "Має більш спокійний характер")
        elif character == "Активний":
            add(2)

    if activity_preference == "balanced":
        if character in ["Лагідний", "Спокійний"]:
            add(10)
        elif character == "Активний":
            add(8)

    if activity_preference == "active":
        if character == "Активний":
            add(16, "Підійде для активного способу життя")
        elif age_group in ["baby", "young"]:
            add(9)

    if experience == "none":
        if character in ["Спокійний", "Лагідний"]:
            add(12, "Може бути кращим варіантом для першого досвіду")
        if age_group == "adult":
            add(8)
        elif age_group == "baby":
            add(2)

    if experience == "some":
        if age_group in ["young", "adult"]:
            add(6)

    if experience == "good":
        if character == "Активний":
            add(8)
        if size == "Великий":
            add(6)

    if preferred_age in ["baby", "young", "adult"]:
        labels = {
            "baby": "малий вік",
            "young": "молодий вік",
            "adult": "дорослий вік"
        }
        if age_group == preferred_age:
            add(12, f"Відповідає побажанню за віком: {labels[preferred_age]}")
        elif preferred_age == "adult" and age_group == "young":
            add(5)

    if has_children == "yes":
        if contains_any(description, [
            "діти",
            "дитина",
            "дітьми",
            "дитини",
            "дитячий",
            "шкільного віку",
            "для сім'ї",
            "для сімʼї",
            "для родини",
            "у родині"
        ]):
            add(12, "В описі є позитивна згадка про дітей")
        elif character == "Лагідний":
            add(6, "Лагідний характер може краще підійти для сімʼї")
    else:
        add(4)

    if has_other_animals == "yes":
        if contains_any(description, [
            "з іншими тваринами",
            "іншими тваринами",
            "з котами",
            "з собаками",
            "уживається з котами",
            "уживається з собаками",
            "добре з іншими тваринами"
        ]):
            add(12, "В описі є ознаки сумісності з іншими тваринами")
        elif character == "Лагідний":
            add(6)
    else:
        add(4)

    if animal.get("sterilized"):
        add(4, "Стерилізований")

    if animal.get("urgent"):
        add(2, "Потребує швидкого прилаштування")

    return min(100, score), reasons[:4]


@app.route("/auth")
def auth_page():
    return redirect(url_for("auth.login"))


@app.route('/')
def main():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT
            a.id,
            a.name,
            a.animal_type,
            a.sex,
            a.age_months,
            a.size,
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
        ORDER BY a.id DESC
        LIMIT 4
    """)
    animals = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("main.html", animals=animals)


@app.route('/adopt')
def adopt():
    selected_types = request.args.getlist("type")
    selected_sexes = request.args.getlist("sex")
    selected_ages = request.args.getlist("age")
    selected_sizes = request.args.getlist("size")
    selected_characters = request.args.getlist("character")
    sterilized = request.args.get("sterilized") == "true"
    urgent = request.args.get("urgent") == "true"

    intelligent_mode = request.args.get("intelligent") == "1"

    intelligent_preferences = {
        "home_type": request.args.get("home_type", ""),
        "housing_size": request.args.get("housing_size", ""),
        "daily_time": request.args.get("daily_time", ""),
        "activity_preference": request.args.get("activity_preference", ""),
        "experience": request.args.get("experience", ""),
        "has_children": request.args.get("has_children", ""),
        "has_other_animals": request.args.get("has_other_animals", ""),
        "preferred_type": request.args.get("preferred_type", ""),
        "preferred_age": request.args.get("preferred_age", "")
    }

    if intelligent_mode:
        derived_filters = derive_filters_from_preferences(intelligent_preferences)

        if not selected_types:
            selected_types = derived_filters["types"]

        if not selected_sizes:
            selected_sizes = derived_filters["sizes"]

        if not selected_ages:
            selected_ages = derived_filters["ages"]

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
        type_conditions = []
        normal_types = [t for t in selected_types if t != "Інші"]

        if normal_types:
            type_conditions.append("a.animal_type = ANY(%s)")
            params.append(normal_types)

        if "Інші" in selected_types:
            type_conditions.append("a.animal_type IS NOT NULL AND a.animal_type NOT IN ('Кіт', 'Пес')")

        if type_conditions:
            query += " AND (" + " OR ".join(type_conditions) + ")"

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

    if intelligent_mode:
        for animal in animals:
            match_score, match_reasons = calculate_match(animal, intelligent_preferences)
            animal["match_score"] = match_score
            animal["match_reasons"] = match_reasons

        animals = sorted(
            animals,
            key=lambda item: (
                item.get("match_score", 0),
                1 if item.get("urgent") else 0
            ),
            reverse=True
        )

        stronger_matches = [animal for animal in animals if animal.get("match_score", 0) >= 35]
        if stronger_matches:
            animals = stronger_matches
    else:
        for animal in animals:
            animal["match_score"] = None
            animal["match_reasons"] = []

    filters = {
        "types": selected_types,
        "sexes": selected_sexes,
        "ages": selected_ages,
        "sizes": selected_sizes,
        "characters": selected_characters,
        "sterilized": sterilized,
        "urgent": urgent
    }

    return render_template("adopt.html", animals=animals, filters=filters, available_characters=available_characters, intelligent_mode=intelligent_mode, intelligent_preferences=intelligent_preferences)


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

    analytics_module = request.args.get("module", "forecast")
    forecast_days = request.args.get("days", default=7, type=int)
    edit_food_id = request.args.get("edit_food_id", type=int)
    food_to_edit = None

    nlp_history = []

    if forecast_days not in [7, 14, 30]:
        forecast_days = 7

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

    food_history = []
    forecast_rows = []
    forecast_daily = None
    forecast_total = None
    forecast_per_animal = None
    current_animals = 0
    alpha_value = 0.45
    reserve_percent = 5
    recommended_total = None

    cur.execute("""
        SELECT COUNT(*) AS count
        FROM animals
        WHERE shelter_id = %s
          AND COALESCE(is_active, TRUE) = TRUE
          AND COALESCE(is_adopted, FALSE) = FALSE
    """, (session["shelter_id"],))
    current_animals = cur.fetchone()["count"]

    cur.execute("""
        SELECT id, "date", kg_used, animals_count
        FROM food_consumption
        WHERE shelter_id = %s
        ORDER BY "date" DESC
    """, (session["shelter_id"],))
    food_history = cur.fetchall()

    if edit_food_id:
        cur.execute("""
            SELECT id, "date", kg_used, animals_count
            FROM food_consumption
            WHERE id = %s AND shelter_id = %s
        """, (edit_food_id, session["shelter_id"]))
        food_to_edit = cur.fetchone()

    do_forecast = section == "analytics" and analytics_module == "forecast" and request.args.get("forecast") == "1"

    if do_forecast and food_history and current_animals > 0:
        history_asc = list(reversed(food_history))

        rates = []
        for row in history_asc:
            animals_cnt = int(row["animals_count"]) if row["animals_count"] else 0
            kg_used = float(row["kg_used"]) if row["kg_used"] is not None else 0

            if animals_cnt > 0:
                rate_per_animal = kg_used / animals_cnt
                rates.append(rate_per_animal)

        if rates:
            smoothed_value = rates[0]

            for value in rates[1:]:
                smoothed_value = alpha_value * value + (1 - alpha_value) * smoothed_value

            forecast_per_animal = round(smoothed_value, 3)
            forecast_daily = round(smoothed_value * current_animals, 2)
            forecast_total = round(forecast_daily * forecast_days, 2)
            recommended_total = round(forecast_total * (1 + reserve_percent / 100), 2)

            last_date = max(row["date"] for row in food_history)

            for i in range(1, forecast_days + 1):
                forecast_rows.append({
                    "date": last_date + timedelta(days=i),
                    "kg_used": forecast_daily,
                    "animals_count": current_animals
                })

    cur.execute("""
        SELECT
            id,
            entity_type,
            target_name,
            note_text,
            normalized_text,
            keywords,
            flags,
            predicted_labels,
            confidence_scores,
            evidence_fragment,
            model_name,
            created_at
        FROM nlp_analysis
        WHERE shelter_id = %s
        ORDER BY created_at DESC
    """, (session["shelter_id"],))
    nlp_history = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("shelter.html", section=section, requests_list=requests_list, animals_list=animals_list,
                           shelter=shelter, selected_animal_id=animal_id, food_history=food_history, forecast_rows=forecast_rows, forecast_daily=forecast_daily,
                           forecast_total=forecast_total, forecast_per_animal=forecast_per_animal, current_animals=current_animals, alpha_value=alpha_value,
                           reserve_percent=reserve_percent, recommended_total=recommended_total, analytics_module=analytics_module, forecast_days=forecast_days,
                           edit_food_id=edit_food_id, food_to_edit=food_to_edit, nlp_history=nlp_history)


@app.route("/profile/superadmin")
@login_required
def superadmin_profile():
    if session.get("role") != "SUPERADMIN":
        abort(403)

    section = request.args.get("section", "users")

    return render_template("superadmin.html", section=section)


@app.route("/profile/shelter/food/add", methods=["POST"])
@admin_required
def add_food_consumption():
    shelter_id = session.get("shelter_id")
    if not shelter_id:
        abort(403)

    date_value = request.form.get("date")
    kg_used = request.form.get("kg_used", type=float)
    animals_count = request.form.get("animals_count", type=int)

    if not date_value or kg_used is None or animals_count is None:
        return redirect(url_for("shelter_profile", section="analytics", module="forecast"))

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO food_consumption (shelter_id, "date", kg_used, animals_count)
        VALUES (%s, %s, %s, %s)
    """, (shelter_id, date_value, kg_used, animals_count))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("shelter_profile", section="analytics", module="forecast"))


@app.route("/profile/shelter/food/<int:food_id>/update", methods=["POST"])
@admin_required
def update_food_consumption(food_id):
    shelter_id = session.get("shelter_id")
    if not shelter_id:
        abort(403)

    date_value = request.form.get("date")
    kg_used = request.form.get("kg_used", type=float)
    animals_count = request.form.get("animals_count", type=int)

    if not date_value or kg_used is None or animals_count is None:
        return redirect(url_for("shelter_profile", section="analytics", module="forecast", edit_food_id=food_id))

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE food_consumption
        SET "date" = %s,
            kg_used = %s,
            animals_count = %s
        WHERE id = %s AND shelter_id = %s
    """, (date_value, kg_used, animals_count, food_id, shelter_id))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("shelter_profile", section="analytics", module="forecast"))


@app.route("/profile/shelter/food/<int:food_id>/delete", methods=["POST"])
@admin_required
def delete_food_consumption(food_id):
    shelter_id = session.get("shelter_id")
    if not shelter_id:
        abort(403)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        DELETE FROM food_consumption
        WHERE id = %s AND shelter_id = %s
    """, (food_id, shelter_id))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("shelter_profile", section="analytics", module="forecast"))


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


@app.route("/profile/shelter/nlp/analyze", methods=["POST"])
@admin_required
def analyze_nlp_note():
    shelter_id = session.get("shelter_id")
    if not shelter_id:
        abort(403)

    entity_type = request.form.get("entity_type", "").strip()
    target_name = request.form.get("target_name", "").strip()
    note_text = request.form.get("note_text", "").strip()

    if entity_type not in ["FOOD", "ANIMAL"] or not note_text:
        return redirect(url_for("shelter_profile", section="analytics", module="nlp"))

    analysis = analyze_note_with_liberta(note_text, entity_type)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO nlp_analysis (
            shelter_id,
            entity_type,
            target_name,
            note_text,
            normalized_text,
            keywords,
            flags,
            predicted_labels,
            confidence_scores,
            evidence_fragment,
            model_name
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        shelter_id,
        entity_type,
        target_name or None,
        note_text,
        analysis["normalized_text"],
        analysis["keywords"],
        analysis["flags"],
        analysis["predicted_labels"],
        analysis["confidence_scores"],
        analysis["evidence_fragment"],
        LIBERTA_MODEL_NAME
    ))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("shelter_profile", section="analytics", module="nlp"))

@app.route("/profile/shelter/nlp/<int:analysis_id>/delete", methods=["POST"])
@admin_required
def delete_nlp_analysis(analysis_id):
    shelter_id = session.get("shelter_id")
    if not shelter_id:
        abort(403)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        DELETE FROM nlp_analysis
        WHERE id = %s AND shelter_id = %s
    """, (analysis_id, shelter_id))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for("shelter_profile", section="analytics", module="nlp"))


@app.route("/api/cv/predict-animal-fields", methods=["POST"])
@admin_required
def cv_predict_animal_fields():
    photo = request.files.get("photo")

    if not photo or not photo.filename:
        return jsonify({
            "success": False,
            "message": "Фото не вибрано"
        }), 400

    try:
        predictions = predict_fields_from_photo(photo)

        return jsonify({
            "success": True,
            "predictions": predictions
        })
    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "message": f"Помилка CV-модуля: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)