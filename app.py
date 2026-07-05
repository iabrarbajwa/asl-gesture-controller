# %%
import os
import time
import json
import urllib.parse
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from cvzone.HandTrackingModule import HandDetector


# ============================================================
# APP CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_OPTIONS = {
    "Basic CNN": {
        "key": "basic_cnn",
        "path": MODEL_DIR / "basic_cnn_weights.keras",
        "preprocess": "rescale"
    },
    "MobileNetV2": {
        "key": "mobilenet",
        "path": MODEL_DIR / "transfer_mobilenet_sign_model.keras",
        "preprocess": "rescale"
    },
    "EfficientNetB0": {
        "key": "efficientnet",
        "path": MODEL_DIR / "efficientnetb0_sign_model.keras",
        "preprocess": "efficientnet"
    }
}

LABEL_FILE = BASE_DIR / "classes_labels.json"
REFERENCE_IMAGE = BASE_DIR / "signs.png"

PADDING = 29
INPUT_SIZE = 224
DRAWING_AREA = 400
ACTION_COOLDOWN = 2.5


# ============================================================
# STREAMLIT PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="ASL Gesture Controller",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM UI STYLE
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
            color: white;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            color: #f8fafc;
        }

        .hero-card {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            padding: 28px;
            border-radius: 22px;
            color: white;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
            margin-bottom: 25px;
        }

        .hero-title {
            font-size: 38px;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .hero-subtitle {
            font-size: 17px;
            color: #e0e7ff;
        }

        .info-card {
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            margin-bottom: 16px;
        }

        .metric-card {
            background: rgba(30, 41, 59, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.25);
            padding: 16px;
            border-radius: 16px;
            text-align: center;
        }

        .metric-label {
            font-size: 13px;
            color: #94a3b8;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 24px;
            color: #ffffff;
            font-weight: 800;
        }

        .status-ready {
            display: inline-block;
            background: rgba(34, 197, 94, 0.18);
            color: #86efac;
            border: 1px solid rgba(34, 197, 94, 0.45);
            padding: 6px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 13px;
        }

        .status-waiting {
            display: inline-block;
            background: rgba(251, 191, 36, 0.18);
            color: #fde68a;
            border: 1px solid rgba(251, 191, 36, 0.45);
            padding: 6px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 13px;
        }

        .text-box {
            background: rgba(248, 250, 252, 0.96);
            color: #0f172a;
            border-radius: 16px;
            padding: 18px;
            min-height: 90px;
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 2px;
            border: 2px solid #e2e8f0;
        }

        .history-box {
            background: rgba(15, 23, 42, 0.9);
            border-radius: 16px;
            padding: 16px;
            min-height: 90px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            color: #e5e7eb;
        }

        .stButton > button {
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            font-weight: 700;
            padding: 0.6rem 1rem;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8, #6d28d9);
            color: white;
        }

        section[data-testid="stSidebar"] {
            background: #020617;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD LABELS AND MODEL
# ============================================================

@st.cache_data
def load_sign_labels(label_path):
    if not os.path.isfile(label_path):
        st.error(f"Class label file was not found: {label_path}")
        st.stop()

    with open(label_path, "r") as file:
        stored_labels = json.load(file)

    if isinstance(stored_labels, dict):
        try:
            stored_labels = [
                stored_labels[str(i)]
                for i in range(len(stored_labels))
            ]
        except Exception:
            stored_labels = list(stored_labels.values())

    return stored_labels


@st.cache_resource
def load_classifier_model(model_path):
    model_path = str(model_path)

    if not os.path.isfile(model_path):
        st.error(f"Model file was not found:\n{model_path}")
        st.stop()

    return load_model(model_path)


labels = load_sign_labels(str(LABEL_FILE))


# ============================================================
# SESSION STATE
# ============================================================

default_state = {
    "generated_text": "",
    "previous_prediction": None,
    "prediction_counter": 0,
    "last_trigger_time": 0,
    "event_history": [],
    "confirmed_prediction": None,
    "last_prediction": "No hand",
    "last_confidence": 0.0,
    "current_fps": 0.0
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("🤟 Control Panel")

selected_model_name = st.sidebar.selectbox(
    "Choose model",
    list(MODEL_OPTIONS.keys()),
    index=1
)

selected_model_config = MODEL_OPTIONS[selected_model_name]
ACTIVE_MODEL = selected_model_config["key"]
MODEL_FILE = selected_model_config["path"]
PREPROCESSING_MODE = selected_model_config["preprocess"]

classifier = load_classifier_model(MODEL_FILE)

camera_enabled = st.sidebar.toggle("Start webcam", value=False)

operation_mode = st.sidebar.radio(
    "Operation mode",
    ["Text Generation", "Web Actions"],
    horizontal=False
)

min_confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.50,
    max_value=0.99,
    value=0.80,
    step=0.01
)

stable_frame_limit = st.sidebar.slider(
    "Stable frames required",
    min_value=5,
    max_value=40,
    value=15,
    step=1
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Active Configuration")
st.sidebar.write(f"**Model:** {selected_model_name}")
st.sidebar.write(f"**Preprocessing:** `{PREPROCESSING_MODE}`")
st.sidebar.write(f"**Input size:** `{INPUT_SIZE}x{INPUT_SIZE}`")
st.sidebar.write(f"**Cooldown:** `{ACTION_COOLDOWN}s`")

st.sidebar.markdown("---")

clear_col_1, clear_col_2 = st.sidebar.columns(2)

with clear_col_1:
    if st.button("Clear Text"):
        st.session_state.generated_text = ""
        st.session_state.confirmed_prediction = None

with clear_col_2:
    if st.button("Clear History"):
        st.session_state.event_history = []


# ============================================================
# GESTURE ACTION MAP
# ============================================================

gesture_actions = {
    "A": {"title": "Open ChatGPT", "link": "https://chat.openai.com/"},
    "B": {"title": "Open Bing", "link": "https://www.bing.com/"},
    "C": {"title": "Open Google Calendar", "link": "https://calendar.google.com/"},
    "D": {"title": "Open Google Drive", "link": "https://drive.google.com/"},
    "E": {"title": "Open Gmail", "link": "https://mail.google.com/"},
    "F": {"title": "Open Facebook", "link": "https://www.facebook.com/"},
    "G": {"title": "Open Google", "link": "https://www.google.com/"},
    "H": {"title": "Open Google Docs", "link": "https://docs.google.com/"},
    "I": {"title": "Open Instagram", "link": "https://www.instagram.com/"},
    "K": {"title": "Open Google Keep", "link": "https://keep.google.com/"},
    "L": {"title": "Open Google Lens", "link": "https://lens.google.com/"},
    "M": {"title": "Open Google Maps", "link": "https://maps.google.com/"},
    "N": {"title": "Open Google News", "link": "https://news.google.com/"},
    "O": {"title": "Open Outlook", "link": "https://outlook.live.com/"},
    "P": {"title": "Open Google Photos", "link": "https://photos.google.com/"},
    "Q": {"title": "Open Quora", "link": "https://www.quora.com/"},
    "R": {"title": "Open Reddit", "link": "https://www.reddit.com/"},
    "S": {"title": "Search Generated Text", "link": None},
    "T": {"title": "Open Google Translate", "link": "https://translate.google.com/"},
    "U": {"title": "Open Google Classroom", "link": "https://classroom.google.com/"},
    "V": {"title": "Open Google Voice", "link": "https://voice.google.com/"},
    "W": {"title": "Open Wikipedia", "link": "https://www.wikipedia.org/"},
    "X": {"title": "Open X / Twitter", "link": "https://x.com/"},
    "Y": {"title": "Open YouTube", "link": "https://www.youtube.com/"}
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def record_event(message):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.event_history.insert(0, f"**{timestamp}** — {message}")
    st.session_state.event_history = st.session_state.event_history[:10]


def launch_action(letter):
    current_time = time.time()

    if current_time - st.session_state.last_trigger_time < ACTION_COOLDOWN:
        return

    st.session_state.last_trigger_time = current_time
    letter = str(letter).upper()

    if letter not in gesture_actions:
        return

    action = gesture_actions[letter]

    if letter == "S":
        text_to_search = st.session_state.generated_text.strip()

        if text_to_search:
            query = urllib.parse.quote(text_to_search)
            webbrowser.open_new_tab(f"https://www.google.com/search?q={query}")
            record_event(f"`S` searched: `{text_to_search}`")
        else:
            webbrowser.open_new_tab("https://www.google.com/")
            record_event("`S` opened Google")

        return

    if action["link"]:
        webbrowser.open_new_tab(action["link"])
        record_event(f"`{letter}` → {action['title']}")


def prepare_image_for_prediction(canvas, preprocessing_mode):
    image = cv2.resize(canvas, (INPUT_SIZE, INPUT_SIZE))
    image = image.astype("float32")

    image = np.expand_dims(image, axis=0)

    if preprocessing_mode == "rescale":
        image = image / 255.0

    elif preprocessing_mode == "efficientnet":
        image = efficientnet_preprocess(image)

    else:
        raise ValueError(f"Unsupported preprocessing mode: {preprocessing_mode}")

    return image


def classify_canvas(canvas):
    input_image = prepare_image_for_prediction(canvas, PREPROCESSING_MODE)

    predictions = classifier.predict(input_image, verbose=0)[0]

    best_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    if best_index >= len(labels):
        return "Unknown", confidence

    predicted_class = labels[best_index]

    return predicted_class, confidence


def draw_landmark_structure(blank_canvas, landmarks, shift_x, shift_y):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
        (0, 5), (0, 17)
    ]

    for start_index, end_index in connections:
        start_point = (
            landmarks[start_index][0] + shift_x,
            landmarks[start_index][1] + shift_y
        )

        end_point = (
            landmarks[end_index][0] + shift_x,
            landmarks[end_index][1] + shift_y
        )

        cv2.line(blank_canvas, start_point, end_point, (0, 255, 0), 3)

    for landmark in landmarks:
        point = (
            landmark[0] + shift_x,
            landmark[1] + shift_y
        )

        cv2.circle(blank_canvas, point, 4, (0, 0, 255), -1)

    return blank_canvas


def extract_hand_canvas(frame, main_detector, crop_detector):
    hands, preview_frame = main_detector.findHands(
        frame,
        draw=False,
        flipType=True
    )

    if not hands:
        return preview_frame, None, False

    hand = hands[0]
    x, y, w, h = hand["bbox"]

    frame_height, frame_width = frame.shape[:2]

    left = max(0, x - PADDING)
    top = max(0, y - PADDING)
    right = min(frame_width, x + w + PADDING)
    bottom = min(frame_height, y + h + PADDING)

    hand_crop = frame[top:bottom, left:right]

    if hand_crop.size == 0:
        return preview_frame, None, False

    canvas = np.ones(
        (DRAWING_AREA, DRAWING_AREA, 3),
        dtype=np.uint8
    ) * 255

    cropped_hands, _ = crop_detector.findHands(
        hand_crop,
        draw=False,
        flipType=True
    )

    if not cropped_hands:
        return preview_frame, None, False

    landmarks = cropped_hands[0]["lmList"]

    x_shift = ((DRAWING_AREA - (right - left)) // 2) - 15
    y_shift = ((DRAWING_AREA - (bottom - top)) // 2) - 15

    canvas = draw_landmark_structure(
        canvas,
        landmarks,
        x_shift,
        y_shift
    )

    cv2.rectangle(
        preview_frame,
        (left, top),
        (right, bottom),
        (34, 197, 94),
        2
    )

    return preview_frame, canvas, True


def handle_stable_prediction(
    prediction,
    confidence,
    confidence_threshold,
    required_frames,
    mode
):
    if confidence < confidence_threshold:
        st.session_state.prediction_counter = 0
        st.session_state.previous_prediction = None
        st.session_state.confirmed_prediction = None
        return

    if prediction == st.session_state.previous_prediction:
        st.session_state.prediction_counter += 1
    else:
        st.session_state.previous_prediction = prediction
        st.session_state.prediction_counter = 1

    if st.session_state.prediction_counter >= required_frames:
        stable_letter = str(prediction).upper()

        if stable_letter != st.session_state.confirmed_prediction:
            if mode == "Web Actions":
                launch_action(stable_letter)

            elif mode == "Text Generation":
                if len(stable_letter) == 1 and stable_letter.isalpha():
                    st.session_state.generated_text += stable_letter
                    record_event(f"`{stable_letter}` added to text")

            st.session_state.confirmed_prediction = stable_letter

        st.session_state.prediction_counter = 0


# ============================================================
# MAIN PAGE HEADER
# ============================================================

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">🤟 ASL Alphabet Web Controller</div>
        <div class="hero-subtitle">
            Real-time hand gesture recognition with text generation and web action control.
        </div>
        <br>
        <span class="status-ready">Active model: {selected_model_name}</span>
        &nbsp;
        <span class="status-ready">Mode: {operation_mode}</span>
        &nbsp;
        <span class="status-waiting">Preprocessing: {PREPROCESSING_MODE}</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD LAYOUT
# ============================================================

top_col_1, top_col_2, top_col_3, top_col_4 = st.columns(4)

with top_col_1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Prediction</div>
            <div class="metric-value">{st.session_state.last_prediction}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with top_col_2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{st.session_state.last_confidence * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with top_col_3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">FPS</div>
            <div class="metric-value">{st.session_state.current_fps:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with top_col_4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Stable Frames</div>
            <div class="metric-value">{st.session_state.prediction_counter}/{stable_frame_limit}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

video_col, processed_col, reference_col = st.columns([2.1, 1.2, 1])

with video_col:
    st.markdown('<div class="info-card"><h3>📷 Webcam Feed</h3>', unsafe_allow_html=True)
    webcam_area = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with processed_col:
    st.markdown('<div class="info-card"><h3>🖐 Processed Hand</h3>', unsafe_allow_html=True)
    processed_area = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with reference_col:
    st.markdown('<div class="info-card"><h3>📘 ASL Reference</h3>', unsafe_allow_html=True)

    if os.path.isfile(str(REFERENCE_IMAGE)):
        st.image(str(REFERENCE_IMAGE), use_container_width=True)
    else:
        st.warning("signs.png was not found.")

    st.markdown("</div>", unsafe_allow_html=True)

text_col, history_col = st.columns([1.2, 1])

with text_col:
    st.markdown("### Generated Letter Sequence")
    text_output_area = st.empty()

with history_col:
    st.markdown("### Recent Activity")
    history_output_area = st.empty()


with st.expander("Gesture Action Reference", expanded=False):
    st.markdown(
        """
| Gesture | Action |
|---|---|
| A | Open ChatGPT |
| B | Open Bing |
| C | Open Google Calendar |
| D | Open Google Drive |
| E | Open Gmail |
| F | Open Facebook |
| G | Open Google |
| H | Open Google Docs |
| I | Open Instagram |
| K | Open Google Keep |
| L | Open Google Lens |
| M | Open Google Maps |
| N | Open Google News |
| O | Open Outlook |
| P | Open Google Photos |
| Q | Open Quora |
| R | Open Reddit |
| S | Search generated text on Google |
| T | Open Google Translate |
| U | Open Google Classroom |
| V | Open Google Voice |
| W | Open Wikipedia |
| X | Open X / Twitter |
| Y | Open YouTube |
"""
    )


# ============================================================
# CAMERA LOOP
# ============================================================

if camera_enabled:
    hand_detector = HandDetector(maxHands=1)
    crop_hand_detector = HandDetector(maxHands=1)

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("Webcam could not be opened. Please check camera permissions.")

    else:
        previous_time = time.time()
        frame_counter = 0
        fps_value = 0.0

        while camera_enabled:
            success, frame = video_capture.read()

            if not success:
                st.error("Unable to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            preview_frame, hand_canvas, hand_found = extract_hand_canvas(
                frame,
                hand_detector,
                crop_hand_detector
            )

            prediction = "No hand"
            confidence = 0.0

            if hand_found and hand_canvas is not None:
                prediction, confidence = classify_canvas(hand_canvas)

                handle_stable_prediction(
                    prediction,
                    confidence,
                    min_confidence,
                    stable_frame_limit,
                    operation_mode
                )

                processed_rgb = cv2.cvtColor(hand_canvas, cv2.COLOR_BGR2RGB)
                processed_area.image(processed_rgb, channels="RGB", use_container_width=True)

            else:
                cv2.putText(
                    preview_frame,
                    "No hand detected",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                processed_area.info("Waiting for hand gesture...")

            frame_counter += 1
            current_time = time.time()
            elapsed_time = current_time - previous_time

            if elapsed_time >= 1.0:
                fps_value = frame_counter / elapsed_time
                frame_counter = 0
                previous_time = current_time

            st.session_state.last_prediction = prediction
            st.session_state.last_confidence = confidence
            st.session_state.current_fps = fps_value

            cv2.putText(
                preview_frame,
                f"Model: {selected_model_name}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            cv2.putText(
                preview_frame,
                f"Prediction: {prediction}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            cv2.putText(
                preview_frame,
                f"Confidence: {confidence * 100:.2f}%",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 0, 0),
                2
            )

            cv2.putText(
                preview_frame,
                f"FPS: {fps_value:.2f}",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (34, 197, 94),
                2
            )

            preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            webcam_area.image(preview_rgb, channels="RGB", use_container_width=True)

            generated_text = (
                st.session_state.generated_text
                if st.session_state.generated_text
                else "No letters detected yet"
            )

            text_output_area.markdown(
                f"""
                <div class="text-box">
                    {generated_text}
                </div>
                """,
                unsafe_allow_html=True
            )

            activity_text = (
                "<br>".join(st.session_state.event_history)
                if st.session_state.event_history
                else "No actions yet."
            )

            history_output_area.markdown(
                f"""
                <div class="history-box">
                    {activity_text}
                </div>
                """,
                unsafe_allow_html=True
            )

            time.sleep(0.01)

        video_capture.release()

else:
    webcam_area.info("Turn on **Start webcam** from the sidebar to begin recognition.")
    processed_area.info("Processed gesture will appear here.")

    text_output_area.markdown(
        f"""
        <div class="text-box">
            {
                st.session_state.generated_text
                if st.session_state.generated_text
                else "No letters detected yet"
            }
        </div>
        """,
        unsafe_allow_html=True
    )

    history_output_area.markdown(
        f"""
        <div class="history-box">
            {
                "<br>".join(st.session_state.event_history)
                if st.session_state.event_history
                else "No actions yet."
            }
        </div>
        """,
        unsafe_allow_html=True
    )