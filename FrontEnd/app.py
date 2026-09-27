import os
import sys
import cv2
import streamlit as st
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Backend"))

from services.face_service import FaceService
from services.session_manager import SessionManager, IDLE, UNKNOWN, EXERCISING, SUMMARY
from services.registration import POSES, POSE_PAUSE_SEC, check_duplicate, finalize_registration
from models.database import init_db, get_all_face_embeddings

st.set_page_config(page_title="RepVision — AI-Powered Exercise Tracking System", layout="wide")

REP_GOAL = 20  # target reps shown as a progress bar in the stats panel

# ======================================================================
# Visual theme -- a gym-console look (dark, flat panels, tabular digits)
# ======================================================================
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500&family=Space+Mono:wght@700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0E1013;
            --bg-panel: #171A1F;
            --bg-panel-alt: #1F242B;
            --text-primary: #ECEFF3;
            --text-muted: #7C8593;
            --accent-live: #FF5A36;
            --accent-ok: #3DDC97;
        }

        .stApp {
            background-color: var(--bg-base);
        }

        .console-title {
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            font-size: 2.4rem;
            letter-spacing: 0.02em;
            color: var(--text-primary);
            margin-bottom: 0.1rem;
            text-align: center;
        }
        .console-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
            text-align: center;
        }

        .status-pill {
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 1.05rem;
            padding: 0.85rem 1.1rem;
            border-left: 3px solid var(--accent-live);
            background: var(--bg-panel);
            color: var(--text-primary);
            border-radius: 2px;
            margin-bottom: 1rem;
        }
        .status-pill.ok {
            border-left-color: var(--accent-ok);
        }

        .console-panel {
            background: var(--bg-panel);
            border: 1px solid var(--bg-panel-alt);
            border-radius: 4px;
            padding: 1.25rem 1.4rem;
        }
        .console-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 0.55rem 0;
            border-bottom: 1px solid var(--bg-panel-alt);
        }
        .console-row:last-child {
            border-bottom: none;
        }
        .console-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        .console-value {
            font-family: 'Space Mono', monospace;
            font-weight: 700;
            font-size: 1.6rem;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
        }
        .console-value.accent {
            color: var(--accent-ok);
        }
        .console-footer {
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.9rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--bg-panel-alt);
        }
        .console-footer.tight {
            margin-top: 0.5rem;
            padding-top: 0;
            border-top: none;
        }
        .progress-track {
            width: 100%;
            height: 8px;
            background: var(--bg-panel-alt);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 1rem;
        }
        .progress-fill {
            height: 100%;
            background: var(--accent-ok);
            transition: width 0.3s ease;
        }
        .timer-row {
            font-family: 'Space Mono', monospace;
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 0.6rem;
        }

        section[data-testid="stSidebar"] {
            background-color: var(--bg-panel);
        }
        .sidebar-section-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            text-transform: uppercase;
            margin: 1.1rem 0 0.4rem 0.1rem;
        }
        .sidebar-divider {
            border-top: 1px solid var(--bg-panel-alt);
            margin: 0.9rem 0;
        }

        .stButton > button {
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            width: 100%;
            border-radius: 3px;
            border: 1px solid var(--bg-panel-alt);
            background: var(--bg-panel-alt);
            color: var(--text-primary);
            padding: 0.55rem 1rem;
            transition: border-color 0.15s ease;
        }
        .stButton > button:hover {
            border-color: var(--accent-live);
            color: var(--text-primary);
        }
        .stButton > button[kind="primary"] {
            background: var(--accent-live);
            border-color: var(--accent-live);
            color: #14161A;
            font-weight: 600;
        }
        .stButton > button[kind="primary"]:hover {
            opacity: 0.9;
        }

        /* Framed video feed */
        div[data-testid="stImage"] img {
            border-radius: 6px;
            border: 1px solid var(--bg-panel-alt);
            box-shadow: 0 0 0 1px rgba(255, 90, 54, 0.12), 0 10px 30px rgba(0, 0, 0, 0.45);
        }

        /* Home screen exercise tiles */
        .exercise-menu {
            max-width: 640px;
            margin: 0 auto;
        }
        .exercise-menu .stButton > button {
            height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            font-size: 1.3rem;
            font-weight: 600;
            border-top: 3px solid var(--accent-live);
            background: var(--bg-panel);
            transition: background-color 0.15s ease, transform 0.15s ease;
        }
        .exercise-menu .stButton > button:hover {
            background: var(--bg-panel-alt);
            transform: translateY(-2px);
        }
        .exercise-caption {
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            color: var(--text-muted);
            text-align: center;
            margin-top: 0.5rem;
        }
        .home-divider {
            width: 60px;
            height: 3px;
            background: var(--accent-live);
            margin: 0 auto 2rem auto;
            border-radius: 2px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_status(text, live=True):
    cls = "ok" if not live else ""
    return f'<div class="status-pill {cls}">{text}</div>'


def format_duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def render_stats(rows, footer=None, progress=None, duration=None):
    body = "".join(
        f'<div class="console-row"><span class="console-label">{label}</span>'
        f'<span class="console-value{" accent" if accent else ""}">{value}</span></div>'
        for label, value, accent in rows
    )

    extra = ""
    if progress is not None:
        pct = int(min(max(progress, 0.0), 1.0) * 100)
        extra += f'<div class="progress-track"><div class="progress-fill" style="width:{pct}%;"></div></div>'
        extra += f'<div class="console-footer tight">{pct}% of {REP_GOAL}-rep goal</div>'
    if duration is not None:
        extra += f'<div class="timer-row">⏱ {duration}</div>'

    footer_html = f'<div class="console-footer">{footer}</div>' if footer else ""
    return f'<div class="console-panel">{body}{extra}{footer_html}</div>'


@st.cache_resource
def load_face_service():
    init_db()
    return FaceService()


def get_manager():
    if "manager" not in st.session_state:
        face_service = load_face_service()
        st.session_state.manager = SessionManager(face_service, get_all_face_embeddings())
    return st.session_state.manager


def reset_capture_state():
    st.session_state.pending_registration = None
    st.session_state.capture_active = False
    st.session_state.capture_pose_index = 0
    st.session_state.capture_samples = []
    st.session_state.capture_pose_start = None


manager = get_manager()

if "pending_registration" not in st.session_state:
    reset_capture_state()

if "selected_exercise" not in st.session_state:
    st.session_state.selected_exercise = None  # None = home screen; "pushup" / "squat" once chosen

EXERCISE_LABELS = {"pushup": "Push-Ups", "squat": "Squats", "plank": "Plank"}

home_placeholder = st.empty()

# ======================================================================
# HOME SCREEN -- shown until an exercise is picked
# ======================================================================
if st.session_state.selected_exercise is None:
    with home_placeholder.container():
        st.markdown('<div class="console-title">🏋️ RepVision — AI-Powered Exercise Tracking System</div>', unsafe_allow_html=True)
        st.markdown('<div class="console-subtitle">Choose an exercise to begin</div>', unsafe_allow_html=True)
        st.markdown('<div class="home-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="exercise-menu">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💪\n\nPush-Ups", key="pick_pushup"):
                manager.set_target_exercise("pushup")
                st.session_state.selected_exercise = "pushup"
                manager.reset_identification()
                home_placeholder.empty()
                st.rerun()
            st.markdown(
                '<div class="exercise-caption">Tracks elbow angle &mdash; face the camera in plank position</div>',
                unsafe_allow_html=True,
            )

        with col2:
            if st.button("🦵\n\nSquats", key="pick_squat"):
                manager.set_target_exercise("squat")
                st.session_state.selected_exercise = "squat"
                manager.reset_identification()
                home_placeholder.empty()
                st.rerun()
            st.markdown(
                '<div class="exercise-caption">Tracks knee angle &mdash; face the camera standing upright</div>',
                unsafe_allow_html=True,
            )

        with col3:
            if st.button("🧘\n\nPlank", key="pick_plank"):
                manager.set_target_exercise("plank")
                st.session_state.selected_exercise = "plank"
                manager.reset_identification()
                home_placeholder.empty()
                st.rerun()
            st.markdown(
                '<div class="exercise-caption">Tracks hold time &mdash; face the camera in plank position</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)


        st.markdown(
            '<p style="text-align:center; color:var(--text-muted); font-family:Inter,sans-serif; '
            'font-size:0.85rem; margin-top:2.5rem;">More exercises coming soon</p>',
            unsafe_allow_html=True,
        )
    st.stop()
else:
    home_placeholder.empty()

# ======================================================================
# EXERCISE SCREEN
# ======================================================================
exercise_label = EXERCISE_LABELS.get(st.session_state.selected_exercise, "Exercise")
st.markdown(f'<div class="console-title">{exercise_label}</div>', unsafe_allow_html=True)
subtitle_placeholder = st.empty()
subtitle_placeholder.markdown(
    '<div class="console-subtitle">Step in front of the camera to begin</div>', unsafe_allow_html=True
)

# ---------------- Sidebar, grouped into labeled sections ----------------
st.sidebar.markdown('<div class="sidebar-section-label">Session</div>', unsafe_allow_html=True)
if st.sidebar.button("⬅ Back to Menu", type="primary"):
    if manager.state == EXERCISING:
        manager.end_session("left exercise screen")
    st.session_state.selected_exercise = None
    manager.set_target_exercise(None)
    manager.reset_identification()
    st.rerun()

end_clicked = st.sidebar.button("End Session")

st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-section-label">Admin</div>', unsafe_allow_html=True)
if st.sidebar.button("Refresh registered users"):
    manager.set_known_users(get_all_face_embeddings())
    st.sidebar.success(f"Loaded {len(manager.known_users)} user(s).")

if st.session_state.capture_active:
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    if st.sidebar.button("Cancel registration"):
        reset_capture_state()
        manager.reset_identification()
        st.rerun()

run = True

video_col, stats_col = st.columns([3, 1])
with video_col:
    frame_placeholder = st.empty()
with stats_col:
    status_placeholder = st.empty()
    stats_placeholder = st.empty()

registration_placeholder = st.empty()

if run:
    if st.session_state.capture_active:
        # ------------------------------------------------------------
        # Guided face-capture loop.
        # ------------------------------------------------------------
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not camera.isOpened():
            st.error("Could not open webcam.")
        else:
            try:
                while run and st.session_state.capture_active:
                    success, frame = camera.read()
                    if not success:
                        st.error("Could not read frame.")
                        break

                    pose_index = st.session_state.capture_pose_index
                    if pose_index >= len(POSES):
                        break

                    prompt = POSES[pose_index]
                    embedding, box = manager.face_service.get_embedding(frame)

                    if box is not None:
                        x, y, bw, bh = box
                        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 255), 2)

                    cv2.putText(frame, prompt, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(frame, f"{pose_index + 1}/{len(POSES)}", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    name = st.session_state.pending_registration["name"]
                    subtitle_placeholder.markdown(
                        f'<div class="console-subtitle">Registering {name}</div>', unsafe_allow_html=True
                    )
                    status_placeholder.markdown(
                        render_status(f"Pose {pose_index + 1}/{len(POSES)}: {prompt}"),
                        unsafe_allow_html=True,
                    )
                    stats_placeholder.empty()

                    if st.session_state.capture_pose_start is None:
                        st.session_state.capture_pose_start = time.time()

                    elapsed = time.time() - st.session_state.capture_pose_start
                    if elapsed >= POSE_PAUSE_SEC and embedding is not None:
                        st.session_state.capture_samples.append(embedding)
                        st.session_state.capture_pose_index += 1
                        st.session_state.capture_pose_start = None
            finally:
                camera.release()

        if st.session_state.capture_pose_index >= len(POSES):
            samples = st.session_state.capture_samples
            duplicate_name = check_duplicate(manager.face_service, manager.known_users, samples)

            if duplicate_name is not None:
                st.error(f"This face is already registered as '{duplicate_name}'. Registration cancelled.")
            else:
                info = st.session_state.pending_registration
                new_id = finalize_registration(
                    info["name"], info["age"], info["height_cm"], info["weight_kg"], samples
                )
                manager.set_known_users(get_all_face_embeddings())
                st.success(f"Registered '{info['name']}' with user id {new_id}.")

            reset_capture_state()
            manager.reset_identification()
            st.rerun()

    else:
        # ------------------------------------------------------------
        # Normal scanning / exercising loop.
        # ------------------------------------------------------------
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not camera.isOpened():
            st.error("Could not open webcam.")
        else:
            try:
                while run and not st.session_state.capture_active:
                    success, frame = camera.read()
                    if not success:
                        st.error("Could not read frame.")
                        break

                    frame = manager.process(frame, draw_overlay=False)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                    if manager.state == IDLE:
                        subtitle_placeholder.markdown(
                            f'<div class="console-subtitle">{manager.message}</div>', unsafe_allow_html=True
                        )
                        status_placeholder.markdown(render_status(manager.message), unsafe_allow_html=True)
                        stats_placeholder.empty()
                        registration_placeholder.empty()

                    elif manager.state == UNKNOWN:
                        subtitle_placeholder.markdown(
                            '<div class="console-subtitle">New face detected</div>', unsafe_allow_html=True
                        )
                        status_placeholder.markdown(
                            render_status("New face detected &mdash; register below."),
                            unsafe_allow_html=True,
                        )
                        stats_placeholder.empty()
                        break
                    elif manager.state == EXERCISING:
                        subtitle_placeholder.markdown(
                            f'<div class="console-subtitle">Live session &mdash; {manager.user_name}</div>',
                            unsafe_allow_html=True,
                        )
                        status_placeholder.markdown(
                            render_status(f"Session live &mdash; {manager.user_name}", live=True),
                            unsafe_allow_html=True,
                        )
                        today_push, today_squat = manager._today_totals()
                        elapsed = time.time() - manager.session_started

                        if st.session_state.selected_exercise == "plank":
                            PLANK_GOAL_SECONDS = 60
                            stats_placeholder.markdown(
                                render_stats(
                                    rows=[("HOLD TIME", format_duration(manager.plank_seconds), True)],
                                    footer="In position" if manager.plank_in_position else "Position broken -- reset to plank",
                                    progress=manager.plank_seconds / PLANK_GOAL_SECONDS,
                                    duration=format_duration(elapsed),
                                ),
                                unsafe_allow_html=True,
                            )
                        else:
                            count = manager.pushup_count if st.session_state.selected_exercise == "pushup" else manager.squat_count
                            today = today_push if st.session_state.selected_exercise == "pushup" else today_squat
                            stats_placeholder.markdown(
                                render_stats(
                                    rows=[(exercise_label.upper(), count, True)],
                                    footer=f"Today: {today} {exercise_label.lower()}",
                                    progress=count / REP_GOAL,
                                    duration=format_duration(elapsed),
                                ),
                                unsafe_allow_html=True,
                            )
                        registration_placeholder.empty()
                        if end_clicked:
                            manager.end_session("ended by user")
                            end_clicked = False



                    elif manager.state == SUMMARY:
                        subtitle_placeholder.markdown(
                            '<div class="console-subtitle">Session saved</div>', unsafe_allow_html=True
                        )
                        status_placeholder.markdown(
                            render_status("Session saved", live=False),
                            unsafe_allow_html=True,
                        )
                        count = manager.pushup_count if st.session_state.selected_exercise == "pushup" else manager.squat_count
                        stats_placeholder.markdown(
                            render_stats(rows=[(exercise_label.upper(), count, True)]),
                            unsafe_allow_html=True,
                        )
                        registration_placeholder.empty()
            finally:
                camera.release()

# --- Registration form ---
if (
    run
    and manager.state == UNKNOWN
    and not st.session_state.capture_active
    and st.session_state.pending_registration is None
):
    with registration_placeholder.container():
        st.subheader("New User Registration")
        with st.form("registration_form"):
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=18, max_value=100, step=1)
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=0.5)
            weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, step=0.5)
            consent = st.checkbox("I consent to storing my face data for recognition.")
            submitted = st.form_submit_button("Continue to face capture")

            if submitted:
                if not name.strip():
                    st.error("Name is required.")
                elif not consent:
                    st.error("Consent is required to register.")
                else:
                    st.session_state.pending_registration = {
                        "name": name.strip(),
                        "age": int(age),
                        "height_cm": float(height_cm),
                        "weight_kg": float(weight_kg),
                    }
                    st.session_state.capture_active = True
                    st.session_state.capture_pose_index = 0
                    st.session_state.capture_samples = []
                    st.session_state.capture_pose_start = None
                    st.rerun()