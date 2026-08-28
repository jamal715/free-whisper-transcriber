_BASE_LOGO_DATA_URI = "data:image/webp;base64,UklGRrACAABXRUJQVlA4IKQCAAAwDQCdASpIAEgAPm00lUgkIqIhJTQKqIANiWUAzwWoC2sXCdJNNE8lzGOJc2uvCU4tFS6PG6AtGrgAyrTadwzShnuQVXonbIczgrtZpN0FXi7cqYQyTgksAyUQKHsr1Kl94OQT19T+eY+qj/AXM8E99tAA/u/ZzdEy1bp9cXNkRRJrw2ZN4D+b/C+BQOia2wbhbfN/J0HyEA55Sm3ipRhmju2viFuAAOJAJmPv4IcIatvXFf+WJ1j5xt5uuZGPDMwoZjCzdqi8UmueLQLgBPwhj7g6tX63iVLiTc8ST7Sw4ITxqPTgGduNqQB7oFkAEGF8603Txe/g2EK+O3j0kTAAz9FjDS6SfwI+/WkBNudjt+ywwsr22jt0MbjS/AHBp261oPCOOuxIdY7IZ/KHyjUxyz96O8WQiVGZygoa8iniG+2fynzNpC9PartFCuLafiw/aK81LnHoJosCXcECOTCKR3zgcaMTrKsu4fpp6juitihzipZob2cHHG5vrWSHqOlGXfjs8g2c5yNIeFOCEr+iHGOp6tPqC+JSBnyO9KNCUJALoeSbgVuLMHSVvp04cw5OvnVADhwlnhGoF3Tb2Fz01L6LKFTVGvYXrrMFnwH47iuGOunEilbpndcvyU/MevVgNfJqLVrYZlTxIMnfcDnMM71j5uiD5T+4x6mnclVXvgfmTNABn3YvffSQDUzqg3D16dPdEBpDY0Xt64NpMutOSAXWMLW9w99/lu0+eH+GFHPTi0k5cmtqX3oHuNLoH28RYvLh/URmJI89PxO41Sd6xK+owX83x6UT+K3sbeune9tbxm1eMAbugG1KahA4e2qISfHqpD2Mlhfky3kfknbt9GJav0SLtAiLKYWiUFJMpmhSFySKJY2NmIYdlAPwMGZZAAAA"

# The app already injects this value into the QuantOra header image.  Appending
# the style block here keeps visual changes isolated from the transcription
# pipeline, so UI releases cannot destabilize the speech/QA logic.
_UI_OVERRIDE = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800&family=Newsreader:opsz,wght@6..72,600;6..72,700;6..72,800&display=swap');

:root {
  --q-editorial: #004d73;
  --q-editorial-deep: #003b59;
  --q-editorial-dark: #002f47;
  --q-editorial-text: #f7fafb;
  --q-editorial-muted: #c6d4db;
  --q-editorial-line: rgba(255,255,255,.17);
  --q-editorial-black: #101820;
}

/* Executive editorial typography */
.stApp {
  background: var(--q-editorial) !important;
  color: var(--q-editorial-text) !important;
  font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
  font-weight: 600 !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp .q-name, .stApp .q-display, .stApp .q-health-title {
  font-family: "Newsreader", Georgia, "Times New Roman", serif !important;
  font-weight: 800 !important;
  letter-spacing: -.025em !important;
}
.stApp .q-name { font-size: 33px !important; }
.stApp .q-hero h1, .stApp .q-display {
  font-size: 48px !important;
  line-height: 1.02 !important;
  letter-spacing: -.035em !important;
}
.stApp p, .stApp label, .stApp span, .stApp div {
  font-weight: 600;
}
.stApp .q-panel-title, .stApp .q-ready-label,
.stApp .q-time, .stApp .q-provider {
  font-weight: 800 !important;
  letter-spacing: .08em !important;
}
.stApp .q-tag, .stApp .q-hero p, .stApp .q-live-sub,
.stApp .q-health-sub, .stApp .q-note {
  color: var(--q-editorial-muted) !important;
  font-weight: 600 !important;
}

/* More institutional card treatment */
.stApp div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(0,0,0,.095) !important;
  border: 1px solid var(--q-editorial-line) !important;
  border-radius: 15px !important;
  box-shadow: none !important;
}
.stApp .q-ready,
.stApp div[data-testid="stMetric"] {
  background: rgba(255,255,255,.055) !important;
  border-color: var(--q-editorial-line) !important;
  border-radius: 13px !important;
}
.stApp .q-pipeline, .stApp .q-health {
  background: var(--q-editorial-dark) !important;
  border-color: var(--q-editorial-line) !important;
}
.stApp .q-live {
  background: var(--q-editorial-deep) !important;
  border-color: var(--q-editorial-line) !important;
}

/* CRITICAL CONTRAST FIX: every white field has dark text. */
.stApp div[data-baseweb="select"] > div,
.stApp div[data-baseweb="input"] > div,
.stApp div[data-baseweb="textarea"],
.stApp .stTextInput input,
.stApp .stTextArea textarea,
.stApp input[type="text"],
.stApp input[type="password"],
.stApp textarea {
  background: #ffffff !important;
  color: var(--q-editorial-black) !important;
  -webkit-text-fill-color: var(--q-editorial-black) !important;
  border-color: #d4dce0 !important;
  border-radius: 10px !important;
  font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
  font-weight: 650 !important;
}
.stApp div[data-baseweb="select"] span,
.stApp div[data-baseweb="select"] svg,
.stApp div[data-baseweb="input"] input,
.stApp div[data-baseweb="textarea"] textarea {
  color: var(--q-editorial-black) !important;
  -webkit-text-fill-color: var(--q-editorial-black) !important;
}
.stApp input::placeholder,
.stApp textarea::placeholder {
  color: #75818a !important;
  -webkit-text-fill-color: #75818a !important;
  opacity: 1 !important;
}

/* Dropdown menu lives in a BaseWeb portal outside the app tree. */
[data-baseweb="popover"], [role="listbox"], [role="option"] {
  background: #ffffff !important;
  color: var(--q-editorial-black) !important;
  font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
  font-weight: 650 !important;
}
[role="option"] *, [role="listbox"] * {
  color: var(--q-editorial-black) !important;
  -webkit-text-fill-color: var(--q-editorial-black) !important;
}

/* Keep non-input controls crisp on blue. */
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] *,
.stApp .stCheckbox label, .stApp .stCheckbox label *,
.stApp label[data-baseweb="checkbox"],
.stApp label[data-baseweb="checkbox"] * {
  color: #f7fafb !important;
  font-weight: 700 !important;
}
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] summary * {
  color: #ffffff !important;
  font-weight: 800 !important;
}

/* Buttons feel like an institutional product rather than default Streamlit. */
.stApp .stButton > button[kind="primary"] {
  background: #ffffff !important;
  color: var(--q-editorial-black) !important;
  border: 1px solid #ffffff !important;
  min-height: 51px !important;
  border-radius: 10px !important;
  font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
  font-weight: 800 !important;
}
.stApp .stButton > button[kind="primary"]:hover {
  background: #eaf4f8 !important;
  color: var(--q-editorial-deep) !important;
}
.stApp .stButton > button[kind="secondary"] {
  background: rgba(255,255,255,.075) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,.22) !important;
  font-weight: 800 !important;
}
.stApp div[data-testid="stFileUploader"] button {
  background: #ffffff !important;
  color: var(--q-editorial-black) !important;
  border: none !important;
  font-weight: 800 !important;
}

/* Tabs and metrics */
.stApp .stTabs [data-baseweb="tab"] {
  color: #bfd0d8 !important;
  font-weight: 800 !important;
}
.stApp .stTabs [aria-selected="true"] {
  color: #ffffff !important;
  border-bottom-color: #b9e8f8 !important;
}
.stApp div[data-testid="stMetricLabel"],
.stApp div[data-testid="stMetricLabel"] * {
  color: var(--q-editorial-muted) !important;
  font-weight: 800 !important;
}
.stApp div[data-testid="stMetricValue"],
.stApp div[data-testid="stMetricValue"] * {
  color: #ffffff !important;
  font-family: "Newsreader", Georgia, serif !important;
  font-size: 1.5rem !important;
  font-weight: 800 !important;
}

/* Transcript remains highly readable; not every sentence needs display-serif. */
.stApp .q-text, .stApp .q-live-text {
  color: #f7fafb !important;
  font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
  font-weight: 600 !important;
  line-height: 1.72 !important;
}
.stApp .q-time {
  color: #b9e8f8 !important;
}
.stApp div[data-testid="stAlert"],
.stApp div[data-testid="stAlert"] * {
  color: #ffffff !important;
}
"""

# app.py embeds LOGO_DATA_URI in an <img src="..."> inside an unsafe HTML block.
# Close that image, inject the style override *after* the app's base CSS, then add
# one invisible pixel image so the surrounding markup remains valid.
_TRANSPARENT_PIXEL = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
LOGO_DATA_URI = (
    _BASE_LOGO_DATA_URI
    + '" alt="QuantOra logo"><style>'
    + _UI_OVERRIDE
    + '</style><img style="display:none" src="'
    + _TRANSPARENT_PIXEL
)
