import gradio as gr

BOOK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Cinzel:wght@600;700&display=swap');

.gradio-container { background: #1C0D05 !important; max-width: 1100px !important; margin: auto !important; }

/* ── Cover ────────────────────────────────────────────────────────────────── */
.cover-panel {
    background: linear-gradient(150deg, #5C1A08 0%, #963B1A 50%, #5C1A08 100%);
    border-radius: 4px;
    padding: 48px 40px;
    min-height: 520px;
    box-shadow: 6px 0 24px rgba(0,0,0,0.5), inset -4px 0 12px rgba(0,0,0,0.3);
}

.cover-title h1 {
    font-family: 'Cinzel', serif !important;
    font-size: 2.8em !important;
    color: #F5DEB3 !important;
    text-align: center !important;
    text-shadow: 1px 2px 8px rgba(0,0,0,0.6) !important;
    letter-spacing: 3px !important;
    margin-bottom: 4px !important;
}

.cover-title h3 {
    font-family: 'Lora', serif !important;
    font-style: italic !important;
    color: #D2A679 !important;
    text-align: center !important;
    font-weight: 400 !important;
}

/* ── Status ───────────────────────────────────────────────────────────────── */
.status-msg p {
    font-family: 'Lora', serif !important;
    font-style: italic !important;
    color: #8B4513 !important;
    text-align: center !important;
    padding: 24px !important;
    background: #FDF6E3 !important;
    border-radius: 4px !important;
    box-shadow: 0 4px 32px rgba(0,0,0,0.4) !important;
}

/* ── Choices ──────────────────────────────────────────────────────────────── */
.choices-area {
    background: #F5EDD5 !important;
    border-top: 1px solid #D2B48C !important;
    border-radius: 0 0 4px 4px !important;
    padding: 16px 24px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
}
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Lora"), "Georgia", "serif"],
).set(
    body_background_fill="#1C0D05",
    button_primary_background_fill="#7B3310",
    button_primary_background_fill_hover="#963B1A",
    button_primary_text_color="#F5DEB3",
    button_primary_text_color_hover="#FDF6E3",
)
