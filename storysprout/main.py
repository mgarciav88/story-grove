import html as _html
import base64
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import gradio as gr

try:
    import spaces
    _gpu = spaces.GPU
except ImportError:
    _gpu = lambda fn: fn

from .narrator import narrate
from .image_generator import generate_image
from .story_generator import start_story, continue_story, StorySession
from .prompts import load_prompts
from .ui.theme import BOOK_CSS, theme as book_theme
from . import vram_manager
from .vram_manager import Model, VRAM_SWAP

# ── Config ─────────────────────────────────────────────────────────────────────

prompts = load_prompts()
age_ranges = prompts["options"]["age_ranges"]
languages = prompts["options"]["languages"]


# ── Book page renderer ─────────────────────────────────────────────────────────

def _book_page(narrative: str, image=None, is_odd: bool = True) -> str:
    text_html = _html.escape(narrative).replace('\n', '<br>')

    text_col = f'''
        <div style="flex:1;min-width:0;font-family:'Lora',Georgia,serif;
                    font-size:1.08em;line-height:1.9;color:#2C1810;padding:0 12px;">
            {text_html}
        </div>'''

    if image is not None:
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        img_col = f'''
            <div style="flex:1;min-width:0;">
                <img src="data:image/jpeg;base64,{b64}"
                     style="width:100%;border-radius:4px;
                            box-shadow:2px 4px 16px rgba(0,0,0,0.15);"/>
            </div>'''
    else:
        img_col = '<div style="flex:1;min-width:0;"></div>'

    left, right = (text_col, img_col) if is_odd else (img_col, text_col)

    return f'''
        <div style="display:flex;gap:40px;align-items:flex-start;
                    background:#FDF6E3;padding:36px;border-radius:2px;
                    box-shadow:0 4px 32px rgba(0,0,0,0.4),
                               inset 4px 0 12px rgba(0,0,0,0.05);
                    min-height:480px;">
            {left}{right}
        </div>'''


# ── Stream helper ──────────────────────────────────────────────────────────────

def _generate_image_and_audio(narrative: str, session: StorySession):
    """
    Generator that yields (image, audio) twice:
      1st yield: image ready, audio=None (still generating)
      2nd yield: audio ready
    Parallel when models coexist in VRAM, sequential otherwise.
    """
    if not VRAM_SWAP:
        vram_manager.request(Model.IMAGE)
        vram_manager.request(Model.NARRATOR)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_image = executor.submit(generate_image, narrative, session.visual_profile, session.image_seed)
            future_audio = executor.submit(narrate, narrative)
            image = future_image.result()
            yield image, None
            audio = future_audio.result()
        yield image, audio
    else:
        image = generate_image(narrative, session.visual_profile, session.image_seed)
        yield image, None
        audio = narrate(narrative)
        yield image, audio


def _stream_and_narrate(generator, session: StorySession):
    last_narrative = ""
    last_beat = None

    for narrative, beat in generator:
        last_narrative = narrative
        last_beat = beat
        yield narrative, None, None, None

    if last_narrative:
        for image, audio in _generate_image_and_audio(last_narrative, session):
            if audio is None:
                yield last_narrative, last_beat, None, image
                time.sleep(1)
            else:
                yield last_narrative, last_beat, audio, image


# ── Output helpers ─────────────────────────────────────────────────────────────

# outputs order: [story_tabs, status_msg, story_page, audio_output, choices_row, choice_selector, session_state]

def _make_outputs(narrative, beat, audio, image, session):
    is_odd = session.beat_number % 2 == 1
    story_text = narrative
    if beat and beat.is_final:
        story_text += "\n\n🌟 The End! What a wonderful adventure!"
    show_choices = beat is not None and not beat.is_final

    return (
        gr.update(selected="story"),
        gr.update(value="", visible=False),
        _book_page(story_text, image, is_odd),
        audio,
        gr.update(visible=show_choices),
        gr.update(choices=beat.choices if show_choices else [], value=None),
        session,
    )


def _loading(message: str):
    return (
        gr.update(selected="story"),
        gr.update(value=message, visible=True),
        "",
        None,
        gr.update(visible=False),
        gr.update(choices=[]),
        None,
    )


# ── Handlers ───────────────────────────────────────────────────────────────────

@_gpu
def on_start_story(character, age_range, theme_input, language):
    if not character.strip():
        raise gr.Error("Please enter a character name.")
    if not theme_input.strip():
        raise gr.Error("Please enter a story theme.")

    yield _loading("✨ Opening the book...")

    generator, session = start_story(
        character=character.strip(),
        age_range=age_range,
        theme=theme_input.strip(),
        language=languages[language],
    )

    last_narrative = ""
    last_beat = None
    last_audio = None
    last_image = None

    for narrative, beat, audio, image in _stream_and_narrate(generator, session):
        last_narrative = narrative
        last_beat = beat
        if audio is not None:
            last_audio = audio
        if image is not None:
            last_image = image
        yield _make_outputs(narrative, beat, audio, image, session)

    yield _make_outputs(last_narrative, last_beat, last_audio, last_image, session)


@_gpu
def on_choice_selected(choice, session):
    if session is None:
        raise gr.Error("No active story. Please start a new story.")

    yield (
        gr.update(selected="story"),
        gr.update(value=f"✨ {choice}…", visible=True),
        gr.update(),
        gr.update(),
        gr.update(visible=False),
        gr.update(choices=[]),
        session,
    )

    generator, session = continue_story(session=session, choice=choice)

    last_narrative = ""
    last_beat = None
    last_audio = None
    last_image = None

    for narrative, beat, audio, image in _stream_and_narrate(generator, session):
        last_narrative = narrative
        last_beat = beat
        if audio is not None:
            last_audio = audio
        if image is not None:
            last_image = image
        yield _make_outputs(narrative, beat, audio, image, session)

    yield _make_outputs(last_narrative, last_beat, last_audio, last_image, session)


# ── UI ──────────────────────────────────────────────────────────────────────────

with gr.Blocks(theme=book_theme, css=BOOK_CSS, title="StorySprout 🌱") as demo:
    session_state = gr.State(None)

    with gr.Tabs(selected="cover") as story_tabs:

        # ── Cover ──────────────────────────────────────────────────────────────
        with gr.Tab(label="📖 New Story", id="cover"):
            with gr.Column(elem_classes="cover-panel"):
                gr.Markdown(
                    "# 🌱 StorySprout\n### *An interactive story just for you*",
                    elem_classes="cover-title",
                )
                character_input = gr.Textbox(
                    label="Main Character",
                    placeholder="e.g. Luna the rabbit, a brave knight called Max...",
                    max_lines=1,
                )
                with gr.Row():
                    age_range_input = gr.Dropdown(
                        label="Age Range",
                        choices=age_ranges,
                        value=age_ranges[0],
                        scale=1,
                    )
                    language_input = gr.Radio(
                        label="Language",
                        choices=list(languages.keys()),
                        value="English",
                        scale=2,
                    )
                theme_input = gr.Textbox(
                    label="Story Theme",
                    placeholder="e.g. making new friends, being brave in the dark...",
                    max_lines=2,
                )
                start_btn = gr.Button("✨ Begin Story", variant="primary", size="lg")

        # ── Story ──────────────────────────────────────────────────────────────
        with gr.Tab(label="📚 Your Story", id="story"):
            status_msg = gr.Markdown("", visible=False, elem_classes="status-msg")
            story_page = gr.HTML("")
            audio_output = gr.Audio(
                label="🔊 Listen",
                type="numpy",
                autoplay=True,
            )
            with gr.Row(visible=False, elem_classes="choices-area") as choices_row:
                choice_selector = gr.Radio(
                    label="What happens next?",
                    choices=[],
                    interactive=True,
                )
                choose_btn = gr.Button("→ Choose", variant="primary")

    # ── Wiring ─────────────────────────────────────────────────────────────────

    _outputs = [story_tabs, status_msg, story_page, audio_output, choices_row, choice_selector, session_state]

    start_btn.click(
        fn=on_start_story,
        inputs=[character_input, age_range_input, theme_input, language_input],
        outputs=_outputs,
    )

    choose_btn.click(
        fn=on_choice_selected,
        inputs=[choice_selector, session_state],
        outputs=_outputs,
    )


if __name__ == "__main__":
    demo.launch()