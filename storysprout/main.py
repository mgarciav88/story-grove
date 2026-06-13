import gradio as gr

try:
    import spaces
    _gpu = spaces.GPU
except ImportError:
    _gpu = lambda fn: fn  # no-op for local dev

from .narrator import narrate
from .image_generator import generate_image
from .story_generator import start_story, continue_story, StorySession
from .prompts import load_prompts

# ── Load config ────────────────────────────────────────────────────────────────

prompts = load_prompts()
age_ranges = prompts["options"]["age_ranges"]
languages = prompts["options"]["languages"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _stream_and_narrate(generator):
    """
    Consume a beat generator, streaming narrative text.
    Once complete, narrate the beat and generate an illustration.
    Yields: (narrative, beat, audio, image)
    """
    last_narrative = ""
    last_beat = None

    for narrative, beat in generator:
        last_narrative = narrative
        last_beat = beat
        yield narrative, None, None, None

    if last_narrative:
        sample_rate, wav = narrate(last_narrative)
        image = generate_image(last_narrative)
        yield last_narrative, last_beat, (sample_rate, wav), image


def _make_outputs(narrative, beat, audio, image, session):
    """Build the tuple of Gradio output updates."""
    if beat is None or beat.is_final:
        story_text = narrative
        if beat and beat.is_final:
            story_text += "\n\n🌟 The End! What a wonderful adventure!"
        return (
            story_text,
            gr.update(visible=False),
            gr.update(choices=[]),
            audio,
            image,
            session,
        )
    else:
        return (
            narrative,
            gr.update(visible=True),
            gr.update(choices=beat.choices, value=None),
            audio,
            image,
            session,
        )


# ── Handlers ───────────────────────────────────────────────────────────────────

@_gpu
def on_start_story(character, age_range, theme, language):
    if not character.strip():
        yield ("Please enter a character name.", gr.update(visible=False), gr.update(choices=[]), None, None, None)
        return

    if not theme.strip():
        yield ("Please enter a story theme.", gr.update(visible=False), gr.update(choices=[]), None, None, None)
        return

    yield ("✨ Starting your story...", gr.update(visible=False), gr.update(choices=[]), None, None, None)

    generator, session = start_story(
        character=character.strip(),
        age_range=age_range,
        theme=theme.strip(),
        language=languages[language],
    )

    last_narrative = ""
    last_beat = None
    last_audio = None
    last_image = None

    for narrative, beat, audio, image in _stream_and_narrate(generator):
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
        yield ("No active story session. Please start a new story.", gr.update(visible=False), gr.update(choices=[]), None, None, None)
        return

    yield (f"✨ You chose: {choice}\n\nContinuing the story...", gr.update(visible=False), gr.update(choices=[]), None, None, session)

    generator, session = continue_story(session=session, choice=choice)

    last_narrative = ""
    last_beat = None
    last_audio = None
    last_image = None

    for narrative, beat, audio, image in _stream_and_narrate(generator):
        last_narrative = narrative
        last_beat = beat
        if audio is not None:
            last_audio = audio
        if image is not None:
            last_image = image
        yield _make_outputs(narrative, beat, audio, image, session)

    yield _make_outputs(last_narrative, last_beat, last_audio, last_image, session)


# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="StorySprout 🌱") as demo:
    session_state = gr.State(None)

    gr.Markdown(
        """
        # 🌱 StorySprout
        ### An interactive story just for you
        """
    )

    with gr.Row():
        # ── Left column: controls ──────────────────────────────────────────────
        with gr.Column(scale=1):
            character_input = gr.Textbox(
                label="Main Character",
                placeholder="e.g. Luna the rabbit, a brave knight called Max...",
                max_lines=1,
            )
            age_range_input = gr.Dropdown(
                label="Age Range",
                choices=age_ranges,
                value=age_ranges[0],
            )
            theme_input = gr.Textbox(
                label="Story Theme",
                placeholder="e.g. making new friends, being brave in the dark...",
                max_lines=2,
            )
            language_input = gr.Radio(
                label="Language",
                choices=list(languages.keys()),
                value="English",
            )
            start_btn = gr.Button("✨ Begin Story", variant="primary", size="lg")

        # ── Right column: story output ─────────────────────────────────────────
        with gr.Column(scale=2):
            story_output = gr.Textbox(
                label="Your Story",
                lines=10,
                interactive=False,
                placeholder="Your story will appear here...",
            )
            image_output = gr.Image(
                label="📖 Story Illustration",
                type="pil",
            )
            audio_output = gr.Audio(
                label="🔊 Listen to the story",
                type="numpy",
                autoplay=True,
            )
            with gr.Row(visible=False) as choices_row:
                choice_selector = gr.Radio(
                    label="What happens next?",
                    choices=[],
                    interactive=True,
                )
                choose_btn = gr.Button("→ Choose", variant="primary")

    # ── Event handlers ─────────────────────────────────────────────────────────

    _outputs = [story_output, choices_row, choice_selector, audio_output, image_output, session_state]

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