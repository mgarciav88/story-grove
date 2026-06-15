import gradio as gr
from .story_generator import generate_story
from .prompts import load_prompts

# ── Load config once ───────────────────────────────────────────────────────────

prompts = load_prompts()
age_ranges = prompts["options"]["age_ranges"]
languages = prompts["options"]["languages"]

# ── Story Generation Handler ───────────────────────────────────────────────────

def on_generate(
    character: str,
    age_range: str,
    theme: str,
    language: str,
):
    # basic validation
    if not character.strip():
        yield "Please enter a character name."
        return
    if not theme.strip():
        yield "Please enter a story theme."
        return

    yield "✨ Writing your story..."

    for partial in generate_story(
        character=character.strip(),
        age_range=age_range,
        theme=theme.strip(),
        language=languages[language],
    ):
        yield partial

# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="StorySpout 🌱") as demo:

    gr.Markdown(
        """
        # 🌱 StoryProut
        ### Create a magical story for your little one
        """
    )

    with gr.Row():
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
            generate_btn = gr.Button(
                "✨ Generate Story",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=2):
            story_output = gr.Textbox(
                label="Your Story",
                lines=20,
                placeholder="Your story will appear here...",
            )

    generate_btn.click(
        fn=on_generate,
        inputs=[
            character_input,
            age_range_input,
            theme_input,
            language_input,
        ],
        outputs=story_output,
    )

if __name__ == "__main__":
    demo.launch()
