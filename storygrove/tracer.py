import json
import os
import uuid
from datetime import datetime, timezone
from threading import Thread

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
TRACES_REPO = "build-small-hackathon/storygrove-traces"


def push_async(session) -> None:
    """Build and push a completed story trace in a background thread (non-blocking)."""
    trace = _build_trace(session)
    Thread(target=_push, args=(trace,), daemon=True).start()


def _build_trace(session) -> dict:
    skeleton = getattr(session, "skeleton", {}) or {}

    trace = {
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "app": "storygrove",
        # user inputs
        "character": session.character,
        "age_range": session.age_range,
        "theme": session.theme,
        "language": session.language,
        # skeleton fields (flat — matches story_train.md schema)
        "paradigm": skeleton.get("paradigm", ""),
        "title": skeleton.get("title", ""),
        "protagonist": skeleton.get("protagonist", session.character),
        "character_appearance": skeleton.get("character_appearance", ""),
        "initial_trait": skeleton.get("initial_trait", ""),
        "initial_emotion": skeleton.get("initial_emotion", ""),
        "main_conflict": skeleton.get("main_conflict", ""),
        "external_goal": skeleton.get("external_goal", ""),
        "internal_goal": skeleton.get("internal_goal", ""),
        "mentor": skeleton.get("mentor", ""),
        "ally": skeleton.get("ally", ""),
        "antagonist": skeleton.get("antagonist", ""),
        "setting": skeleton.get("setting", ""),
        "symbolic_object": skeleton.get("symbolic_object", ""),
        "value_learned": skeleton.get("value_learned", ""),
        "final_emotion": skeleton.get("final_emotion", ""),
        "completed": any(b.get("is_final") for b in getattr(session, "full_beats", [])),
    }

    # flatten beats as beat_N_narrative / beat_N_choices / beat_N_choice_made
    for beat in getattr(session, "full_beats", []):
        n = beat["beat_number"]
        trace[f"beat_{n}_narrative"] = beat.get("narrative", "")
        trace[f"beat_{n}_choices"] = " | ".join(beat.get("choices", []))
        trace[f"beat_{n}_choice_made"] = beat.get("choice_made") or ""

    return trace


def _push(trace: dict) -> None:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        content = (json.dumps(trace, ensure_ascii=False) + "\n").encode("utf-8")
        api.upload_file(
            path_or_fileobj=content,
            path_in_repo=f"traces/{trace['trace_id']}.jsonl",
            repo_id=TRACES_REPO,
            repo_type="dataset",
            commit_message=f"trace: {trace['character']} / {trace.get('paradigm', '?')}",
        )
        print(f"[Tracer] Pushed trace {trace['trace_id']}")
    except Exception as e:
        print(f"[Tracer] Push failed (non-fatal): {e}")
