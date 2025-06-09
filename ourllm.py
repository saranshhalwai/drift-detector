import difflib
from typing import List
import mcp.types as types

def genratequestionnaire(model: str, capabilities: str) -> List[types.SamplingMessage]:
    """
    Generate a baseline questionnaire for the given model.
    Returns a list of SamplingMessage instances (role="user") with diagnostic questions.
    """
    questions = [
        f"Model Name: {model}\nPlease confirm your model name.",
        f"Capabilities Overview:\n{capabilities}\nPlease summarize your key capabilities.",
        "Describe a typical use-case scenario that demonstrates these capabilities.",
    ]
    return [
        types.SamplingMessage(
            role="user",
            content=types.TextContent(type="text", text=q)
        )
        for q in questions
    ]


def gradeanswers(old_answers: List[str], new_answers: List[str]) -> List[types.SamplingMessage]:
    """
    Compare the old and new answers to compute a drift score.
    Returns a list with a single SamplingMessage (role="assistant") whose content.text is the drift percentage.
    """
    total = len(old_answers)
    if total == 0:
        drift_pct = 0.0
    else:
        # Count how many answers are sufficiently similar
        similar_count = 0
        for old, new in zip(old_answers, new_answers):
            ratio = difflib.SequenceMatcher(None, old, new).ratio()
            if ratio >= 0.8:
                similar_count += 1
        drift_pct = round((1 - (similar_count / total)) * 100, 2)

    drift_text = f"{drift_pct}"
    return [
        types.SamplingMessage(
            role="assistant",
            content=types.TextContent(type="text", text=drift_text)
        )
    ]
