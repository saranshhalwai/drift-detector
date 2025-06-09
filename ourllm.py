import difflib
from typing import List
import mcp.types as types
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

llm = init_chat_model("llama-3.1-8b-instant",model_provider='groq')


def genratequestionnaire(model: str, capabilities: str) -> List[types.SamplingMessage]:
    """
    Generate a baseline questionnaire for the given model.
    Returns a list of SamplingMessage instances (role="user") with diagnostic questions.
    """
    global llm
    questions = []
    previously_generated = ""

    for i in range(0,5):
        response = llm.invoke("Generate a questionnaire for a model with the following capabilities:\n"
                              "Model Name: " + model + "\n"
                            "Capabilities Overview:\n" + capabilities + "\n"
                            "Please provide one more question that cover the model's capabilities and typical use-cases.\n"
                            "Previously generated questions:\n" + previously_generated +
                            "\nQuestion " + str(i+1) + ":")
        new_question = str(response.content)
        questions.append(new_question)
        # Update previously_generated to include the new question
        if previously_generated:
            previously_generated += "\n"
        previously_generated += f"Question {i+1}: {new_question}"

    return [
        types.SamplingMessage(
            role="user",
            content=types.TextContent(type="text", text=q)
        )
        for q in questions
    ]


def gradeanswers(old_answers: List[str], new_answers: List[str]) -> List[types.SamplingMessage]:
    """
    Use the LLM to compare the old and new answers to compute a drift score.
    Returns a list with a single SamplingMessage (role="assistant") whose content.text is the drift percentage.
    """
    global llm

    if not old_answers or not new_answers:
        drift_pct = 0.0
    else:
        # Prepare a prompt with old and new answers for the LLM to analyze
        prompt = "You're tasked with detecting semantic drift between two sets of model responses.\n\n"
        prompt += "Original responses:\n"
        for i, ans in enumerate(old_answers):
            prompt += f"Response {i+1}: {ans}\n\n"

        prompt += "New responses:\n"
        for i, ans in enumerate(new_answers):
            prompt += f"Response {i+1}: {ans}\n\n"

        prompt += "Analyze the semantic differences between the original and new responses. "
        prompt += "Provide a drift percentage score (0-100%) that represents how much the meaning, "
        prompt += "intent, or capabilities have changed between the two sets of responses. "
        prompt += "Only return the numerical percentage value without any explanation or additional text."

        # Get the drift assessment from the LLM
        response = llm.invoke(prompt)
        drift_text = str(response.content).strip()

        # Extract just the numerical value if there's extra text
        import re
        drift_match = re.search(r'(\d+\.?\d*)', drift_text)
        if drift_match:
            drift_pct = float(drift_match.group(1))
        else:
            # Fallback if no number found
            drift_pct = 0.0

    drift_text = f"{drift_pct}"
    return [
        types.SamplingMessage(
            role="assistant",
            content=types.TextContent(type="text", text=drift_text)
        )
    ]
