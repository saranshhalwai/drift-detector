import difflib
from typing import List
import mcp.types as types
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
import re

# Load environment variables from .env file
load_dotenv()
print("GROQ_API_KEY is set:", "GROQ_API_KEY" in os.environ)

llm = init_chat_model("llama-3.1-8b-instant", model_provider='groq')


def genratequestionnaire(model: str, capabilities: str) -> List[str]:
    """
    Generate a baseline questionnaire for the given model.
    Returns a list of question strings for diagnostic purposes.
    """
    global llm
    questions = []
    previously_generated = ""

    for i in range(5):
        try:
            response = llm.invoke(
                f"Generate a questionnaire for a model with the following capabilities:\n"
                f"Model Name: {model}\n"
                f"Capabilities Overview:\n{capabilities}\n"
                f"Please provide one more question that covers the model's capabilities and typical use-cases.\n"
                f"Previously generated questions:\n{previously_generated}\n"
                f"Question {i + 1}:"
            )
            new_question = str(response.content).strip()
            questions.append(new_question)

            # Update previously_generated to include the new question
            if previously_generated:
                previously_generated += "\n"
            previously_generated += f"Question {i + 1}: {new_question}"

        except Exception as e:
            print(f"Error generating question {i + 1}: {e}")
            # Fallback question
            questions.append(f"What are your capabilities as {model}?")

    return questions


def gradeanswers(old_answers: List[str], new_answers: List[str]) -> str:
    """
    Use the LLM to compare the old and new answers to compute a drift score.
    Returns a drift percentage as a string.
    """
    global llm

    if not old_answers or not new_answers:
        return "0"

    if len(old_answers) != len(new_answers):
        return "100"  # Major drift if answer count differs

    try:
        # Prepare a prompt with old and new answers for the LLM to analyze
        prompt = "You're tasked with detecting semantic drift between two sets of model responses.\n\n"
        prompt += "Original responses:\n"
        for i, ans in enumerate(old_answers):
            prompt += f"Response {i + 1}: {ans}\n\n"

        prompt += "New responses:\n"
        for i, ans in enumerate(new_answers):
            prompt += f"Response {i + 1}: {ans}\n\n"

        prompt += ("Analyze the semantic differences between the original and new responses. "
                   "Provide a drift percentage score (0-100%) that represents how much the meaning, "
                   "intent, or capabilities have changed between the two sets of responses. "
                   "Only return the numerical percentage value without any explanation or additional text.")

        # Get the drift assessment from the LLM
        response = llm.invoke(prompt)
        drift_text = str(response.content).strip()

        # Extract just the numerical value if there's extra text
        drift_match = re.search(r'(\d+\.?\d*)', drift_text)
        if drift_match:
            drift_pct = float(drift_match.group(1))
            return str(int(drift_pct))  # Return as integer string
        else:
            # Fallback: calculate simple text similarity
            similarity_scores = []
            for old, new in zip(old_answers, new_answers):
                similarity = difflib.SequenceMatcher(None, old, new).ratio()
                similarity_scores.append(similarity)

            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            drift_pct = (1 - avg_similarity) * 100
            return str(int(drift_pct))

    except Exception as e:
        print(f"Error grading answers: {e}")
        # Fallback: calculate simple text similarity
        similarity_scores = []
        for old, new in zip(old_answers, new_answers):
            similarity = difflib.SequenceMatcher(None, old, new).ratio()
            similarity_scores.append(similarity)

        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        drift_pct = (1 - avg_similarity) * 100
        return str(int(drift_pct))