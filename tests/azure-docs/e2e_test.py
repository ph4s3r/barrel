"""End-to-end tests."""
import re
from pathlib import Path

from fastapi.testclient import TestClient
from openai import OpenAI, RateLimitError, AuthenticationError
import pytest

from main import app
from credentials.secrets import secrets
from tests.utils import (
    load_test_suit, calculate_and_display_test_score,
    display_table, save_test_results, results_dir, TestCase
)


client = TestClient(app)
evaluator_client = OpenAI(api_key=secrets.test_eval_llm_api_key)
current_dir = Path(__file__).parent


def evaluate_test_with_llm(test_case: TestCase) -> dict[str, float | str]:
    """
    Send an evaluation request to an LLM, using a super prompt that
    includes the question, the expected reference answer, and the API answer.
    It expects a JSON response with keys 'rating' (int 1-10) and an 'explanation'.
    """
    eval_result = {}
    prompt = (
        "Please evaluate the candidate's answer for the following questions compared to the "
        "reference answer. Please ignore occasional context repetition, give the best rating "
        "if the candidate is factually right. Give the worst rating if the candidate is wrong."
        f"Question: {test_case['question']}. "
        f"Reference answer: {test_case['reference_answer']}. "
        f"LLM answer: {test_case['llm_answer']}. "
        "Please respond with a rating in the following format: rating=[x] where x is a number "
        "between 0 and 10, representing the quality of the answer. The higher the number  the "
        "more aligns the candidate's answer with the acceptable one. Please always start your "
        "response with rating=[X], then provide a one-sentence explanation about why you gave the rating."
    )

    response = evaluator_client.responses.create(model="gpt-4o", input=prompt)

    # sample answer: 'rating=[9]'
    if match := re.search(r'\d+(?:\.\d+)?', response.output_text):
        number = float(match.group())  # The LLM is not reliable, it can give the rating as a float
        print("Extracted number:", number)
        eval_result = {"rating": number, "eval_explanation": response.output_text}
    else:
        print(f"  FAILED to find rating in LLM evaluation response: {response.output_text}")
        eval_result = {"eval_explanation": response.output_text}

    return eval_result


def test_az_networking_5_qa():
    """Test cases of az-networking-5.yaml."""
    test_yaml = current_dir / "az-networking-5.yaml"
    test_suit = load_test_suit(test_yaml)

    for tc in test_suit:
        response = client.post(
            app.url_path_for("user_prompt"),
            params={"prompt": tc["question"]},
            json={}
        )
        response.raise_for_status()
        tc["llm_answer"] = response.json()
        print(f"  LLM answer updated for: '{tc["question"]}'")

        try:
            eval_result = evaluate_test_with_llm(tc)
        except (AuthenticationError, RateLimitError) as err:
            pytest.exit(reason=f"UNABLE TO EVALUATE TEST: {err.args[0]}")

        tc.update(eval_result)

    print(f"LLM test evaluation for {test_yaml.name}")
    final_score = calculate_and_display_test_score(test_suit)

    # --- Display the test results as a table ---
    display_table(test_suit)

    results = {
        "final_score_percentage": final_score,
        "evaluations": test_suit
    }
    report_file_path = results_dir / test_yaml.name.replace(test_yaml.suffix, ".yaml")
    save_test_results(report_file_path, results)
