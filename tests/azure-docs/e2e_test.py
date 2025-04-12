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
evaluator_client = OpenAI(api_key=secrets.llm_api_key)
current_dir = Path(__file__).parent


def evaluate_test_with_llm(test_case: TestCase) -> dict[str, float | str]:
    """
    Send an evaluation request to an LLM, using a super prompt that
    includes the question, the expected reference answer, and the API answer.
    It expects a JSON response with keys 'rating' (int 1-10) and an 'explanation'.
    """
    eval_result = {}
    prompt = (
        "Please compare the [Candidate's answer] to the [Reference answer] for the [Question] They are "
        "encapsulated with START and END markers respectively. Start your response with \"rating=[x]\" where"
        " x is a number between 0 and 10, the better the answer the higher the number. Give 10 rating if "
        "the candidate is factually right. Give 0 rating if the candidate is factually wrong. Ignore "
        "occasional context repetition or additional information the candidate provides. Do not use your "
        "own general knowledge in the evaluation, only focus on the distance between the reference answer "
        "and the candidate answer."
        f"""\n[Question START]: {test_case['question']} [Question END]
        \n[Reference answer START]: {test_case['reference_answer']} [Reference answer END]
        \n[Candidate's answer START]: {test_case['llm_answer']}. [Candidate's answer END]"""
        "Provide explanation about your rating in one sentence alone. Sometimes there are candidate "
        "answers like 'We don't have information about this in our vector store', which is a valid answer, "
        "if it aligns with the reference answer, give it a 10."
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


def executor(test_yaml: Path):
    """Test cases of az-networking-2.yaml."""
    test_suit = load_test_suit(test_yaml)

    for tc in test_suit:
        response = client.post(
            app.url_path_for("user_prompt"),
            params={"prompt": tc["question"]},
            json={}
        )
        response.raise_for_status()
        tc["llm_answer"] = response.json()
        # print(f"  Barrel has answered the question: '{tc["question"]}'")

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


def test_az_networking_2_qa():
    """Test cases of az-networking-2.yaml."""
    test_yaml = current_dir / "az-networking-2.yaml"
    executor(test_yaml)
