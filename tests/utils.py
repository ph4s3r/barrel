"""Test helper methods."""
from pathlib import Path
from typing import TypedDict

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


results_dir = Path(__file__).parent / "reports"
results_dir.mkdir(parents=True, exist_ok=True)

TestCase = TypedDict(
    "TestCase",
    {
        "question": str,
        "reference_answer": str,
        "llm_answer": None | str,
        "eval_explanation": str | None,
        "rating": int | float
    }
)


def load_test_suit(yaml_file: Path) -> list[TestCase]:
    """Gather Q&A data"""
    with open(yaml_file, "r", encoding="UTF-8") as file:
        data = yaml.safe_load(file)

    return [
        {
            "question": item["question"].rstrip(),
            "reference_answer": item["answer"].rstrip(),
            "llm_answer": None,
            "eval_explanation": "",
            "rating": -1
        }
        for item in data["questions"]
    ]


def display_table(test_suit: list[TestCase]):
    """Print the evaluation results to the console in a nice table format."""
    table = Table(title="Test Results", show_lines=True)

    table.add_column("Question", style="magenta", justify="full")
    table.add_column("Reference Answer", style="green", justify="full")
    table.add_column("LLM Answer", style="yellow", justify="full")
    table.add_column("Rating", style="green", justify="center", highlight=True)
    table.add_column("Evaluation", style="green", justify="full")

    for tc in test_suit:
        table.add_row(
            tc["question"],
            tc["reference_answer"],
            tc["llm_answer"],
            str(tc["rating"]),
            tc["eval_explanation"]
        )

    console = Console()
    console.print(table)


def calculate_and_display_test_score(test_suit: list[dict]) -> int:
    """Calculate the score of the LLM evaluation in percentage."""
    final_score_percentage = 0

    ratings = [tc["rating"] for tc in test_suit if tc["rating"] != -1]
    if ratings:
        final_score_percentage = (sum(ratings) / (len(ratings) * 10)) * 100

    # --- Display final score header ---
    console = Console()
    header_text = f"[bold blue]Final Score: {final_score_percentage:.2f}%[/bold blue]"
    console.print(Panel(header_text, title="Evaluation Score", expand=False))

    return final_score_percentage


def save_test_results(file_path: Path, results: dict) -> None:
    """Save the test results to a file."""
    with open(file_path, "w", encoding="utf-8") as out_file:
        yaml.dump(
            results,
            out_file,
            default_flow_style=False,
            sort_keys=False
        )
