from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Commit 7 final repository checks")
    parser.add_argument(
        "--notebook-path",
        type=str,
        default="f23-0647_Assignment2_DS-C.ipynb",
    )
    parser.add_argument("--report-path", type=str, default="report.pdf")
    parser.add_argument(
        "--checklist-out",
        type=str,
        default="data/submission_checklist.json",
    )
    return parser.parse_args()


def notebook_execution_status(notebook_path: Path) -> dict[str, object]:
    if not notebook_path.exists():
        return {
            "exists": False,
            "executed_code_cells": 0,
            "total_code_cells": 0,
            "all_code_cells_executed": False,
            "all_code_cells_have_output": False,
        }

    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    executed = [c for c in code_cells if c.get("execution_count") is not None]
    with_output = [c for c in code_cells if c.get("outputs")]

    has_code_cells = len(code_cells) > 0

    return {
        "exists": True,
        "executed_code_cells": len(executed),
        "total_code_cells": len(code_cells),
        "all_code_cells_executed": has_code_cells and len(code_cells) == len(executed),
        "all_code_cells_have_output": has_code_cells and len(code_cells) == len(with_output),
    }


def file_status(path: Path) -> dict[str, object]:
    return {
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def main() -> None:
    args = parse_args()

    required_paths = {
        "README": PROJECT_ROOT / "README.md",
        "notebook": PROJECT_ROOT / args.notebook_path,
        "scripts_commit3": PROJECT_ROOT / "scripts/run_commit3_part1.py",
        "scripts_commit4": PROJECT_ROOT / "scripts/run_commit4_part2.py",
        "scripts_commit5": PROJECT_ROOT / "scripts/run_commit5_part2.py",
        "scripts_commit6": PROJECT_ROOT / "scripts/run_commit6_part3.py",
        "scripts_commit7": PROJECT_ROOT / "scripts/run_commit7_finalize.py",
        "word2vec_utils": PROJECT_ROOT / "src/utils/word2vec.py",
        "annotation_utils": PROJECT_ROOT / "src/utils/annotation.py",
        "sequence_models": PROJECT_ROOT / "src/models/sequence_models.py",
        "transformer_models": PROJECT_ROOT / "src/models/transformer_classifier.py",
    }

    artifact_paths = {
        "tfidf_matrix": PROJECT_ROOT / "embeddings/tfidf_matrix.npy",
        "ppmi_matrix": PROJECT_ROOT / "embeddings/ppmi_matrix.npy",
        "word2vec_embeddings": PROJECT_ROOT / "embeddings/embeddings_w2v.npy",
        "pos_model": PROJECT_ROOT / "models/bilstm_pos.pt",
        "ner_model": PROJECT_ROOT / "models/bilstm_ner.pt",
        "transformer_model": PROJECT_ROOT / "models/transformer_cls.pt",
        "pos_train": PROJECT_ROOT / "data/pos_train.conll",
        "pos_test": PROJECT_ROOT / "data/pos_test.conll",
        "ner_train": PROJECT_ROOT / "data/ner_train.conll",
        "ner_test": PROJECT_ROOT / "data/ner_test.conll",
        "report_pdf": PROJECT_ROOT / args.report_path,
    }

    notebook_status = notebook_execution_status(PROJECT_ROOT / args.notebook_path)

    checklist = {
        "required_files": {name: file_status(path) for name, path in required_paths.items()},
        "artifacts": {name: file_status(path) for name, path in artifact_paths.items()},
        "notebook_execution": notebook_status,
        "notes": [
            "If any artifact is missing, run the corresponding commit scripts in order.",
            "Notebook should be re-run end-to-end before final submission packaging.",
            "Ensure report.pdf is exported from your final write-up before zipping.",
        ],
    }

    out_path = PROJECT_ROOT / args.checklist_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(checklist, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved final checklist to: {out_path}")
    print("Summary of missing artifacts:")
    for key, status in checklist["artifacts"].items():
        if not status["exists"]:
            print(f" - missing: {key}")


if __name__ == "__main__":
    main()
