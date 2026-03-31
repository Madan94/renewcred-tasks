from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_path("task1_pipeline.py", run_name="__main__")
    runpy.run_path("task2_soc_model.py", run_name="__main__")
    runpy.run_path("task3_gps.py", run_name="__main__")


if __name__ == "__main__":
    main()

