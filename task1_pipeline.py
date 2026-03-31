import logging

from src.ingestion.pipeline import parse_ev_payload
from src.quality.report import generate_report
from src.eda.visualizations import generate_all_charts
from pathlib import Path

def main():
    logging.basicConfig(level=logging.INFO)
    df = parse_ev_payload("data/raw/ev_prod_data.csv")

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(processed_dir / "clean_ev_prod_data.csv", index=False)

    try:
        df.to_parquet(processed_dir / "clean_ev_prod_data.parquet", index=False)
    except Exception as e:
        logging.warning("Parquet export skipped: %s", e)

    generate_report(df, out_dir=Path("outputs/task1/reports"))
    generate_all_charts(df, output_dir=Path("outputs/task1/eda_charts"))

    print("Task 1 Completed Successfully")


if __name__ == "__main__":
    main()