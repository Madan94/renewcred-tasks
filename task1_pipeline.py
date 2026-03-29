from src.ingestion.pipeline import parse_ev_payload
from src.quality.report import generate_report
from src.eda.visualizations import generate_all_charts

def main():
    df = parse_ev_payload("data/raw/ev_prod_data.csv")

    df.to_parquet("data/processed/clean_ev_prod_data.parquet", index=False)

    generate_report(df)
    generate_all_charts(df)

    print("Task 1 Completed Successfully")


if __name__ == "__main__":
    main()