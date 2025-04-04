import argparse
import datetime
from data_fetch import get_all_tickers, get_fmp_data, calculate_technical_indicators, save_data_to_csv
from recommendation_engine import main_recommendation_flow

def main():
    parser = argparse.ArgumentParser(description="Stock Data Fetching and Processing")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--tickers", nargs="+", help="List of tickers to fetch (space-separated)")
    parser.add_argument("--today", action="store_true", help="Fetch only today's data")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetching and use existing CSV")
    parser.add_argument("--recommend", action="store_true", help="Generate recommendations")
    parser.add_argument("--csv_path", type=str, default="processed_stocks_data.csv", help="Path to processed CSV file")

    args = parser.parse_args()

    if args.today:
        today = datetime.date.today().strftime("%Y-%m-%d")
        args.start_date = today
        args.end_date = today

    if not args.skip_fetch:
        tickers = args.tickers if args.tickers else get_all_tickers()

        all_data = []
        for ticker in tickers:
            print(f"Getting data for {ticker} from {args.start_date} to {args.end_date}...")
            df = get_fmp_data(ticker, start=args.start_date, end=args.end_date)
            if df is not None:
                df = calculate_technical_indicators(df)
                all_data.append(df)
                print(f"Metrics for {ticker} downloaded and calculated.")
            else:
                print(f"No data for {ticker}.")
            print("-")

        if all_data:
            save_data_to_csv(all_data)
        else:
            print("No data to save.")

    if args.recommend:
        main_recommendation_flow(
            model_path=args.model_path,
            csv_path=args.csv_path
        )

if __name__ == "__main__":
    main()
