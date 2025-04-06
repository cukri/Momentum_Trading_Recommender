import argparse
import datetime
from data_fetch import load_tickers, get_all_tickers, get_fmp_data
from data_engineering import calculate_technical_indicators, save_data_to_csv
from recommendation_engine import main_recommendation_flow

def main():
    parser = argparse.ArgumentParser(description="Stock Data Fetching and Processing")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--tickers", nargs="+", help="List of tickers to fetch (space-separated). Jeśli nie podano, użyje pliku tickers.txt")
    parser.add_argument("--today", action="store_true", help="Fetch only today's data")
    parser.add_argument("--skip_fetch", action="store_true", help="Skip data fetching and use existing CSV")
    parser.add_argument("--recommend", action="store_true", help="Generate recommendations")
    parser.add_argument("--csv_path", type=str, default="processed_stocks_data.csv", help="Path to processed CSV file")
    parser.add_argument("--tickers_file", type=str, default="tickers.txt", help="Path to the file containing tickers")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file")

    args = parser.parse_args()

    if args.today:
        today = datetime.date.today().strftime("%Y-%m-%d")
        args.start_date = today
        args.end_date = today

    if not args.skip_fetch:
        if args.tickers:
            tickers = args.tickers
        else:
            tickers = load_tickers(args.tickers_file)
            if not tickers:
                # Jeśli plik nie zawiera tickerów, możesz użyć domyślnej listy z API
                tickers = get_all_tickers()

        all_data = []
        for ticker in tickers:
            ticker = ticker.strip()
            print(f"Getting data for {ticker} from {args.start_date} to {args.end_date}...")
            df = get_fmp_data(ticker, start=args.start_date, end=args.end_date)
            if df is not None:
                df = calculate_technical_indicators(df)
                all_data.append(df)
                print(f"Metrics for {ticker} downloaded and calculated.")
            else:
                print(f"No data for {ticker}.")
            print("-" * 40)

        if all_data:
            save_data_to_csv(all_data, filename=args.csv_path)
        else:
            print("No data to save.")

    if args.recommend:
        main_recommendation_flow(
            model_path=args.model_path,
            csv_path=args.csv_path
        )

if __name__ == "__main__":
    main()
