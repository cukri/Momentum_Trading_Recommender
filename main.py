import argparse
import datetime
from data_fetch import get_all_tickers, get_fmp_data, calculate_technical_indicators, save_data_to_csv

def main():
    parser = argparse.ArgumentParser(description="Stock Data Fetching and Processing")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--tickers", nargs="+", help="List of tickers to fetch (space-separated)")
    parser.add_argument("--today", action="store_true", help="Fetch only today's data")

    args = parser.parse_args()

    # Pobieranie dzisiejszej daty, jeśli podano --today
    if args.today:
        today = datetime.date.today().strftime("%Y-%m-%d")
        args.start_date = today
        args.end_date = today

    # Jeśli brak tickerów, pobierz domyślną listę
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

if __name__ == "__main__":
    main()
