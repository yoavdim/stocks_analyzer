#!/usr/bin/env python3
"""
Fetch historical NAV data for Israeli mutual funds from TASE Maya.

Opens a headful Playwright browser to bypass Cloudflare, navigates through
the TASE site (main -> fund page -> history), and captures the API responses.

The TASE API returns fund prices in ILA (agorot, 1/100 ILS).

Usage:
    python tase_fund_fetcher.py 5118872          # single fund
    python tase_fund_fetcher.py 5118872 5137047  # multiple funds
    python tase_fund_fetcher.py --list           # show cached funds
    python tase_fund_fetcher.py --price 5118872  # just print latest price (from cache)
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

CACHE_DIR = Path(__file__).parent / ".tase_cache"
FUND_PAGE_URL = "https://maya.tase.co.il/en/funds/mutual-funds/{fund_id}"
HISTORY_URL = "https://maya.tase.co.il/en/funds/mutual-funds/{fund_id}/historical-data?period=4&fromDate={from_date}T04:00:00.000Z&toDate={to_date}T04:00:00.000Z"

# Known API endpoint pattern (discovered via network capture)
HISTORY_API = "api/v1/funds/mutual/{fund_id}/history"


def load_cache(fund_id: str) -> dict | None:
    cache_file = CACHE_DIR / f"{fund_id}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None


def save_cache(fund_id: str, data: dict):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{fund_id}.json"
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def get_latest_price(fund_id: str) -> float | None:
    """Get latest NAV price in ILA (agorot) from cache."""
    cached = load_cache(fund_id)
    if not cached or not cached.get("history"):
        return None
    latest = cached["history"][0]  # sorted most recent first
    return latest.get("price")


def get_price_series(fund_id: str) -> list[dict] | None:
    """Get full price history from cache. Each entry: {date, price} in ILA."""
    cached = load_cache(fund_id)
    if not cached:
        return None
    return cached.get("history", [])


def _parse_history_response(data: list) -> list[dict]:
    """Parse the TASE mutual fund history API response into normalized price points."""
    prices = []
    for item in data:
        if not isinstance(item, dict):
            continue
        trade_date = item.get("tradeDate")
        # purchasePrice is the NAV in ILA (agorot)
        price = item.get("purchasePrice") or item.get("sellPrice")
        if trade_date and price:
            # Normalize date format (remove time portion)
            date_str = trade_date.split("T")[0] if "T" in str(trade_date) else str(trade_date)
            prices.append({"date": date_str, "price": float(price)})
    return prices


def _try_csv_download(page, fund_id: str) -> list[dict]:
    """Try to click the CSV/Excel download button and parse the result."""
    import csv
    import tempfile

    try:
        # Look for download/export button - common selectors on TASE
        download_selectors = [
            'button:has-text("Excel")',
            'button:has-text("CSV")',
            'button:has-text("Download")',
            'button:has-text("ייצוא")',
            'button:has-text("הורדה")',
            '[class*="download"]',
            '[class*="export"]',
            'a:has-text("Excel")',
            'a:has-text("CSV")',
        ]

        download_btn = None
        for selector in download_selectors:
            try:
                el = page.query_selector(selector)
                if el and el.is_visible():
                    download_btn = el
                    print(f"  Found download button: {selector}", file=sys.stderr)
                    break
            except:
                continue

        if not download_btn:
            print(f"  No download button found.", file=sys.stderr)
            return []

        # Trigger download and capture the file
        with page.expect_download(timeout=15000) as download_info:
            download_btn.click()
        download = download_info.value

        # Save to temp file and parse
        tmp_path = Path(tempfile.mktemp(suffix=download.suggested_filename or ".csv"))
        download.save_as(str(tmp_path))
        print(f"  Downloaded: {tmp_path.name} ({tmp_path.stat().st_size} bytes)", file=sys.stderr)

        # Parse CSV/Excel
        prices = _parse_downloaded_file(tmp_path)
        tmp_path.unlink(missing_ok=True)
        return prices

    except Exception as e:
        print(f"  CSV download failed: {e}", file=sys.stderr)
        return []


def _parse_downloaded_file(path: Path) -> list[dict]:
    """Parse a downloaded CSV/Excel file into price entries."""
    import csv

    prices = []
    content = path.read_text(encoding='utf-8-sig', errors='replace')
    lines = content.strip().split('\n')

    if not lines:
        return []

    # Try to detect delimiter
    delimiter = ',' if ',' in lines[0] else '\t'
    reader = csv.reader(lines, delimiter=delimiter)
    header = next(reader, None)
    if not header:
        return []

    # Find date and price columns (handle Hebrew and English headers)
    date_col = None
    price_col = None
    header_lower = [h.strip().lower() for h in header]

    for i, h in enumerate(header_lower):
        if any(kw in h for kw in ['date', 'תאריך', 'trade']):
            date_col = i
        elif any(kw in h for kw in ['purchase', 'price', 'קניה', 'שער', 'nav', 'מחיר']):
            price_col = i

    # If we couldn't identify columns, try first two numeric-looking columns
    if date_col is None or price_col is None:
        print(f"  CSV headers: {header}", file=sys.stderr)
        # Assume first col is date, second is price
        date_col = date_col or 0
        price_col = price_col or 1

    for row in reader:
        if len(row) <= max(date_col, price_col):
            continue
        try:
            date_str = row[date_col].strip()
            price_str = row[price_col].strip().replace(',', '')
            # Normalize date formats
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    date_str = dt.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            price = float(price_str)
            if price > 0:
                prices.append({"date": date_str, "price": price})
        except (ValueError, IndexError):
            continue

    return prices


def fetch_fund_history(fund_ids: list[str], headless=False, years=10):
    results = {}
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
            ],
        )

        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='Asia/Jerusalem',
        )

        page = context.new_page()

        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        captured_history = {}  # fund_id -> list of price entries
        captured_downloads = {}  # fund_id -> file path

        def handle_response(response):
            url = response.url
            if response.status != 200:
                return
            content_type = response.headers.get("content-type", "")
            if "tase.co.il" not in url:
                return
            if "json" in content_type:
                try:
                    data = response.json()
                except:
                    return
                for fid in fund_ids:
                    if f"funds/mutual/{fid}/history" in url:
                        if isinstance(data, list):
                            captured_history[fid] = data
                            print(f"  Captured history JSON for {fid}: {len(data)} entries", file=sys.stderr)
                        return

        page.on("response", handle_response)

        # Navigate to main page to establish session/cookies
        print("Loading TASE main page...", file=sys.stderr)
        try:
            page.goto("https://maya.tase.co.il/en", wait_until='domcontentloaded', timeout=30000)
            page.wait_for_timeout(4000)
            print(f"  Session established.", file=sys.stderr)
        except PlaywrightTimeout:
            print("  Timeout on main page, continuing...", file=sys.stderr)

        for fund_id in fund_ids:
            print(f"\nFetching fund {fund_id}...", file=sys.stderr)

            try:
                # Navigate to fund page first (establishes fund context)
                fund_url = FUND_PAGE_URL.format(fund_id=fund_id)
                print(f"  Loading fund page...", file=sys.stderr)
                page.goto(fund_url, wait_until='domcontentloaded', timeout=30000)
                page.wait_for_timeout(4000)

                # Navigate to historical data page with full date range
                history_url = HISTORY_URL.format(fund_id=fund_id, from_date=from_date, to_date=to_date)
                print(f"  Loading historical data ({from_date} to {to_date})...", file=sys.stderr)
                page.goto(history_url, wait_until='domcontentloaded', timeout=30000)
                page.wait_for_timeout(5000)

                # Wait a bit more if we haven't captured yet
                if fund_id not in captured_history:
                    print(f"  Waiting for API response...", file=sys.stderr)
                    page.wait_for_timeout(8000)

                prices = []

                if fund_id in captured_history:
                    raw_history = captured_history[fund_id]
                    prices = _parse_history_response(raw_history)
                    print(f"  JSON response: {len(prices)} price points", file=sys.stderr)

                # If JSON gave us limited data, try CSV download
                if len(prices) <= 20:
                    print(f"  JSON was limited ({len(prices)} pts), trying CSV download...", file=sys.stderr)
                    csv_prices = _try_csv_download(page, fund_id)
                    if csv_prices and len(csv_prices) > len(prices):
                        prices = csv_prices
                        print(f"  CSV download: {len(prices)} price points", file=sys.stderr)

                if prices:
                    result = {
                        "fund_id": fund_id,
                        "fetched_at": datetime.now().isoformat(),
                        "history": prices,
                    }
                    results[fund_id] = result
                    save_cache(fund_id, result)
                    print(f"  Got {len(prices)} total price points (ILA)", file=sys.stderr)
                    print(f"  Latest: {prices[0]['date']} = {prices[0]['price']} ILA (₪{prices[0]['price']/100:.4f})", file=sys.stderr)
                    print(f"  Oldest: {prices[-1]['date']} = {prices[-1]['price']} ILA (₪{prices[-1]['price']/100:.4f})", file=sys.stderr)
                else:
                    print(f"  No price data obtained.", file=sys.stderr)
                    print(f"  Current URL: {page.url}", file=sys.stderr)
                    results[fund_id] = {"fund_id": fund_id, "error": "no_history_response"}

            except PlaywrightTimeout:
                print(f"  Timeout for fund {fund_id}", file=sys.stderr)
                results[fund_id] = {"fund_id": fund_id, "error": "timeout"}
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                results[fund_id] = {"fund_id": fund_id, "error": str(e)}

        browser.close()

    return results


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    if sys.argv[1] == "--list":
        CACHE_DIR.mkdir(exist_ok=True)
        for f in sorted(CACHE_DIR.glob("*.json")):
            data = json.loads(f.read_text())
            fund_id = f.stem
            fetched = data.get("fetched_at", "?")
            history = data.get("history", [])
            latest = history[0] if history else {}
            print(f"  {fund_id}: {len(history)} pts, latest={latest.get('price', '?')} ILA @ {latest.get('date', '?')}, fetched={fetched}")
        sys.exit(0)

    if sys.argv[1] == "--price":
        if len(sys.argv) < 3:
            print("Usage: --price <fund_id>", file=sys.stderr)
            sys.exit(1)
        fund_id = sys.argv[2]
        price = get_latest_price(fund_id)
        if price is not None:
            print(f"{price}")
        else:
            print(f"No cached price for {fund_id}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    fund_ids = sys.argv[1:]
    results = fetch_fund_history(fund_ids)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
