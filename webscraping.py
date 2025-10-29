import requests # only works for static HTML pages, cannot execute JavaScript
from time import sleep

# Some URLs of the webpage tried
#url = "https://www.benzinga.com/news/20/06/16190091/stocks-that-hit-52-week-highs-on-friday"
#url = "http://www.zacks.com/stock/news/858761/timing-the-market-is-it-possible-april-07-2020?cid=CS-BENZ-FT-retirement_ideas|market_timing_secrets-858761"
url = "https://seekingalpha.com/news/3521666-notable-earnings-monday-s-close?source=partner_benzinga"

# gurufocus seems to block requests
#forbidden status code 403 url = "http://www.gurufocus.com/news/1129235/3-stocks-growing-their-earnings-fast"
# you may want to this with headless browser like selenium or playwright

# To mimic a real browser visit, set User-Agent and other headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}

# Send GET request
for i in range(3):  # Try up to 3 times
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 200:
        break
    else:
        print(f"Attempt {i+1} failed with status code {response.status_code}")
        sleep(2)  # Wait before retrying

# Check if request was successful
if response.status_code == 200:
    html_content = response.text
    with open("page.html", "w", encoding="utf-8") as f:
        f.write(html_content)    
    print("Success: HTML downloaded")
    print(response.text[:500]) # Print first 500 characters of the HTML
else:
    print(f"Failed to retrieve page. Status code: {response.status_code}")
