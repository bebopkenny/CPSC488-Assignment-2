# Playwrite is a headless browser that can execute JavaScript like selenium
# To install Playwright and the necessary browsers:
# pip install playwright
# playwright install
from playwright.sync_api import sync_playwright

# BeautifulSoup for parsing HTML
# To install BeautifulSoup:
# pip install beautifulsoup4
# pip install lxml # for faster parser
from bs4 import BeautifulSoup
import os 


url="https://www.benzinga.com/news/20/05/16095921/46-stocks-moving-in-fridays-mid-day-session"
#url="http://www.zacks.com/stock/news/858761/timing-the-market-is-it-possible-april-07-2020?cid=CS-BENZ-FT-retirement_ideas|market_timing_secrets-858761"
#url="https://seekingalpha.com/news/3521666-notable-earnings-monday-s-close?source=partner_benzinga"

#+++ GuruFocus article URL that still blocks simple requests
#url = "http://www.gurufocus.com/news/1086307/bill-ackmans-pershing-square-releases-letter-to-investors"

with sync_playwright() as p:
    # Launch Chromium in headless mode
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Set a User-Agent to mimic a real browser
    page.set_extra_http_headers({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    })

    # Go to the page
    page.goto(url, wait_until="domcontentloaded", timeout=30000)

    # Get the full page HTML
    html = page.content()

    # Save to a local file
    with open("article.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Page downloaded successfully")

    browser.close()

# Parse HTML with BeautifulSoup
# Path to your downloaded HTML file
html_file_path = "article.html"

with open(html_file_path, "r", encoding="utf-8") as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, "html.parser")

# Try multiple ways to find the article content
# 1. Using class contains 'article-body'
article_div = soup.find("div", class_=lambda x: x and "article-body" in x)

# 2. If not found, check if it's in <article> tag
if not article_div:
    article_div = soup.find("article")

# 3. If still not found, print all divs to inspect
if not article_div:
    print("Couldn't find article body automatically. Here are available div classes:")
    for div in soup.find_all("div"):
        print(div.get("class"))
else:
    article_text = article_div.get_text(separator="\n", strip=True)
    print(article_text)
