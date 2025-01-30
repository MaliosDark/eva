# eva_tools.py
# -------------------------------------------------------------
# Contains various tools or functions that Eva can use or expand.
# Eva may update this file automatically if she needs more tools.
# -------------------------------------------------------------

import os
import re
import json
import math
import time
import requests
from bs4 import BeautifulSoup

try:
    # If Selenium is available, we can do more advanced interaction
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


def sample_tool(data):
    """
    A simple example demonstrating a custom tool.
    """
    return f"Processed data: {data}"


def parse_json(json_string):
    """
    Parse a JSON string and return a Python dictionary.
    Returns None if parsing fails.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None


def calculate_expression(expression):
    """
    Safely evaluate a mathematical expression and return its result.
    e.g., '1+2*3' -> 7
    """
    try:
        # Strictly limit builtins for security reasons
        return eval(expression, {"__builtins__": {}}, {})
    except Exception:
        return None


def summarize_text(text, max_chars=200):
    """
    Returns a truncated summary of the given text (up to `max_chars`).
    """
    if len(text) <= max_chars:
        return text
    else:
        return text[:max_chars] + "..."


def html_title_extractor(html_content):
    """
    Extract the <title> from an HTML string, if present.
    Returns None if no title is found.
    """
    match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def meta_description_extractor(html_content):
    """
    Attempt to extract the <meta name="description" content="..."> from HTML.
    Returns None if not found.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    tag = soup.find("meta", attrs={"name": "description"})
    if tag and tag.get("content"):
        return tag["content"].strip()
    return None


def web_fetch(url, timeout=10):
    """
    Fetch the raw HTML from a URL using requests.
    Returns the text content or an error message.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error fetching {url}: {e}"


def google_search(query, limit=3):
    """
    Basic Google search using requests, returning up to `limit` result titles.
    Note: Google's HTML structure changes often, so this can break.
    """
    # Warning: scraping Google search results can violate TOS in some contexts.
    # Use an official API or a library if possible.
    if not query.strip():
        return ["No valid query."]
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = [g.get_text() for g in soup.find_all('h3')]
        return results[:limit] if results else ["No real-time data found."]
    except Exception as e:
        return [f"❌ Google Search Error: {e}"]


# ==============================
# ADVANCED: SELENIUM-BASED TOOLS
# ==============================
def selenium_fetch(url, accept_cookies=True, timeout=15, headless=True):
    """
    Use Selenium to fetch a webpage's HTML, optionally interacting with
    cookie banners.
    - `accept_cookies=True` tries to find and accept the cookie banner.
    - `headless=True` runs in headless mode. If set to False, a browser window appears.
    Requires:
      pip install selenium webdriver-manager
    Returns the page HTML or error message.
    """
    if not SELENIUM_AVAILABLE:
        return "Selenium not installed or import failed. Cannot fetch with Selenium."

    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.options import Options

    try:
        # Setup Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")

        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")

        # Install and initialize driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(timeout)

        driver.get(url)

        if accept_cookies:
            # Attempt to detect a common cookie banner
            # This is heuristic-based and may fail on some sites
            try:
                # Example: searching for a button with text matching 'accept' or 'agree'
                WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//button[contains(translate(text(),'ACCEING','acceing'), 'accept') or "
                                   "contains(translate(text(),'AGREING','agreing'), 'agree')]")
                    )
                ).click()
            except (TimeoutException, NoSuchElementException):
                # No typical cookie banner found or failed
                pass

        # Wait a bit for any dynamic content
        time.sleep(2)
        page_html = driver.page_source
        driver.quit()
        return page_html
    except WebDriverException as e:
        return f"Error in selenium_fetch: {e}"


def detect_and_handle_cookie_banner(driver, accept=True, cookie_button_xpath=None):
    """
    If there's a known cookie banner, try to accept or reject cookies.
    - `accept=True` => tries to accept them. Otherwise rejects if found.
    - `cookie_button_xpath`: An optional custom XPath to the accept or reject button.
    If not provided, tries a generic approach.
    """
    try:
        if cookie_button_xpath:
            # user-provided XPath
            button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, cookie_button_xpath)))
            button.click()
            return "Cookie banner clicked via user-provided XPath."
        else:
            # generic approach: search for button with 'accept' or 'agree' or 'reject'
            # This is very heuristic and might fail.
            if accept:
                button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//button[contains(translate(text(),'ACCEING','acceing'), 'accept') or "
                                   "contains(translate(text(),'AGREING','agreing'), 'agree')]")
                    )
                )
                button.click()
                return "Cookie banner accepted (heuristic)."
            else:
                # Try 'reject'
                button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//button[contains(translate(text(),'REJECT','reject'), 'reject')]")
                    )
                )
                button.click()
                return "Cookie banner rejected (heuristic)."
    except (TimeoutException, NoSuchElementException):
        return "No cookie banner found or not clickable."

