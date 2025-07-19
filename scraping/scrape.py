import time
import os
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# List of loan URLs from Bank of Maharashtra
urls = [
    "https://bankofmaharashtra.in/personal-banking/loans/home-loan",
    "https://bankofmaharashtra.in/personal-banking/loans/car-loan",
    "https://bankofmaharashtra.in/personal-banking/loans/education-loan",
    "https://bankofmaharashtra.in/personal-banking/loans/personal-loan",
]

def setup_driver():
    """Configure and return a Chrome WebDriver"""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver

def scrape_bom_page(url):
    """Scrape a single page with enhanced content extraction"""
    driver = setup_driver()
    try:
        driver.get(url)
        
        # Wait for dynamic content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Scroll to trigger dynamic content loading
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        
        # Try multiple selectors to find main content
        selectors = [
            "div.main-content",  # Common content container
            "div.content-area",
            "div.page-content",
            "div.col-md-9",
            "main",
            "article",
            "div#content",
            "body"  # Fallback to entire body
        ]
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Try each selector until we find content
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                text = ' '.join([elem.get_text(separator=" ", strip=True) for elem in elements])
                text = ' '.join(text.split())  # Normalize whitespace
                if len(text) > 100:  # Only return if we got substantial content
                    return text
        
        # If all selectors failed, return the whole page text
        return soup.get_text(separator=" ", strip=True)
        
    except Exception as e:
        print(f"⚠️ Error scraping {url}: {str(e)}")
        return "Scraping failed"
    finally:
        driver.quit()

# Main scraping loop
data = []
total_urls = len(urls)

print(f"🚀 Starting to scrape {total_urls} pages...")
for i, url in enumerate(urls, 1):
    print(f"\n🔍 [{i}/{total_urls}] Scraping: {url}")
    start_time = time.time()
    
    text = scrape_bom_page(url)
    data.append({"url": url, "content": text})
    
    elapsed = time.time() - start_time
    print(f"✅ Completed in {elapsed:.2f}s | Content length: {len(text)} characters")

# Save output
os.makedirs("data", exist_ok=True)
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
output_file = f"data/raw_data_{timestamp}.csv"

df = pd.DataFrame(data)
df.to_csv(output_file, index=False)
print(f"\n🎉 Scraping complete! Data saved to → {output_file}")
print(f"Total pages scraped: {len(data)}")
print(f"Total content collected: {df['content'].str.len().sum() / 1024:.2f} KB")