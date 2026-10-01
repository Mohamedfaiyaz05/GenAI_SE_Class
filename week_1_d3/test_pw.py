from playwright.sync_api import sync_playwright
from datetime import datetime

print("Starting the Playwright automation script...")
print(f"Script started at: {datetime.now()}")

#Daily Wather Report Bot
#chromium --> weather site --> Extract the report --> Screenshot --> final text file
with sync_playwright() as p:
    print("Launching the Chromium browser...")
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    print("Navigating to the AccuWeather website...")
    page.goto("https://www.accuweather.com/en/in/chennai/206671/weather-forecast/206671")
    page.wait_for_load_state("networkidle")
    page.screenshot(path="accuweather.png")

    print("Extracting the weather report...")

    # Extract the temperature text (e.g., "33°C")
    temp = page.locator('div.current-weather-info div.display-temp').inner_text()

    # Extract the weather status phrase (e.g., "Mostly cloudy")
    phrase = page.locator('div.current-weather div.phrase').inner_text()

    print(f"Temperature: {temp}")
    print(f"Status: {phrase}")


    weather_report = page.locator('div.current_weather_card').inner_text()
    print(f"Current Weather Report: {weather_report}")

    print("Saving the weather report to a text file...")
    with open("weather_report.txt", "w") as file:
        file.write(f"Weather Report for Chennai on {datetime.now()}:\n")
        file.write(weather_report)

    print("Closing the browser...")
    browser.close()
print(f"Script completed at: {datetime.now()}")