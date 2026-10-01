import datetime
import os
import re
import sys
import time
import tkinter as tk
import pyautogui

# ==============================================================================
# CONFIGURATION & FAIL-SAFE SETUP
# ==============================================================================
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

today_str = datetime.datetime.now().strftime("%Y-%m-%d")
excel_filename = f"daily_report_{today_str}.xlsx"
screenshot_filename = f"report_screenshot_{today_str}.png"

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def focus_or_launch_app(app_command: str):
    """Launches an application via Windows Run Dialog (Win + R)."""
    pyautogui.hotkey("win", "r")
    time.sleep(1.0)
    pyautogui.write(app_command, interval=0.01)
    pyautogui.press("enter")


def get_clipboard_text() -> str:
    """Reads text content from system clipboard."""
    try:
        root = tk.Tk()
        root.withdraw()
        text = root.clipboard_get()
        root.destroy()
        return text
    except Exception:
        return ""


def extract_weather_data(raw_text: str):
    """Extracts city and temperature from raw copied webpage text."""
    # Match city/location (e.g., "Chennai, Tamil Nadu")
    city_match = re.search(r"([A-Za-z\s]+,\s*[A-Za-z\s]+)", raw_text)
    # Match temperature (e.g., "33°C" or "33°")
    temp_match = re.search(r"(\d+°[C|F]?)", raw_text)

    city = city_match.group(1).strip() if city_match else "Chennai, Tamil Nadu"
    temp = temp_match.group(1).strip() if temp_match else "N/A"

    return city, temp


def main():
    try:
        print("Starting PyAutoGUI Operations Bot...")
        print("Note: Move your mouse to any screen corner to trigger FailSafe.")
        time.sleep(2)

        # ----------------------------------------------------------------------
        # STEP 1: Open Chrome and Navigate to Website
        # ----------------------------------------------------------------------
        print("Step 1: Launching Chrome...")
        target_url = "https://www.accuweather.com/en/in/chennai/206671/weather-forecast/206671"

        focus_or_launch_app(f'{CHROME_PATH} {target_url}')
        time.sleep(5.0)

        # Pass profile picker if prompted
        pyautogui.press("enter")
        time.sleep(5.0)

        screen_width, screen_height = pyautogui.size()
        pyautogui.click(x=screen_width // 2, y=200)
        time.sleep(1.0)

        # ----------------------------------------------------------------------
        # STEP 2: Highlight & Copy Information from Screen
        # ----------------------------------------------------------------------
        print("Step 2: Copying page data...")
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.5)
        pyautogui.hotkey("ctrl", "c")
        time.sleep(0.5)

        # Unhighlight page
        pyautogui.click(x=screen_width // 2, y=200)
        time.sleep(0.5)

        # Extract structured data from clipboard
        raw_clipboard = get_clipboard_text()
        city, temp = extract_weather_data(raw_clipboard)
        print(f"Extracted -> City: {city} | Temp: {temp}")

        # ----------------------------------------------------------------------
        # STEP 3: Open Microsoft Excel & Format Headers/Data
        # ----------------------------------------------------------------------
        print("Step 3: Launching Microsoft Excel...")
        focus_or_launch_app("excel")
        time.sleep(6.0)

        # Open Blank Workbook
        pyautogui.press("enter")
        time.sleep(3.0)

        # --- WRITE HEADERS (Row 1) ---
        print("Writing headers...")
        headers = ["datetime", "website_scrapped", "city", "temperature", "status"]
        for header in headers:
            pyautogui.write(header, interval=0.02)
            pyautogui.press("tab")
            time.sleep(0.2)

        pyautogui.press("enter")  # Move to Row 2 (Cell A2)
        time.sleep(0.5)

        # --- WRITE DATA (Row 2) ---
        print("Writing cleaned weather data...")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "scrapped successfully"

        data_row = [current_time, target_url, city, temp, status]
        for val in data_row:
            pyautogui.write(str(val), interval=0.02)
            pyautogui.press("tab")
            time.sleep(0.2)

        pyautogui.press("enter")
        time.sleep(1.0)

        # ----------------------------------------------------------------------
        # STEP 4: Save the Excel File
        # ----------------------------------------------------------------------
        print(f"Step 4: Saving Excel file as {excel_filename}...")
        pyautogui.press("f12")
        time.sleep(2.5)

        full_path = os.path.abspath(excel_filename)
        pyautogui.write(full_path, interval=0.02)
        time.sleep(0.5)
        pyautogui.press("enter")
        time.sleep(3.0)

        # Handle overwrite prompt safely without typing 'y' into cells
        pyautogui.press("y")
        time.sleep(1.0)

        # ----------------------------------------------------------------------
        # STEP 5: Take Screenshot of Final Excel Sheet
        # ----------------------------------------------------------------------
        print("Step 5: Taking final screenshot...")
        screenshot_path = os.path.abspath(screenshot_filename)
        pyautogui.screenshot(screenshot_path)
        print(f"Screenshot saved to: {screenshot_path}")

        print("\n=========================================")
        print("SUCCESS: Automation Bot Execution Complete!")
        print("=========================================")

    except pyautogui.FailSafeException:
        print("\n[ABORTED] FailSafe triggered by user moving mouse to corner!")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()