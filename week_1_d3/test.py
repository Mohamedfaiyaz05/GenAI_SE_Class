import datetime
import os
import sys
import time
import pyautogui

# ==============================================================================
# CONFIGURATION & FAIL-SAFE SETUP
# ==============================================================================
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

today_str = datetime.datetime.now().strftime("%Y-%m-%d")
excel_filename = f"daily_report_{today_str}.xlsx"
screenshot_filename = f"report_screenshot_{today_str}.png"

# Path to Chrome Executable (Standard 64-bit install path)
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def focus_or_launch_app(app_command: str):
    """Launches an application via Windows Run Dialog (Win + R)."""
    pyautogui.hotkey("win", "r")
    time.sleep(1.0)
    pyautogui.write(f'"{app_command}"', interval=0.01)
    pyautogui.press("enter")


def main():
    try:
        print("Starting PyAutoGUI Operations Bot...")
        print("Note: Move your mouse to any screen corner to trigger FailSafe.")
        time.sleep(2)

        # ----------------------------------------------------------------------
        # STEP 1: Open Chrome and Navigate to Website
        # ----------------------------------------------------------------------
        print("Step 1: Launching Chrome...")
        import datetime
import os
import sys
import time
import tkinter as tk
import pyautogui

# ==============================================================================
# CONFIGURATION & FAIL-SAFE SETUP
# ==============================================================================
# FailSafe: Move mouse to ANY of the 4 corners of the screen to abort execution immediately.
pyautogui.FAILSAFE = True

# Default pause between PyAutoGUI actions
pyautogui.PAUSE = 0.5

# Define Output Filenames
today_str = datetime.datetime.now().strftime("%Y-%m-%d")
excel_filename = f"daily_report_{today_str}.xlsx"
screenshot_filename = f"report_screenshot_{today_str}.png"


def focus_or_launch_app(app_command: str):
    """Launches an application or URL via Windows Run Dialog (Win + R)."""
    pyautogui.hotkey("win", "r")
    time.sleep(1.0)
    # Using 'start' ensures Windows resolves app paths like 'chrome' or 'excel' properly
    pyautogui.write(f"start {app_command}", interval=0.01)
    pyautogui.press("enter")


def get_clipboard_text() -> str:
    """Safely retrieves text content currently stored in system clipboard."""
    try:
        root = tk.Tk()
        root.withdraw()
        text = root.clipboard_get()
        root.destroy()
        return text
    except Exception:
        return ""


def main():
    try:
        print("Starting PyAutoGUI Operations Bot...")
        print("Note: Move your mouse to any screen corner to trigger FailSafe.")
        time.sleep(2)

        # ----------------------------------------------------------------------
        # STEP 1: Open Chrome & Navigate to Target URL
        # ----------------------------------------------------------------------
        target_url = "https://news.ycombinator.com/"
        print(f"Step 1: Launching Chrome and navigating to {target_url}...")

        # Launch Chrome directly with the target URL
        focus_or_launch_app(f"chrome {target_url}")

        # Wait for Chrome process and webpage to fully load
        time.sleep(5.0)

        # In case Chrome profile picker pops up, press Enter to pick default profile
        pyautogui.press("enter")
        time.sleep(2.0)

        # Click center of screen to guarantee focus on webpage
        screen_w, screen_h = pyautogui.size()
        pyautogui.click(x=screen_w // 2, y=screen_h // 2)
        time.sleep(0.5)

        # ----------------------------------------------------------------------
        # STEP 2: Highlight & Copy Information from Screen
        # ----------------------------------------------------------------------
        print("Step 2: Copying data from page...")

        # Select all page text and copy to clipboard
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.5)
        pyautogui.hotkey("ctrl", "c")
        time.sleep(0.5)

        # Deselect/click away
        pyautogui.click(x=100, y=100)

        # Clean/truncate clipboard content so Excel doesn't break across hundreds of cells
        copied_text = get_clipboard_text().replace("\n", " ").replace("\r", " ").replace("\t", " ")
        short_copied_data = copied_text[:150] if copied_text else "No content copied."

        # ----------------------------------------------------------------------
        # STEP 3: Open Microsoft Excel & Prepare Report
        # ----------------------------------------------------------------------
        print("Step 3: Launching Microsoft Excel...")
        focus_or_launch_app("excel")

        # Wait for Excel splash screen
        time.sleep(6.0)

        # Select 'Blank Workbook' on start screen
        pyautogui.press("enter")
        time.sleep(3.0)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        custom_comment = "Daily automated status check - Completed successfully."

        print("Writing data to Excel sheet...")

        # Cell A1: Today's Date & Time
        pyautogui.write(current_time, interval=0.03)
        pyautogui.press("tab")  # Move to Cell B1
        time.sleep(0.5)

        # Cell B1: Write cleaned sample data from page
        pyautogui.write(short_copied_data, interval=0.01)
        pyautogui.press("tab")  # Move to Cell C1
        time.sleep(0.5)

        # Cell C1: Short Executive Comment
        pyautogui.write(custom_comment, interval=0.03)
        pyautogui.press("enter")  # Complete row entry
        time.sleep(1.0)

        # ----------------------------------------------------------------------
        # STEP 4: Save the Excel File
        # ----------------------------------------------------------------------
        print(f"Step 4: Saving Excel file as '{excel_filename}'...")

        # Open 'Save As' Dialog (F12 shortcut in Excel)
        pyautogui.press("f12")
        time.sleep(2.5)

        # Type absolute path into Save dialog
        full_path = os.path.abspath(excel_filename)
        pyautogui.write(full_path, interval=0.02)
        time.sleep(0.5)
        pyautogui.press("enter")

        # Wait for file saving to complete
        time.sleep(3.0)

        # Confirm overwrite prompt if file already exists
        pyautogui.press("y")
        time.sleep(1.0)

        # ----------------------------------------------------------------------
        # STEP 5: Take Screenshot of Final Excel Sheet
        # ----------------------------------------------------------------------
        print("Step 5: Taking final screenshot...")
        screenshot_path = os.path.abspath(screenshot_filename)
        pyautogui.screenshot(screenshot_path)
        print(f"Screenshot successfully saved to: {screenshot_path}")

        print("\n=========================================")
        print("SUCCESS: Automation Bot Execution Complete!")
        print("=========================================")

    except pyautogui.FailSafeException:
        print("\n[ABORTED] FailSafe triggered by moving mouse to corner!")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() = "https://www.accuweather.com/en/in/chennai/206671/weather-forecast/206671"
        
        # Launching Chrome directly with the target URL in the command
        focus_or_launch_app(f"{CHROME_PATH} {target_url}")

        # Wait for Chrome process to open and load the webpage
        time.sleep(5.0)

        # If a profile picker pops up, pressing Enter selects default profile
        pyautogui.press("enter")
        time.sleep(5.0)

        # Ensure window is focused by clicking near the top center
        screen_width, screen_height = pyautogui.size()
        pyautogui.click(x=screen_width // 2, y=200)
        time.sleep(0.5)

        print(f"Successfully launched Chrome and navigated to {target_url}")

    except pyautogui.FailSafeException:
        print("\n[ABORTED] FailSafe triggered by user moving mouse to corner!")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()