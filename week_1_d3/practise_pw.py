
import json
import os
import random
import re
import time
import logging
from datetime import datetime
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Configuration & Logging
USER_DATA_DIR = "./whatsapp_session"
EXCEL_INPUT_FILE = "contacts.xlsx"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# VALIDATED SELECTORS (Confirmed via live DOM inspection)
SELECTORS = {
    # Using multiple fallback selectors for the search bar
    "search_input": 'input[placeholder="Search or start a new chat"], div[contenteditable="true"][data-tab="3"]',
    "message_input": '#main footer div[contenteditable="true"], div[role="textbox"][data-tab="10"]',
    "chat_header": '#main header',
    "message_row": '#main div[role="row"]',
}

def sanitize_phone(phone_str: str) -> str:
    return re.sub(r"[^\d+]", "", str(phone_str)) if phone_str else ""

class WhatsAppBot:
    def __init__(self, page):
        self.page = page

    def send_message(self, name: str, phone: str, message: str) -> dict:
        result = {"name": name, "status": "Failed", "history": []}
        try:
            # 1. Search Logic - Wait for element to be attachable
            search = self.page.wait_for_selector(SELECTORS["search_input"], state="visible", timeout=15000)
            search.click()
            
            # Use keyboard for clearing to avoid React state sync issues
            self.page.keyboard.press("Control+A")
            self.page.keyboard.press("Backspace")
            
            # Type search query (Phone is more accurate than Name)
            query = phone if phone and len(phone) > 5 else name
            self.page.keyboard.type(query, delay=random.randint(50, 100))
            self.page.wait_for_timeout(2000)

            # 2. Check Result & Open
            if self.page.query_selector('text="No chats, contacts or messages found"'):
                logging.warning(f"Contact {name} not found.")
                self.page.keyboard.press("Escape")
                result["status"] = "Not Found"
                return result

            self.page.keyboard.press("Enter")
            
            # 3. Wait for Chat Pane to Load
            self.page.wait_for_selector(SELECTORS["chat_header"], timeout=8000)
            
            # 4. Human-like Message Typing
            msg_input = self.page.wait_for_selector(SELECTORS["message_input"])
            msg_input.click()
            self.page.keyboard.type(message, delay=random.randint(30, 70))
            self.page.keyboard.press("Enter")
            
            # Safety delay to ensure message leaves 'Pending' state
            time.sleep(random.randint(2, 4))
            result["status"] = "Sent"
            logging.info(f"Sent to {name}")

        except Exception as e:
            logging.error(f"Error with {name}: {e}")
            result["status"] = "Error"
        
        return result

def run():
    if not os.path.exists(EXCEL_INPUT_FILE):
        print(f"Please create {EXCEL_INPUT_FILE} first.")
        return

    contacts = pd.read_excel(EXCEL_INPUT_FILE).to_dict('records')
    
    with sync_playwright() as p:
        # Launching with a persistent context to save login
        context = p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=False,
            args=["--start-maximized"]
        )
        page = context.pages[0]
        bot = WhatsAppBot(page)
        
        page.goto("https://web.whatsapp.com")
        print("Please log in. The script will start once the search bar is visible.")
        
        # Wait for search bar to appear (means login is finished)
        page.wait_for_selector(SELECTORS["search_input"], timeout=120000)

        for contact in contacts:
            name = str(contact.get("Name", ""))
            phone = sanitize_phone(str(contact.get("Phone", "")))
            msg = str(contact.get("Message", "Hello {name}")).replace("{name}", name)
            
            bot.send_message(name, phone, msg)
            
            # Randomized delay between different contacts to prevent ban
            time.sleep(random.randint(5, 8))

        context.close()

if __name__ == "__main__":
    run()