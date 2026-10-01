import json
import os
import random
import re
import time
from datetime import datetime
import pandas as pd
from playwright.sync_api import sync_playwright

USER_DATA_DIR = "./user_data"
EXCEL_INPUT_FILE = "contacts.xlsx"

# Refined stable selectors

SEARCH_BOX_SELECTOR = 'input[placeholder="Search or start a new chat"], div[contenteditable="true"][data-tab="3"]'
MSG_BOX_SELECTOR = 'footer div[contenteditable="true"], div[data-testid="conversation-compose-box-input"]'
#MESSAGE_BUBBLE_SELECTOR = '[data-pre-plain-text]'
CHAT_HEADER_SELECTOR = '#main header'
# Store selector in a variable

def sanitize_phone(phone_str: str) -> str:
    if not phone_str: return ""
    return re.sub(r"[^\d+]", "", str(phone_str))

def load_contacts_file(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f"[ERROR] Input file '{file_path}' not exist!")
        return None
    try:
        df = pd.read_excel(file_path)
        print(f"[SUCCESS] Loaded {len(df)} contacts.")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read excel: {e}")
        return None

def smart_data_read_last3_msg(page, contact_name: str) -> list:
    """Extracts last 3 messages from the current active chat."""
    last_3_messages = []
    
    try:
        MESSAGE_BUBBLE_SELECTOR = f'#main div.copyable-text[data-pre-plain-text*="{contact_name}"]'
        # Wait for messages to load
        page.wait_for_selector(MESSAGE_BUBBLE_SELECTOR, timeout=5000)
        
        # Get all message elements
        all_msgs = page.query_selector_all(MESSAGE_BUBBLE_SELECTOR)
        
        # Slice the last 3
        count = len(all_msgs)
        start_index = max(0, count - 3)
        
        for i in range(start_index, count):
            el = all_msgs[i]
            # WhatsApp stores sender/time in data-pre-plain-text
            meta = el.get_attribute('data-pre-plain-text')
            # The actual text is usually inside a selectable span
            text_node = el.query_selector('span[data-testid="selectable-text"] span')
            content = text_node.inner_text() if text_node else el.inner_text()
            
            last_3_messages.append({
                "meta": meta.strip(" []:") if meta else "Unknown",
                "text": content.strip()
            })

        print(f"[INFO] Extracted {len(last_3_messages)} messages.")
    except Exception as e:
        print(f"[WARNING] Extraction failed: {e}")
    return last_3_messages


def send_whatsapp_message(page, name: str, phone: str, message: str) -> tuple:
    screenshot_path = ""
    extracted_messages = []

    # 1. Search
    search_box = page.wait_for_selector(SEARCH_BOX_SELECTOR, state="visible",timeout=15000)
    search_box.click()
    # Clear and fill
    page.keyboard.press("Control+A")
    page.keyboard.press("Backspace")
    search_box.fill(name)
    #phone if phone else name
    page.wait_for_timeout(2000)

    # 2. Select Chat
    page.keyboard.press("Enter")
    page.wait_for_timeout(1000)

               # 2. Check Result & Open
    if page.query_selector('text="No chats, contacts or messages found"'):
        print(f"[WARNING] Contact {name} not found.")
        result = "Not Found"
        return result

    page.wait_for_selector(CHAT_HEADER_SELECTOR, timeout=8000)
    # 3. Send Message
    msg_box = page.wait_for_selector(MSG_BOX_SELECTOR, timeout=5000)
    msg_box.fill(message)
    page.keyboard.press("Enter")
    
    page.wait_for_timeout(random.randint(2000, 4000))

    # 4. Screenshot & Extract
    screenshot_path = f"screenshot_{name.replace(' ', '_')}_{int(time.time())}.png"
    page.screenshot(path=screenshot_path)
    
    extracted_messages = smart_data_read_last3_msg(page, name)
    return "Sent", screenshot_path, extracted_messages

def save_reports(report_details: list, summary_list: list):
    today = datetime.now().strftime("%Y-%m-%d")
    with open(f"report_{today}.json", "w", encoding="utf-8") as f:
        json.dump(report_details, f, indent=4, ensure_ascii=False)
    pd.DataFrame(summary_list).to_excel(f"summary_{today}.xlsx", index=False)
    print("\n[REPORT] Saved JSON and Excel reports.")

def run_whatsapp_bot():
    df_contacts = load_contacts_file(EXCEL_INPUT_FILE)
    if df_contacts is None or df_contacts.empty: return

    report_details, summary_list = [], []

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR, headless=False, args=["--start-maximized"]
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto("https://web.whatsapp.com")

        print("Waiting for login...")
        page.wait_for_selector(SEARCH_BOX_SELECTOR, timeout=9000)

        for index, row in df_contacts.iterrows():
            name = str(row.get("Name", "")).strip()
            phone = sanitize_phone(str(row.get("Phone", "")))
            msg = str(row.get("Message", "Hello {name}")).replace("{name}", name)

            try:
                status, ss, msgs = send_whatsapp_message(page, name, phone, msg)
            except Exception as e:
                status, ss, msgs = f"Error: {e}", "", []

            report_details.append({"name": name, "status": status, "last_3_messages": msgs})
            summary_list.append({"Name": name, "Phone": phone, "Status": status})
            page.wait_for_timeout(random.randint(3000, 5000))

        context.close()
    save_reports(report_details, summary_list)

if __name__ == "__main__":
    run_whatsapp_bot()