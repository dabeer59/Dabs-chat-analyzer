# streamlit_app.py
import streamlit as st
import pandas as pd
import time
import re
import os
import traceback
import json

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

import google.generativeai as genai

# ------------------- FILE PATHS -------------------
OUTPUT_FILE = "digiconn_closed_chats_all_sorted.csv"
RELEVANT_OUTPUT_FILE = "digiconn_relevant_chats.csv"
HANDOVER_FILE = "chat_handover_analysis_gemini.csv"
ESCALATION_FILE = "chat_escalation_gap_analysis.csv"

# ------------------- API CONFIG -------------------
# Step 3: single key
HANDOVER_API_KEY = "AIzaSyD6y_n_u3WvPrk03artyngKiIAxb6FN5k8"
# Step 4: list of keys with retry/rotation
ESCALATION_API_KEYS = [
    "AIzaSyD6y_n_u3WvPrk03artyngKiIAxb6FN5k8",
    "AIzaSyCwB7q4L70JA5m7nd99gmRyEj56kXXC_d4"
]
api_index = 0

# ------------------- DRIVER -------------------
def setup_driver(headless=True):
    options = Options()
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--remote-allow-origins=*")
    options.add_argument("--window-size=1920,1080")
    if headless:
        options.add_argument("--headless=new")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# ------------------- LOGIN -------------------
def login(driver, wait, username, password):
    driver.get("https://www.digiconn.co/portal/")
    wait.until(EC.element_to_be_clickable((By.ID, "username"))).send_keys(username)
    wait.until(EC.presence_of_element_located((By.ID, "password"))).send_keys(password + "\n")
    time.sleep(4)

# ------------------- NAVIGATE -------------------
def go_to_closed_chats(driver, wait):
    driver.get("https://www.digiconn.co/portal/chatmanager/index")
    time.sleep(3)
    closed_tab = wait.until(EC.element_to_be_clickable((By.ID, "closedLi")))
    driver.execute_script("arguments[0].click();", closed_tab)
    time.sleep(2)
    return closed_tab

# ------------------- EXTRACT -------------------
def extract_messages_from_chat(driver, chat_params):
    name, whatsapp_id, display_name, token, date_str = chat_params
    customer_number = whatsapp_id.split('@')[0]
    js_call = f'GetClosedMessages("{name}", "{whatsapp_id}", "{display_name}", "{token}", "{date_str}");'
    driver.execute_script(js_call)
    time.sleep(3)

    messages = driver.find_elements(By.CSS_SELECTOR, "span[id^='txt#']")
    chat_data = []

    for msg in messages:
        try:
            text = msg.text.strip()
            chat_id = msg.get_attribute("id").split("#")[-1]

            timestamp_elem = msg.find_elements(By.XPATH, "./preceding::small[1]")
            timestamp = timestamp_elem[0].text.strip() if timestamp_elem else ""

            sender_elem = msg.find_elements(By.XPATH, "./preceding::div[contains(@class,'avatar')][1]")
            sender = sender_elem[0].get_attribute("title").strip() if sender_elem else ""

            chat_data.append({
                "Customer Number": customer_number,
                "Sender": sender,
                "Timestamp": timestamp,
                "Message": text,
                "Chat ID": chat_id
            })
        except Exception:
            continue
    return chat_data

# ------------------- SCRAPER -------------------
def run_scraper(username, password, start_index=1, max_chats=0, headless=True):
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    driver = setup_driver(headless=headless)
    wait = WebDriverWait(driver, 20)
    all_messages = []

    try:
        login(driver, wait, username, password)
        closed_tab = go_to_closed_chats(driver, wait)

        chat_items = wait.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "li[id][onclick^='GetClosedMessages']")))
        total_chats = len(chat_items)

        if not max_chats or max_chats == 0:
            max_chats = total_chats

        scraped_chats = 0
        idx = start_index - 1

        progress = st.progress(0)

        while idx < total_chats and scraped_chats < max_chats:
            try:
                chat_items = wait.until(EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "li[id][onclick^='GetClosedMessages']")))
                if idx >= len(chat_items):
                    break

                chat_li = chat_items[idx]
                onclick_attr = chat_li.get_attribute("onclick")
                params = re.findall(r'"(.*?)"', onclick_attr)
                if len(params) != 5:
                    idx += 1
                    continue

                messages = extract_messages_from_chat(driver, params)
                if messages:
                    all_messages.extend(messages)
                    scraped_chats += 1

                driver.execute_script("arguments[0].click();", closed_tab)
                time.sleep(1)

                idx += 1
                progress.progress(scraped_chats / max_chats)
            except Exception:
                idx += 1
                continue

        df_out = pd.DataFrame(all_messages)
        if not df_out.empty:
            df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        else:
            df_out = pd.DataFrame(columns=["Customer Number", "Sender", "Timestamp", "Message", "Chat ID"])
            df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    finally:
        driver.quit()

    return OUTPUT_FILE

# ------------------- FILTER -------------------
def filter_relevant_chats(input_file):
    df = pd.read_csv(input_file)
    df["Sender"] = df["Sender"].astype(str).str.strip()
    df["Message"] = df["Message"].astype(str).str.strip()

    keywords = ["agent", "representative", "human", "talk to person", "talk to an person"]
    customer_requested_agent = (
        (df["Sender"].str.lower() != "outgoingapi") &
        df["Message"].str.lower().apply(lambda x: any(k in x for k in keywords))
    )
    bot_transfer_message = (
        (df["Sender"].str.lower() == "outgoingapi") &
        df["Message"].str.startswith("It seems like I need some extra help")
    )
    df["Relevant"] = customer_requested_agent | bot_transfer_message
    relevant_chats = df.groupby("Customer Number").filter(lambda x: x["Relevant"].any())
    relevant_chats.to_csv(RELEVANT_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    return df, relevant_chats

# ------------------- HANDOVER ANALYSIS -------------------
def run_handover_analysis():
    genai.configure(api_key=HANDOVER_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    df = pd.read_csv(RELEVANT_OUTPUT_FILE)
    df["Sender"] = df["Sender"].fillna("").astype(str)
    df["Message"] = df["Message"].fillna("").astype(str)
    df = df.sort_values(by=["Customer Number", "Timestamp"])

    results = []
    customers = df["Customer Number"].unique()
    progress = st.progress(0)

    for idx, cust in enumerate(customers):
        group = df[df["Customer Number"] == cust].reset_index(drop=True)
        for i, row in group.iterrows():
            if row["Sender"].lower() == "outgoingapi" and row["Message"].startswith("It seems like I need some extra help"):
                before_transfer = group.iloc[max(0, i-5):i]
                before_text = "\n".join([f"{r['Sender']}: {r['Message']}" for _, r in before_transfer.iterrows()])
                after_transfer = group.iloc[i+1:]
                after_text = "\n".join([f"{r['Sender']}: {r['Message']}" for _, r in after_transfer.iterrows()])

                prompt = f"""
                Customer conversation before transfer (last few messages):
                {before_text}

                Conversation after transfer (agent joined):
                {after_text}

                Task:
                1. From the last few messages before transfer, identify the customer's real issue
                   (ignore filler like 'hello?', '?', 'pls reply').
                2. From the post-transfer conversation, summarize the actual solution provided by the agent
                   (ignore greetings & formalities).
                3. Answer concisely in this format:
                   - Scenario: <short summary of issue>
                   - Solution: <short summary of agent’s solution>
                """
                try:
                    response = model.generate_content(prompt)
                    summary = response.text.strip()
                    scenario, solution = "", ""
                    for line in summary.splitlines():
                        if line.lower().startswith("- scenario"):
                            scenario = line.split(":", 1)[-1].strip()
                        elif line.lower().startswith("- solution"):
                            solution = line.split(":", 1)[-1].strip()
                except Exception as e:
                    scenario, solution = f"[ERROR] {e}", ""

                results.append({
                    "Customer Number": row["Customer Number"],
                    "Chat ID": row["Chat ID"],
                    "Timestamp": row["Timestamp"],
                    "Chat transfer scenario": scenario,
                    "Solution provided by agent": solution
                })
        progress.progress((idx + 1) / len(customers))

    output = pd.DataFrame(results)
    output.to_csv(HANDOVER_FILE, index=False, encoding="utf-8-sig")
    return output

# ------------------- ESCALATION GAP ANALYSIS -------------------
def call_gemini(prompt, retries=0):
    global api_index
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        err = str(e)
        if "429" in err:  # rate limit
            wait_time = 5 * (2 ** retries)
            time.sleep(wait_time)
            api_index = (api_index + 1) % len(ESCALATION_API_KEYS)
            genai.configure(api_key=ESCALATION_API_KEYS[api_index])
            if retries < 5:
                return call_gemini(prompt, retries + 1)
        elif "500" in err:
            time.sleep(10)
            if retries < 5:
                return call_gemini(prompt, retries + 1)
        return f"Error: {err}"

def classify_batch(batch):
    prompt = """
You are analyzing customer support chats.

For each scenario, classify the escalation:
1. "Bot could’ve answered" → If the request is FAQ/simple info.
2. "Agent required" → If the issue is complex, account-specific, or needs authentication.

Return your output STRICTLY in JSON Lines (NDJSON), one JSON object per line, like this:
{"RowIndex": 12, "EscalationType": "Agent required", "EscalationReason": "Customer asking about account-specific charges, needs agent."}
{"RowIndex": 13, "EscalationType": "Bot could’ve answered", "EscalationReason": "Customer only asked for branch opening hours, bot can handle this."}
---
Scenarios:
"""
    for idx, row in batch.iterrows():
        prompt += f"\nRow {idx}: {row['Chat transfer scenario']}"
    output = call_gemini(prompt)
    results = {}
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            row_id = obj.get("RowIndex")
            esc_type = obj.get("EscalationType", "")
            reason = obj.get("EscalationReason", "")
            if row_id is not None:
                results[row_id] = (esc_type, reason)
        except Exception:
            continue
    return results

def run_escalation_gap_analysis():
    genai.configure(api_key=ESCALATION_API_KEYS[0])
    df = pd.read_csv(HANDOVER_FILE)
    df["EscalationType"] = ""
    df["EscalationReason"] = ""

    BATCH_SIZE = 5
    progress = st.progress(0)

    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))
        batch = df.iloc[start:end]
        results = classify_batch(batch)
        for idx, (etype, reason) in results.items():
            df.at[idx, "EscalationType"] = etype
            df.at[idx, "EscalationReason"] = reason
        df.to_csv(ESCALATION_FILE, index=False, encoding="utf-8-sig")
        progress.progress(end / len(df))

    return df

# ------------------- STREAMLIT APP -------------------
def main():
    st.set_page_config(page_title="Digiconn Chat Pipeline", layout="wide")
    st.title("💬 Dabs Chat Analyzer")

    st.sidebar.header("Scraper Settings")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    start_index = st.sidebar.number_input("Start Index", min_value=1, value=1)
    max_chats = st.sidebar.number_input("Max Chats (0 = all)", min_value=0, value=0)
    headless = st.sidebar.checkbox("Run Headless", value=True)

    # Step 1: Scraping
    if st.button("🚀 Start Scraping"):
        with st.spinner("Scraping chats..."):
            run_scraper(username, password, start_index, max_chats, headless)
        if os.path.exists(OUTPUT_FILE):
            df = pd.read_csv(OUTPUT_FILE)
            st.subheader("📑 Scraped Chats")
            st.dataframe(df, use_container_width=True)
            st.download_button("📥 Download Full CSV",
                               data=df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=OUTPUT_FILE,
                               mime="text/csv")

    # Step 2: Filter Relevant
    if os.path.exists(OUTPUT_FILE):
        if st.button("🎯 Filter Relevant Chats"):
            with st.spinner("Filtering relevant chats..."):
                _, df_relevant = filter_relevant_chats(OUTPUT_FILE)
            st.subheader("🎯 Relevant Chats")
            st.dataframe(df_relevant, use_container_width=True)
            st.download_button("📥 Download Relevant CSV",
                               data=df_relevant.to_csv(index=False).encode("utf-8-sig"),
                               file_name=RELEVANT_OUTPUT_FILE,
                               mime="text/csv")

    # Step 3: Handover Analysis
    if os.path.exists(RELEVANT_OUTPUT_FILE):
        if st.button("🔎 Run Handover Analysis"):
            with st.spinner("Running Agent handover analysis..."):
                df_handover = run_handover_analysis()
            st.subheader("🔎 Handover Analysis")
            st.dataframe(df_handover, use_container_width=True)
            st.download_button("📥 Download Handover CSV",
                               data=df_handover.to_csv(index=False).encode("utf-8-sig"),
                               file_name=HANDOVER_FILE,
                               mime="text/csv")

    # Step 4: Escalation Gap Analysis
    if os.path.exists(HANDOVER_FILE):
        if st.button("🧩 Run Escalation Gap Analysis"):
            with st.spinner("Running escalation gap analysis..."):
                df_escalation = run_escalation_gap_analysis()
            st.subheader("🧩 Escalation Gap Analysis")
            st.dataframe(df_escalation, use_container_width=True)
            st.download_button("📥 Download Escalation CSV",
                               data=df_escalation.to_csv(index=False).encode("utf-8-sig"),
                               file_name=ESCALATION_FILE,
                               mime="text/csv")

if __name__ == "__main__":
    main()

