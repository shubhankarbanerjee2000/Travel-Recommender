# app.py
# Streamlit travel MVP with improved UI, chat, YouTube/blog embeds, and personal history
# Model and endpoint REMAIN the same: meta-llama/llama-4-scout-17b-16e-instruct via Groq Chat Completions

import os
import json
import requests
import feedparser
import streamlit as st
import re

# ---------------------------------------
# Configuration
# ---------------------------------------
st.set_page_config(page_title="AI Travel Buddy !", page_icon="✈️", layout="wide")

def load_and_inject_css(path="assets/custom.css"):
  try:
    with open(path, "r", encoding="utf-8") as f:
      css = f.read()
      st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
  except FileNotFoundError:
    st.warning(f"CSS file not found: {path} — continuing without custom styles.")

# call right away to apply styles (adjust path if you place the file elsewhere)
load_and_inject_css("assets/custom.css")

# --- Load Groq API key from Streamlit Secrets ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("⚠️ GROQ_API_KEY is missing! Please set it in Streamlit secrets.")
    st.stop()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # unchanged

HEADERS = {
  "Authorization": f"Bearer {GROQ_API_KEY}",
  "Content-Type": "application/json"
}

def extract_json_array(text: str) -> str:
  """Return the first top-level JSON array substring from text (handles noisy wrappers)."""
  if not isinstance(text, str):
    raise ValueError("extract_json_array expects a string")
  start = text.find("[")
  if start == -1:
    raise ValueError("no '[' found in text")
  depth = 0
  for i in range(start, len(text)):
    c = text[i]
    if c == "[":
      depth += 1
    elif c == "]":
      depth -= 1
      if depth == 0:
        return text[start:i+1]
  raise ValueError("no matching closing ']' found")

# ---------------------------------------
# Core API calls (unchanged model)
# ---------------------------------------
def get_travel_recommendations(destination, budget, experience_type, month):
  """
  Uses the SAME Groq endpoint and model to return structured recs as JSON.
  """
  prompt = f"""
Suggest 3-4 unique travel activities for {destination} in {month} with a {budget} budget focused on {experience_type}.
For each suggestion, return JSON with fields:
- title (string)
- desc (string, 2-3 sentences)
- links (array of 2 URLs: mix of blog and YouTube)
Respond ONLY with a valid JSON array.
"""
  payload = {
    "model": GROQ_MODEL,  # unchanged
    "messages": [
      {"role": "system", "content": "You are a travel AI assistant. Always return ONLY valid JSON."},
      {"role": "user", "content": prompt}
    ],
    "temperature": 0.7
  }

  response = requests.post(GROQ_URL, headers=HEADERS, json=payload)
  response.raise_for_status()
  # ensure bytes are decoded as UTF-8 (prevents double‑decoded chars like "Ã¢ÂÂ¬")
  response.encoding = "utf-8"
  # choices is a list; take the first item
  text = response.json()["choices"][0]["message"]["content"]
  # Parse JSON robustly (allow noisy wrapper text)
  try:
    try:
      recs = json.loads(text)
    except json.JSONDecodeError:
      arr = extract_json_array(text)
      recs = json.loads(arr)
    return recs
  except Exception as e:
    # store raw assistant output for debugging in the session and raise a friendly error
    st.session_state["last_recs_raw"] = text
    raise ValueError(f"Failed to parse recommendations JSON: {e}. Raw output stored in session_state['last_recs_raw'].")

def fetch_global_events(month_name, limit=4):
  """
  Ask the LLM to return a JSON array of global events/festivals/experiences for the given month.
  Each item must contain: title, date, hook (short one-line sell).
  Returns list[dict].
  """
  prompt = f"""
Return a JSON array with {limit} notable global events / festivals / experiences happening in {month_name}.
For each item return an object with fields:
- title (string)
- date (string)
- location (string)           
- description (string)
- hook (string, one short marketing line)
Respond ONLY with a valid JSON array.
"""
  messages = [
    {"role": "system", "content": "You are a concise travel assistant. Always return ONLY valid JSON."},
    {"role": "user", "content": prompt}
  ]
  try:
    text = send_chat_completion(messages, temperature=0.3)
    try:
      try:
        events = json.loads(text)
      except json.JSONDecodeError:
        arr = extract_json_array(text)
        events = json.loads(arr)
      return events
    except Exception as e:
      st.session_state["last_fetch_raw"] = text
      raise ValueError(f"Failed to parse events JSON: {e}. Raw output stored in session_state['last_fetch_raw'].")
  except Exception:
    return []

def groq_chat_stream(messages, temperature=0.5):
  """
  Streams assistant output using the SAME model and endpoint via SSE.
  Yields content deltas for write_stream.
  """
  payload = {
    "model": GROQ_MODEL,  # unchanged
    "messages": messages,
    "temperature": temperature,
    "stream": True
  }
  resp = requests.post(GROQ_URL, headers=HEADERS, json=payload, stream=True)
  resp.raise_for_status()
  # ensure the SSE stream is decoded as UTF-8
  resp.encoding = "utf-8"

  for line in resp.iter_lines(decode_unicode=True):
    if not line:
      continue
    if line.startswith("data: "):
      data = line[len("data: "):]
      if data == "[DONE]":
        break
      try:
        obj = json.loads(data)
        # choices is a list when streaming; use index 0
        delta = obj["choices"][0]["delta"].get("content")
        if delta:
          yield delta
      except Exception:
        continue
  # end groq_chat_stream

def send_chat_completion(messages, temperature=0.5):
  """
  Synchronous, non-streaming chat completion. Returns full assistant text.
  Used to immediately send prompts created from buttons (Your Journeys).
  """
  payload = {
    "model": GROQ_MODEL,
    "messages": messages,
    "temperature": temperature
  }
  resp = requests.post(GROQ_URL, headers=HEADERS, json=payload)
  resp.raise_for_status()
  resp.encoding = "utf-8"
  # choices is a list; take the first item
  text = resp.json()["choices"][0]["message"]["content"]
  return text.strip()

# ---------------------------------------
# Helpers: media and RSS
# ---------------------------------------
def render_links(links):
  yt = [l for l in links if "youtube.com" in l or "youtu.be" in l]
  blogs = [l for l in links if l not in yt]
  if yt:
    st.write("Watch")
    for l in yt:
      # render each YouTube link as a clickable Markdown link
      st.markdown(f"- [{l}]({l})")
  if blogs:
    st.write("Read")
    for b in blogs:
      # render each blog link as a clickable Markdown link
      st.markdown(f"- [{b}]({b})")

def fetch_rss_items(feed_url, limit=3):
  feed = feedparser.parse(feed_url)
  out = []
  for e in feed.entries[:limit]:
    out.append({
      "title": getattr(e, "title", "Untitled"),
      "link": getattr(e, "link", ""),
      "summary": getattr(e, "summary", getattr(e, "description", "")),
      "published": getattr(e, "published", "")
    })
  return out

def month_signals(month_name):
  return [
    f"{month_name}: consider regional festivals and public holidays that affect pricing and crowds.",
    "Combine city stays with nearby countryside day trips to avoid weekend surges.",
    "Use weekday museum entries, city passes, and early timed tickets to cut queues."
  ]

# ---------------------------------------
# Session state
# ---------------------------------------
if "chat" not in st.session_state:
  st.session_state.chat = []
if "profile" not in st.session_state:
  st.session_state.profile = {
    "name": "Arpita",
    "visited": ["Manali", "Goa", "Bali"],
    "interests": ["Culture", "Food", "Adventure"]
  }

# If a previous action requested navigation, apply it before rendering sidebar
if "navigate_to" in st.session_state:
  st.session_state["page"] = st.session_state.pop("navigate_to")

# --- new: prefer any page set via URL query param (stable across reruns) ---
qp = st.query_params
if qp.get("page"):
  st.session_state["page"] = qp["page"][0]

# ---------------------------------------
# Sidebar
# ---------------------------------------
with st.sidebar:
  st.header("Navigation")
  page = st.radio("Go to", ["Discover", "Chatbot", "Your Journeys", "What's Happening"], key="page")
  page = st.session_state.get("page", page)
  st.divider()
  st.header("Quick price check")
  st.caption("Use the Chatbot to estimate activity prices. For example: 'Average price for a Seine dinner cruise in December? Include min/median/max.'")

# ---------------------------------------
# Header
# ---------------------------------------
st.title("✈️ AI Travel Buddy ! ")

# ---------------------------------------
# Pages
# ---------------------------------------
if page == "Discover":
  st.subheader("Welcome Arpita ! Let's plan a trip")

  with st.form("travel_form"):
    # if a prefill exists (from "Plan a Trip"), use it as the default
    prefill_dest = st.session_state.get("prefill", {}).get("destination", "Paris")
    destination = st.text_input("Destination", prefill_dest)
    budget = st.slider("Budget (INR)", min_value=0, max_value=200000, value=50000, step=5000, format="%d Rs")
    experience = st.multiselect(
      "Experience Type (choose one or more)",
      ["Relaxed", "Adventure", "Family", "Romantic", "Cultural", "Solo", "Friends"],
      default=["Relaxed"]
    )
    # convert list to human-readable string for the assistant prompt
    experience_str = ", ".join(experience) if isinstance(experience, (list, tuple)) else experience
    month = st.text_input("Travel Month", "December")
    submitted = st.form_submit_button("Get Recommendations")
 
  if submitted:
     st.write("🔎 Fetching AI recommendations...")
     try:
      recs = get_travel_recommendations(destination, f"{budget} Rs", experience_str, month)
      for r in recs:
        title = r.get("title", "")
        desc = r.get("desc", "")
        # Build clickable links HTML (each on its own line)
        links_html = ""
        if "links" in r:
          for l in r["links"]:
            links_html += f'<div><a href="{l}" target="_blank">{l}</a></div>'

        # Plain "Book Now" link (no boxed button). Append Chatbot prompt text after the suggestion.
        card_html = f'''
        <div class="rec-card">
          <div class="rec-title">{title}</div>
          <div class="rec-desc">{desc}</div>
          <div class="rec-links">{links_html}</div>
          <div style="margin-top:8px;"><a href="https://example.com/booking" target="_blank">📌 Book Now</a></div>
          <div style="margin-top:8px;color:#444;font-size:13px;">For more questions ask the Chatbot.</div>
        </div>
        '''
        st.markdown(card_html, unsafe_allow_html=True)
        st.divider()
     except Exception as e:
      st.error(f"⚠️ Something went wrong: {e}")

elif page == "Chatbot":
  st.subheader("Hey Arpita ! I am the Travel Chatbot")
  st.caption("Ask for prices, itineraries, neighborhoods, or visa notes. Responses stream live.")

  # Display prior chat
  for m in st.session_state.chat:
    with st.chat_message(m["role"]):
      st.markdown(m["content"])

  user_msg = st.chat_input("Ask a travel question or price check...")
  if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
      st.markdown(user_msg)

    messages = [
                 {"role": "system", "content": "You are a concise, reliable travel assistant. Prefer ranges and practical steps."}
               ] + st.session_state.chat

    def stream_gen():
      payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.5,
        "stream": True
      }
      resp = requests.post(GROQ_URL, headers=HEADERS, json=payload, stream=True)
      resp.raise_for_status()
      # ensure the SSE stream is decoded as UTF-8
      resp.encoding = "utf-8"
      for line in resp.iter_lines(decode_unicode=True):
        if not line:
          continue
        if line.startswith("data: "):
          data = line[len("data: "):]
          if data == "[DONE]":
            break
          try:
            obj = json.loads(data)
            # Streaming delta path: choices[0].delta.content
            delta = obj["choices"][0]["delta"].get("content")
            if delta:
              yield delta
          except Exception:
            continue

    with st.chat_message("assistant"):
      full_text = st.write_stream(stream_gen)  # returns full concatenated text

    st.session_state.chat.append({"role": "assistant", "content": full_text})

  # Prompt ideas moved here (only visible on Chatbot page)
  with st.expander("Prompt ideas"):
    st.write("- Estimate min/median/max price in local currency for a 2-hour food tour, this month, with 2 bookable examples.")
    st.write("- Two lesser-known neighborhoods with evening food markets and average dish prices; add one rainy-day activity.")

elif page == "Your Journeys":
  st.subheader("Your Journeys")
  st.caption("Relive your past trips and discover where to go next")
  prof = st.session_state.profile
  st.subheader(f"Hi {prof['name']}, you have visited " + ", ".join(prof["visited"]) + ".")
  st.write("Preferences: " + ", ".join(prof["interests"]) + ".")

  # Visual: use provided real images for visited destinations
  st.markdown("")  # spacing
  cols = st.columns(min(len(prof["visited"]), 3))
  for i, city in enumerate(prof["visited"]):
    # Use the Pexels images provided (guaranteed themes)
    q = city.lower()
    if "manali" in q:
      img_url = "https://images.pexels.com/photos/785419/pexels-photo-785419.jpeg"
    elif "goa" in q:
      img_url = "https://images.pexels.com/photos/4428285/pexels-photo-4428285.jpeg"
    elif "bali" in q:
      img_url = "https://images.pexels.com/photos/2166559/pexels-photo-2166559.jpeg"
    else:
      img_url = f"https://source.unsplash.com/800x600/?{city.replace(' ', '%20')}"
    col = cols[i % len(cols)]
    with col:
      # replace deprecated use_container_width with the new width mapping
      st.image(img_url, caption=f"{city}", width='stretch')

  st.divider()
  st.subheader("Our suggestions — inspired by your journeys")
  st.write("Handpicked next-destination ideas based on your past trips. Tap a suggestion to add a ready-made planning prompt to the Chatbot.")

  # Attractive suggestion cards with CTA buttons
  s1_col, s2_col, s3_col = st.columns(3)
  with s1_col:
    # visual card with light-yellow background (see CSS .suggest-card)
    st.markdown('<div class="suggest-card"><strong>From Manali → Darjeeling</strong><div class="suggest-desc">Crisp hill-station mornings, tea gardens, and scenic toy-train rides — a perfect September escape.</div></div>', unsafe_allow_html=True)
    if st.button("Explore Darjeeling"):
       user_prompt = "Suggest a 3-day Darjeeling trip this September with top activities and price ranges."
       st.session_state.chat.append({"role": "user", "content": user_prompt})
       # immediately send to LLM and append assistant reply
       with st.spinner("Getting assistant suggestion..."):
         try:
           messages = [{"role": "system", "content": "You are a concise, reliable travel assistant. Prefer ranges and practical steps."}] + st.session_state.chat
           assistant_text = send_chat_completion(messages, temperature=0.5)
           st.session_state.chat.append({"role": "assistant", "content": assistant_text})
           st.success("Suggestion added and assistant replied (check Chatbot).")
         except Exception as e:
           st.error(f"Assistant error: {e}")
  with s2_col:
    st.markdown('<div class="suggest-card"><strong>From Goa → Pondicherry</strong><div class="suggest-desc">Laid-back beaches, French-colonial charm, and coastal cafés — a mellow December getaway idea.</div></div>', unsafe_allow_html=True)
    if st.button("Plan Pondicherry"):
      user_prompt = "Suggest a 3-day Pondicherry plan for December with food highlights and price ranges."
      st.session_state.chat.append({"role": "user", "content": user_prompt})
      with st.spinner("Getting assistant suggestion..."):
        try:
          messages = [{"role": "system", "content": "You are a concise, reliable travel assistant. Prefer ranges and practical steps."}] + st.session_state.chat
          assistant_text = send_chat_completion(messages, temperature=0.5)
          st.session_state.chat.append({"role": "assistant", "content": assistant_text})
          st.success("Suggestion added and assistant replied (check Chatbot).")
        except Exception as e:
          st.error(f"Assistant error: {e}")
  with s3_col:
    st.markdown('<div class="suggest-card"><strong>From Bali → Phuket</strong><div class="suggest-desc">Sunset beaches, island hopping, and lively night markets — a tropical switch for your next summer.</div></div>', unsafe_allow_html=True)
    if st.button("Explore Phuket"):
      user_prompt = "Suggest top beach activities in Phuket this summer with estimated prices."
      st.session_state.chat.append({"role": "user", "content": user_prompt})
      with st.spinner("Getting assistant suggestion..."):
        try:
          messages = [{"role": "system", "content": "You are a concise, reliable travel assistant. Prefer ranges and practical steps."}] + st.session_state.chat
          assistant_text = send_chat_completion(messages, temperature=0.5)
          st.session_state.chat.append({"role": "assistant", "content": assistant_text})
          st.success("Suggestion added and assistant replied (check Chatbot).")
        except Exception as e:
          st.error(f"Assistant error: {e}")

  st.divider()
  st.markdown("Saved articles and videos")
  feed_url = st.text_input("RSS feed URL (e.g., https://www.intrepidtravel.com/adventures/rss/ )", "")
  if feed_url:
    try:
      items = fetch_rss_items(feed_url, limit=3)
      for it in items:
        st.markdown(f"**{it['title']}**")
        st.write(it["summary"], unsafe_allow_html=True)
        st.markdown(f"[Open full article]({it['link']})")
        st.caption(it["published"])
        st.divider()
    except Exception as e:
      st.error(f"RSS error: {e}")

elif page == "What's Happening":
  st.subheader("What’s happening around the world")
  st.caption("Discover festivals, events and experiences worth travelling for.")
  month_name = st.text_input("Month", "September")

  st.markdown("_We will send a quick prompt we send to the Chatbot to fetch global highlights for the selected month_")

  if st.button("Fetch global highlights"):
    with st.spinner("Fetching global events..."):
      events = fetch_global_events(month_name, limit=4)
    if not events:
      st.info("No events found or the assistant returned malformed JSON. Try another month.")
    else:
      for i, ev in enumerate(events):
        title = ev.get("title", "Untitled")
        date = ev.get("date", "")
        location = ev.get("location", "")  # display location returned by the assistant
        description = ev.get("description", "")
        hook = ev.get("hook", "")
        link = ev.get("link", "")

        # two-column layout: left = headline + hook, right = details + CTA
        left_col, right_col = st.columns([2, 3])
        with left_col:
          st.markdown(f"### {title}")
          if location:
            st.markdown(f"**Location:** {location}")
          if date:
            st.markdown(f"**Date:** {date}")
          if hook:
            st.write(hook)
        with right_col:
          st.markdown("**Details**")
          if description:
            st.write(description)
          if link:
            st.markdown(f"[Learn more]({link})")
          st.markdown("")  # spacing
          if st.button("Plan a Trip", key=f"plan_trip_{i}"):
            # prefill destination
            if location:
              st.session_state.setdefault("prefill", {})["destination"] = location
            # set a URL query param (and session) then rerun — this reliably updates the sidebar radio
            st.session_state["page"] = "Discover"
            st.experimental_set_query_params(page="Discover")
            st.experimental_rerun()
        st.divider()
  else:
    # show plain inline instruction (not an info/alert block)
    st.write("Click **Fetch global highlights** to ask the Chatbot for notable events this month.")
