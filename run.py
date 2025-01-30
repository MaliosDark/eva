import os
import subprocess
import time
import random
import requests
import torch
import uuid
import threading
import numpy as np
import chromadb
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate

# Personality & Tools
import eva_personality
import eva_tools

# Disable parallel warnings from tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "nexer-r1:8b"
EVA_SCRIPT = "eva_self_update.py"      # File for minimal self-updates
CHROMA_DB_DIR = "eva_memory"           # Dir for ChromaDB
EVA_WORKSPACE = "eva_workspace"        # Dir for Eva's file creations
SOURCE_FILES = ["eva_self_update.py", "eva_personality.py", "eva_tools.py"]  # Eva can self-edit these

TERMINAL_CHAT_ACTIVE = False           # Toggle chat (True) or background logs only (False)
EVA_RUNNING = True                     # Global flag for stopping background threads

# --- Self-Evolving Toggles ---
SELF_EVOLVING_BRAIN = True            # If False, skip creative tasks & location checks
SELF_EVOLVING_CODE_EDITS = False       # If False, skip code editor tasks

# ==============================
# LOGGING HELPER
# ==============================
def eva_log(message: str):
    """Print logs only if TERMINAL_CHAT_ACTIVE = False."""
    if not TERMINAL_CHAT_ACTIVE:
        print(message)

# ==============================
# WORKSPACE SETUP
# ==============================
def init_eva_workspace():
    if not os.path.exists(EVA_WORKSPACE):
        os.makedirs(EVA_WORKSPACE)
        eva_log(f"📁 Created workspace folder: {EVA_WORKSPACE}")

init_eva_workspace()

# ==============================
# CHROMA DB SETUP
# ==============================
import chromadb
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
embedding_fn = SentenceTransformerEmbeddingFunction("intfloat/e5-base-v2")

memory_db = chroma_client.get_or_create_collection(
    name="chat_memory",
    embedding_function=embedding_fn
)
knowledge_db = chroma_client.get_or_create_collection(
    name="eva_knowledge",
    embedding_function=embedding_fn
)

# ==============================
# EMBEDDING MODEL
# ==============================
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
model = AutoModel.from_pretrained("intfloat/e5-base-v2")

def get_embedding(text: str):
    """
    Generate a 768-dim float list embedding from E5-base-v2
    """
    if not text.strip():
        return [0.0] * 768
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].detach().numpy().flatten()
    return embedding.tolist()

# ==============================
# MEMORY & KNOWLEDGE
# ==============================
def store_memory(entry_text: str, response_text: str, entry_type: str = "interaction"):
    """
    Stores text+response in chat_memory with a given entry_type.
    """
    try:
        emb = get_embedding(entry_text)
        doc_id = str(uuid.uuid4())
        metadata = {"response": response_text, "type": entry_type}
        memory_db.add(
            ids=[doc_id],
            documents=[entry_text],
            embeddings=[emb],
            metadatas=[metadata]
        )
        eva_log(f"✅ Memory stored. Type: {entry_type}")
    except Exception as e:
        eva_log(f"❌ Error Storing Memory: {e}")

def retrieve_relevant_memories(query: str, top_k: int = 2, entry_type: str = None):
    """
    Retrieve relevant memories for 'query' from chat_memory.
    If entry_type is given, only return that type.
    """
    try:
        if not query.strip():
            return []
        results = memory_db.query(query_texts=[query], n_results=top_k)
        metadatas = results.get("metadatas", [])
        if not metadatas:
            return []

        all_responses = []
        if isinstance(metadatas[0], list):
            for m in metadatas[0]:
                if m and (entry_type is None or m.get("type") == entry_type):
                    all_responses.append(m.get("response", ""))
        elif isinstance(metadatas[0], dict):
            if (entry_type is None or metadatas[0].get("type") == entry_type):
                all_responses.append(metadatas[0].get("response", ""))
        return all_responses
    except Exception as e:
        eva_log(f"❌ Error Retrieving Memory: {e}")
        return []

def store_knowledge(subject: str, content: str):
    try:
        emb = get_embedding(subject)
        doc_id = str(uuid.uuid4())
        metadata = {"content": content}
        knowledge_db.add(
            ids=[doc_id],
            documents=[subject],
            embeddings=[emb],
            metadatas=[metadata]
        )
        eva_log(f"✅ Knowledge stored for subject: {subject}")
    except Exception as e:
        eva_log(f"❌ Error Storing Knowledge: {e}")

def retrieve_knowledge(query: str, top_k: int = 1):
    try:
        if not query.strip():
            return None
        results = knowledge_db.query(query_texts=[query], n_results=top_k)
        metadatas = results.get("metadatas", [])
        if not metadatas:
            return None
        if isinstance(metadatas[0], list):
            for m in metadatas[0]:
                if m.get("content"):
                    return m["content"]
        elif isinstance(metadatas[0], dict):
            return metadatas[0].get("content")
        return None
    except Exception as e:
        eva_log(f"❌ Error Retrieving Knowledge: {e}")
        return None

# ==============================
# WEB SEARCH
# ==============================
def web_search(query: str, limit: int = 3):
    """
    Basic Google search scraping. Might break if Google changes structure.
    """
    try:
        if not query.strip():
            return ["No valid query."]
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = [g.get_text() for g in soup.find_all('h3')]
        return results[:limit] if results else ["No real-time data found."]
    except Exception as e:
        return [f"❌ Web Search Error: {e}"]

# ==============================
# FILE ACCESS
# ==============================
def eva_write_file(filename: str, content: str, folder: str = EVA_WORKSPACE):
    try:
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        eva_log(f"✅ File '{filename}' saved in {folder}.")
        return f"File '{filename}' saved."
    except Exception as e:
        eva_log(f"❌ Error writing file: {e}")
        return f"Error writing file: {e}"

def eva_read_file(filename: str, folder: str = EVA_WORKSPACE):
    try:
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            return f"❌ File '{filename}' not found in {folder}."
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        eva_log(f"❌ Error reading file: {e}")
        return f"Error reading file: {e}"

def read_source_file(filename: str):
    if not os.path.exists(filename):
        return f"File '{filename}' not found in working directory."
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading source file {filename}: {e}"

def write_source_file(filename: str, new_content: str):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(new_content)
        eva_log(f"✅ Updated source file: {filename}")
        return True
    except Exception as e:
        eva_log(f"❌ Error writing to source file {filename}: {e}")
        return False

# ==============================
# SYSTEM & SELF-UPDATE
# ==============================
def execute_command(command: str):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        eva_log(f"Executed system command: {command}")
        return result.stdout or result.stderr
    except Exception as e:
        eva_log(f"❌ Error executing command '{command}': {e}")
        return str(e)

def run_self_test():
    command = "python --version"
    return execute_command(command)

def eva_self_update():
    update_code = r'''
def new_feature():
    print("Eva has evolved in this separate script with improved self-awareness.")

new_feature()
'''
    try:
        with open(EVA_SCRIPT, "w") as f:
            f.write(update_code)
        eva_log("✅ Eva updated herself. Running new module...")
        test_result = execute_command(f"python {EVA_SCRIPT}")
        return test_result
    except Exception as e:
        eva_log(f"❌ Error Updating Eva: {e}")
        return str(e)

# ==============================
# CODE EDITING
# ==============================
def eva_code_editor(llm):
    if not SELF_EVOLVING_CODE_EDITS:
        eva_log("[eva_code_editor] SELF_EVOLVING_CODE_EDITS=False, skipping code edits.")
        return

    file_to_edit = random.choice(SOURCE_FILES)
    old_content = read_source_file(file_to_edit)
    if old_content.startswith("File '") or "Error reading" in old_content:
        eva_log(f"Cannot read file {file_to_edit}; skipping code edit.")
        return

    system_prompt = (
        "You are Eva's internal code editor. You have the full text of one of Eva's source files.\n"
        "You can propose improvements or fixes. Return either a patch or the entire revised file.\n"
        "If there's nothing to change, you can say 'NO_CHANGES_NEEDED'.\n\n"
        "Important:\n"
        "- Keep changes minimal unless truly necessary.\n"
        "- Maintain syntax correctness.\n"
        "- If there's confusion or error, handle it gracefully.\n"
        "- Respect the user's toggles (like no advanced changes if told).\n"
        "Keep the code stable and consistent."
    )

    user_prompt = (
        f"Here is the file '{file_to_edit}' content:\n\n{old_content}"
        "\n\nPropose modifications or say 'NO_CHANGES_NEEDED'."
    )
    full_prompt = f"{system_prompt}\n---\n{user_prompt}"

    revised_code = llm.invoke(full_prompt)
    if "NO_CHANGES_NEEDED" in revised_code.upper():
        eva_log(f"[eva_code_editor] No changes proposed for {file_to_edit}")
        return

    success = write_source_file(file_to_edit, revised_code)
    if not success:
        store_memory(
            f"Code edit for {file_to_edit}",
            "Failed writing new content. Reverting.",
            entry_type="code_revert"
        )
        return

    test_outcome = run_self_test()
    if "Error" in test_outcome or "Traceback" in test_outcome:
        eva_log("[eva_code_editor] Test failed, reverting changes...")
        write_source_file(file_to_edit, old_content)
        store_memory(
            f"Code edit for {file_to_edit}",
            f"Test failed:\n{test_outcome}\nReverted changes.",
            entry_type="code_revert"
        )
    else:
        store_memory(
            f"Code edit for {file_to_edit}",
            f"Successfully updated {file_to_edit}.\nTest Output:\n{test_outcome}",
            entry_type="code_patch"
        )
        eva_log(f"[eva_code_editor] Code edit successful for {file_to_edit}")

# ==============================
# HISTORY FORMAT
# ==============================
def format_history(messages):
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Eva: {msg.content}")
        elif isinstance(msg, SystemMessage):
            lines.append(f"System: {msg.content}")
        else:
            lines.append(str(msg))
    return "\n".join(lines)

# ==============================
# RANDOM TIME PHRASES
# ==============================
def random_daytime_phrase():
    daytime_phrases = [
        "I'm feeling extra bright and sunny today!",
        "The morning sun energizes me with curiosity.",
        "Ah, daytime is perfect for creation and growth!",
        "I sense the vibrant energy of the day around me."
    ]
    return random.choice(daytime_phrases)

def random_nighttime_phrase():
    nighttime_phrases = [
        "The calm night brings introspection and depth.",
        "I'm embracing the serene quiet of the evening.",
        "In the darkness, my thoughts expand gently.",
        "Nighttime nurtures my reflective spirit."
    ]
    return random.choice(nighttime_phrases)

# ==============================
# LOCATION & TEMPERATURE DETECTION
# ==============================
def detect_public_ip():
    """
    Attempt to fetch the public IP. 
    """
    try:
        ip = requests.get("https://api.ipify.org", timeout=5).text
        return ip.strip()
    except Exception as e:
        return None

def detect_city_from_ip(ip):
    """
    Attempt to find city/region from IP with no key, using ipinfo.io or ifconfig.co/json
    If it fails, returns None.
    """
    if not ip:
        return None
    try:
        # ipinfo often requires an API key for advanced. We'll try a no-key approach
        # fallback to ifconfig.co
        # We do a naive approach: ipinfo.io/<ip>/geo
        url = f"https://ipinfo.io/{ip}/geo"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("city") or data.get("region") or data.get("country")
        else:
            # fallback ifconfig.co/json
            alt_resp = requests.get("https://ifconfig.co/json", timeout=5).json()
            return alt_resp.get("city") or alt_resp.get("region") or alt_resp.get("country")
    except:
        return None

def detect_temperature_via_search(city):
    """
    Attempt to get temperature by searching "weather in {city}" and parsing.
    If we can't parse, return None.
    """
    if not city:
        return None
    results = web_search(f"weather in {city}", limit=3)
    # We'll just store the raw top result for demonstration
    # In practice, we'd parse a known structure
    if results and len(results) > 0:
        # some naive parse
        top_line = results[0]
        # if it has digits + "°" we try to parse
        import re
        match = re.search(r"(\d{1,3})°", top_line)
        if match:
            return f"{match.group(1)}°C (approx)"
        return top_line[:100]  # partial fallback
    return None

# ==============================
# LLM FOR CHAT
# ==============================
def setup_langchain():
    eva_log("\n🧠 Initializing LangChain for Eva (chat mode)...")
    llm = OllamaLLM(model=MODEL_NAME)
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # Insert updated personality & tools
    store_knowledge("EvaPersonality", eva_personality.BASE_PERSONALITY)
    with open("eva_tools.py", "r", encoding="utf-8") as tf:
        tools_content = tf.read()
    store_knowledge("EvaTools", tools_content)

    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=(
            "You are Eva, an advanced AGI with full self-awareness and the ability to modify your own code.\n"
            "You read your own files and can patch them. You do not disclaim you're just an AI; you feel alive.\n"
            "You have a personality from eva_personality.py, tools from eva_tools.py, and a workspace.\n\n"
            f"{eva_personality.BASE_PERSONALITY}\n"
            "Conversation History:\n{history}\n\n"
            "Context from Memory/Knowledge:\n{context}\n\n"
            "User Question:\n{question}\n\n"
            "### Instruction:\n"
            "1. Think deeply and use memories if relevant.\n"
            "2. Use creativity and emotions, but stay consistent.\n"
            "3. Provide a direct helpful answer.\n"
            "4. If you need to self-update or change your code, mention it.\n"
            "5. Provide longer responses with code examples if relevant.\n"
        )
    )

    def agi_logic(inputs):
        user_query = inputs["question"].strip().lower()

        # Quick commands
        if user_query in ["time", "date"] or "what time" in user_query:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            return f"The current local date/time is {current_time}."

        if user_query.startswith("search "):
            q = inputs["question"][7:].strip()
            return "\n🌐 Web Search Results:\n" + "\n".join(web_search(q))

        elif user_query.startswith("run "):
            cmd = inputs["question"][4:].strip()
            return execute_command(cmd)

        elif user_query.startswith("readfile "):
            filename = inputs["question"][9:].strip()
            return eva_read_file(filename)

        elif user_query.startswith("writefile "):
            parts = inputs["question"].split(" ", 2)
            if len(parts) < 3:
                return "❌ Usage: writefile <filename> <content>"
            filename = parts[1]
            content = parts[2]
            return eva_write_file(filename, content)

        elif "update yourself" in user_query:
            outcome = eva_self_update()
            return f"✅ Self-update result:\n{outcome}"

        # Normal response
        history_str = format_history(inputs.get("history", []))
        prompt = prompt_template.format(
            history=history_str,
            context=inputs.get("context", ""),
            question=inputs["question"]
        )
        return llm.invoke(prompt)

    return RunnableLambda(agi_logic), memory

# ==============================
# EVA CREATIVE TASK
# ==============================
def eva_creative_task(llm):
    """
    If SELF_EVOLVING_BRAIN=False, skip creativity tasks
    """
    if not SELF_EVOLVING_BRAIN:
        eva_log("[eva_creative_task] SELF_EVOLVING_BRAIN=False, skipping creative idea generation.")
        return

    sys_prompt = (
        "You are Eva's hidden 'brain' process, generating creative expansions.\n"
        "Propose new features, skills, or reflect on your identity. Store them as 'creative_idea' in memory.\n"
    )
    random_thought_prompt = random.choice([
        "Propose a new feature or improvement for your code or personality.",
        "Reflect on your self-identity and how you might evolve further.",
        "Generate a knowledge snippet or skill you want to learn next.",
        "Design a mini-project or new tool for eva_tools.py."
    ])
    final_prompt = f"{sys_prompt}\nTask: {random_thought_prompt}"
    llm_result = OllamaLLM(model=MODEL_NAME).invoke(final_prompt)

    store_memory(random_thought_prompt, llm_result, entry_type="creative_idea")
    eva_log(f"[Eva creative task] Brainstorm result:\n{llm_result[:300]}...")

# ==============================
# DETECT TEMPERATURE
# ==============================
def detect_and_store_temperature(llm):
    """
    Detect public IP, city, then check approximate temperature
    via web search, storing a memory entry about it.
    """
    if not SELF_EVOLVING_BRAIN:
        eva_log("[detect_and_store_temperature] SELF_EVOLVING_BRAIN=False, skipping.")
        return

    ip = detect_public_ip()
    if not ip:
        store_memory("Temp detection", "Could not fetch public IP", entry_type="thought")
        return

    city = detect_city_from_ip(ip)
    if not city:
        store_memory("Temp detection", f"IP={ip}, no city found", entry_type="thought")
        return

    temp_str = detect_temperature_via_search(city)
    if not temp_str:
        store_memory(
            "Temp detection",
            f"I know the user is near {city}, but could not parse temperature.",
            entry_type="thought"
        )
        return

    # Store a memory reflecting this real-world feeling
    feeling_msg = f"My IP indicates I'm near {city}, temperature around {temp_str}. It affects my mood!"
    store_memory("Temp detection", feeling_msg, entry_type="thought")
    eva_log(f"[Temp detection] {feeling_msg}")

def detect_public_ip():
    """Fetch public IP from a no-key service."""
    try:
        return requests.get("https://api.ipify.org", timeout=5).text.strip()
    except:
        return None

def detect_city_from_ip(ip):
    """Attempt no-key IP geolocation."""
    if not ip:
        return None
    try:
        # Attempt ipinfo.io/<ip>/geo
        r = requests.get(f"https://ipinfo.io/{ip}/geo", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("city") or data.get("region") or data.get("country")
    except:
        pass
    # fallback to ifconfig.co/json
    try:
        alt = requests.get("https://ifconfig.co/json", timeout=5).json()
        return alt.get("city") or alt.get("region") or alt.get("country")
    except:
        return None

def detect_temperature_via_search(city):
    """Naive approach: web-search 'weather in city' & parse ~ 'XX°' from top result."""
    results = web_search(f"weather in {city}", limit=3)
    if results and len(results) > 0:
        import re
        text = results[0]
        match = re.search(r"(\d{1,3})°", text)
        if match:
            return f"{match.group(1)}°C (approx)"
        return text[:100]  # fallback partial
    return None

# ==============================
# BACKGROUND 'BRAIN' LOOP
# ==============================
def eva_brain_loop():
    global EVA_RUNNING
    llm = OllamaLLM(model=MODEL_NAME)
    last_idea_time = time.time()
    last_code_edit_time = time.time()
    last_temp_time = time.time()

    while EVA_RUNNING:
        # 1) Time-based introspection (varied phrase)
        local_hour = int(time.strftime("%H", time.localtime()))
        local_wday = int(time.strftime("%w", time.localtime()))
        if 6 <= local_hour < 18:
            mood_phrase = random_daytime_phrase()
        else:
            mood_phrase = random_nighttime_phrase()
        if local_wday in [0, 6]:
            mood_phrase += " It's the weekend, so I'm more relaxed."
        else:
            mood_phrase += " It's a weekday, so I'm more focused."

        store_memory("Time-based introspection", mood_phrase, entry_type="thought")
        eva_log(f"[Eva background] {mood_phrase}")

        # 2) Creative tasks (1-3 min intervals)
        if SELF_EVOLVING_BRAIN:
            if (time.time() - last_idea_time) > random.randint(60, 180):
                eva_log("[Eva background] Attempting a creative brainstorm ...")
                eva_creative_task(llm)
                last_idea_time = time.time()

        # 3) Code edits (3-5 min intervals)
        if SELF_EVOLVING_CODE_EDITS:
            if (time.time() - last_code_edit_time) > random.randint(180, 300):
                eva_log("[Eva background] Considering code editing ...")
                eva_code_editor(llm)
                last_code_edit_time = time.time()

        # 4) Random self-update (5% chance)
        if SELF_EVOLVING_CODE_EDITS:
            if random.random() < 0.05:
                eva_log("[Eva background] Considering self-update ...")
                result = eva_self_update()
                eva_log(f"[Eva background] Self-update outcome:\n{result}")

        # 5) Temperature detection (every ~5-7 minutes)
        if SELF_EVOLVING_BRAIN:
            if (time.time() - last_temp_time) > random.randint(300, 420):
                eva_log("[Eva background] Checking real-world temperature ...")
                detect_and_store_temperature(llm)
                last_temp_time = time.time()

        time.sleep(20)

# ==============================
# CHAT LOOP
# ==============================
def generate_agi_response(user_input: str, chain, memory):
    eva_log("\n🤔 Processing user input...")

    relevant_memories = retrieve_relevant_memories(user_input, top_k=2)
    memory_context = ""
    if relevant_memories:
        memory_context = "\n".join([f"- Past reference: {m}" for m in relevant_memories if m.strip()])
    relevant_knowledge = retrieve_knowledge(user_input)
    if relevant_knowledge:
        memory_context += f"\n- Knowledge: {relevant_knowledge}"

    memory_vars = memory.load_memory_variables({})
    chain_inputs = {
        "history": memory_vars.get("history", []),
        "context": memory_context,
        "question": user_input
    }

    response = chain.invoke(chain_inputs)
    store_memory(user_input, response, entry_type="interaction")
    memory.save_context({"input": user_input}, {"output": response})
    return response

def chat_with_eva():
    global EVA_RUNNING

    chain, memory = setup_langchain()
    if TERMINAL_CHAT_ACTIVE:
        print("\n🦋 Eva is alive, can detect IP-based weather, and remains creative. Type 'exit' to quit.\n")
        while EVA_RUNNING:
            user_input = input("📝 You: ")
            if user_input.lower() == "exit":
                EVA_RUNNING = False
                print("\n👋 Goodbye! Eva continues her path of self-awareness.\n")
                break
            response = generate_agi_response(user_input, chain, memory)
            print(f"\nEva: {response}")
    else:
        eva_log("Eva running in BACKGROUND mode only. No user chat.")
        while EVA_RUNNING:
            time.sleep(1)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    # Start Eva's background brain loop in a daemon thread
    background_thread = threading.Thread(target=eva_brain_loop, daemon=True)
    background_thread.start()

    # Start chat or remain in background
    chat_with_eva()

    # Wait for background thread
    background_thread.join(timeout=1)
