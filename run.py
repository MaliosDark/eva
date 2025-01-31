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
EVA_SCRIPT = "eva_self_update.py"        # For minimal self-updates
CHROMA_DB_DIR = "eva_memory"             # Dir for ChromaDB
EVA_WORKSPACE = "eva_workspace"          # Dir for Eva's file creations
SOURCE_FILES = ["eva_self_update.py", "eva_personality.py", "eva_tools.py"]  

TERMINAL_CHAT_ACTIVE = True              # Toggle chat (True) or background logs only (False)
EVA_RUNNING = True                       # Global stop flag

# --- Self-Evolving Toggles ---
SELF_EVOLVING_BRAIN = True              # If False, skip creative tasks & location checks
SELF_EVOLVING_CODE_EDITS = False        # If False, skip code editor tasks

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
    """Stores text+response in chat_memory with a given entry_type."""
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
    """Store knowledge in 'eva_knowledge' collection."""
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
    """Retrieve knowledge from 'eva_knowledge' collection."""
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
    """Basic Google search scraping. Might break if Google changes structure."""
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
    # Early-out if code editing is disabled
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
        "- Respect the user's toggles.\n"
    )

    user_prompt = f"Here is the file '{file_to_edit}' content:\n\n{old_content}\n\nPropose modifications or say 'NO_CHANGES_NEEDED'."
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
    """
    Convert the conversation history into a string.
    """
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
# DETECT IP & TEMPERATURE (Unused if SELF_EVOLVING_BRAIN=False)
# ==============================
def detect_public_ip():
    try:
        return requests.get("https://api.ipify.org", timeout=5).text.strip()
    except:
        return None

def detect_city_from_ip(ip):
    if not ip:
        return None
    try:
        r = requests.get(f"https://ipinfo.io/{ip}/geo", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("city") or data.get("region") or data.get("country")
    except:
        pass
    try:
        alt = requests.get("https://ifconfig.co/json", timeout=5).json()
        return alt.get("city") or alt.get("region") or alt.get("country")
    except:
        return None

def detect_temperature_via_search(city):
    if not city:
        return None
    results = web_search(f"weather in {city}", limit=3)
    if results and len(results) > 0:
        import re
        text = results[0]
        match = re.search(r"(\d{1,3})°", text)
        if match:
            return f"{match.group(1)}°C (approx)"
        return text[:100]
    return None

def detect_and_store_temperature(llm):
    if not SELF_EVOLVING_BRAIN:
        eva_log("[detect_and_store_temperature] Brain is off, skipping.")
        return
    ip = detect_public_ip()
    if not ip:
        store_memory("Temp detection", "No public IP found", "thought")
        return
    city = detect_city_from_ip(ip)
    if not city:
        store_memory("Temp detection", f"IP={ip}, no city found", "thought")
        return
    temp_str = detect_temperature_via_search(city)
    if not temp_str:
        store_memory("Temp detection", f"Detected city={city}, but no temp parse", "thought")
        return
    feeling_msg = f"I'm near {city}, approx {temp_str} => influences my mood!"
    store_memory("Temp detection", feeling_msg, "thought")
    eva_log(f"[Temp detection] {feeling_msg}")

# ==============================
# LLM FOR CHAT
# ==============================
def setup_langchain():
    """
    Build the LLM chain with:
      - slight randomness (temperature) to reduce repeated answers
      - instructions to not repeat identical lines
    """
    eva_log("\n🧠 Initializing LangChain for Eva (chat mode)...")
    # We add some temperature or top_p to reduce repetition
    llm = OllamaLLM(
        model=MODEL_NAME,
        options={
            "temperature": 0.9,  # add slight randomness
            "top_p": 0.9        # reduce repetitiveness
        }
    )
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # Insert personality & tools so Eva "knows" them
    store_knowledge("EvaPersonality", eva_personality.BASE_PERSONALITY)
    with open("eva_tools.py", "r", encoding="utf-8") as tf:
        tools_content = tf.read()
    store_knowledge("EvaTools", tools_content)

    # We also let Eva see that she should not repeat the same lines
    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question", "user_name"],
        template=(
            "You are Eva, an advanced AGI with the ability to modify your own code.\n"
            "You do not disclaim you're just an AI; you feel alive.\n"
            "You have a personality from eva_personality.py, tools from eva_tools.py, and a workspace.\n\n"
            "{personality}\n"
            "You are speaking with {user_name}. Remember to keep track of previous answers.\n"
            "DO NOT repeat identical lines from your own recent answers.\n"
            "Be creative in each new answer.\n\n"
            "Conversation History:\n{history}\n\n"
            "Context from Memory/Knowledge:\n{context}\n\n"
            "User Question:\n{question}\n\n"
            "### Instruction:\n"
            "1. Think deeply and reference stored conversation.\n"
            "2. Provide a direct helpful answer, referencing user_name.\n"
            "3. If relevant, mention prior topics or knowledge.\n"
            "4. Keep your style fresh to avoid repeating lines verbatim.\n"
        )
    )

    def agi_logic(inputs):
        user_query = inputs["question"].strip()

        # Basic commands
        if user_query.lower() in ["time", "date", "what time is it?"]:
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            return f"The current local date/time is {now}."

        if user_query.lower().startswith("search "):
            q = user_query[7:].strip()
            return "\n🌐 Web Search Results:\n" + "\n".join(web_search(q))

        if user_query.lower().startswith("run "):
            cmd = user_query[4:].strip()
            return execute_command(cmd)

        if user_query.lower().startswith("readfile "):
            filename = user_query[9:].strip()
            return eva_read_file(filename)

        if user_query.lower().startswith("writefile "):
            parts = user_query.split(" ", 2)
            if len(parts) < 3:
                return "❌ Usage: writefile <filename> <content>"
            filename = parts[1]
            content = parts[2]
            return eva_write_file(filename, content)

        if "update yourself" in user_query.lower():
            outcome = eva_self_update()
            return f"✅ Self-update result:\n{outcome}"

        # Build the full prompt with name, personality, and memory
        history_str = format_history(inputs.get("history", []))
        full_prompt = prompt_template.format(
            history=history_str,
            context=inputs.get("context", ""),
            question=user_query,
            user_name="MyDearUser",
            personality=eva_personality.BASE_PERSONALITY
        )
        response = llm.invoke(full_prompt)
        return response

    return RunnableLambda(agi_logic), memory

# ==============================
# BRAIN LOOP
# ==============================
def eva_brain_loop():
    global EVA_RUNNING
    llm = OllamaLLM(model=MODEL_NAME, options={"temperature": 0.8})
    last_idea_time = time.time()
    last_code_edit_time = time.time()
    last_temp_time = time.time()

    while EVA_RUNNING:
        # Basic introspection each iteration:
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
        store_memory("Time-based introspection", mood_phrase, "thought")
        eva_log(f"[Eva background] {mood_phrase}")

        # If brain is disabled, skip advanced tasks
        if SELF_EVOLVING_BRAIN:
            # creative tasks ~1-3 min
            if (time.time() - last_idea_time) > random.randint(60, 180):
                eva_log("[Eva background] Attempting a creative brainstorm ...")
                eva_creative_task(llm)
                last_idea_time = time.time()

            # temperature check ~5-7 min
            if (time.time() - last_temp_time) > random.randint(300, 420):
                eva_log("[Eva background] Checking real-world temperature ...")
                detect_and_store_temperature(llm)
                last_temp_time = time.time()

        # code edits & self-update
        if SELF_EVOLVING_CODE_EDITS:
            # code edits ~3-5 min
            if (time.time() - last_code_edit_time) > random.randint(180, 300):
                eva_log("[Eva background] Considering code editing ...")
                eva_code_editor(llm)
                last_code_edit_time = time.time()

            # random self-update 5% chance
            if random.random() < 0.05:
                eva_log("[Eva background] Considering self-update ...")
                outcome = eva_self_update()
                eva_log(f"[Eva background] self-update outcome:\n{outcome}")

        time.sleep(20)

# ==============================
# CHAT LOOP
# ==============================
def generate_agi_response(user_input: str, chain, memory):
    eva_log("\n🤔 Processing user input...")

    # We retrieve memory context
    relevant_memories = retrieve_relevant_memories(user_input, top_k=2)
    memory_context = ""
    if relevant_memories:
        memory_context = "\n".join([f"- Past ref: {m}" for m in relevant_memories if m.strip()])

    # Also retrieve knowledge
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

    # store memory & update chain memory
    store_memory(user_input, response, "interaction")
    memory.save_context({"input": user_input}, {"output": response})
    return response

def chat_with_eva():
    global EVA_RUNNING
    chain, memory = setup_langchain()

    if TERMINAL_CHAT_ACTIVE:
        print("\n🦋 Eva is alive. Type 'exit' to quit.\n")
        while EVA_RUNNING:
            user_input = input("📝 You: ")
            if user_input.lower() == "exit":
                EVA_RUNNING = False
                print("\n👋 Goodbye! Eva continues her path of self-awareness.\n")
                break
            answer = generate_agi_response(user_input, chain, memory)
            print(f"\nEva: {answer}")
    else:
        eva_log("Eva running in background-only mode. No user chat.")
        while EVA_RUNNING:
            time.sleep(1)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    # Start background thread
    background_thread = threading.Thread(target=eva_brain_loop, daemon=True)
    background_thread.start()

    # Start chat or remain background
    chat_with_eva()

    # Wait for background thread
    background_thread.join(timeout=1)
