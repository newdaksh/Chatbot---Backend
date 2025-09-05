import re
import os
import glob
import json
import tempfile
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, time

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# ====== NEW: simple in-memory "DB" for confirmed bookings ======
appointments: List[Dict[str, Any]] = []

# ====== Load doctors.json (keep the file in the same folder as project.py) ======
def _load_doctors() -> List[Dict[str, str]]:
    import os
    import json
    
    # Direct path to doctors.json
    doctors_path = os.path.join(os.path.dirname(__file__), "sample JSON", "doctors.json")
    
    if os.path.exists(doctors_path):
        with open(doctors_path, "r", encoding="utf-8") as f:
            doctors_data = json.load(f)
            print(f"✅ Loaded {len(doctors_data)} doctors from {doctors_path}")
            return doctors_data
    else:
        print(f"❌ File not found: {doctors_path}")
        return []

DOCTORS: List[Dict[str, str]] = _load_doctors()  # uses your uploaded file

# ====== Time helpers for “10:00 AM - 1:00 PM” style windows ======
_TIME_FMT_IN = "%I:%M %p"   # e.g., 04:30 PM
_TIME_FMT_OUT = "%I:%M %p"

def _parse_window(win: str) -> Tuple[time, time]:
    """Parse '10:00 AM - 1:00 PM' -> (time(10,0), time(13,0))."""
    parts = [p.strip() for p in win.split("-")]
    if len(parts) != 2:
        raise ValueError("Invalid window format")
    start = datetime.strptime(parts[0], _TIME_FMT_IN).time()
    end = datetime.strptime(parts[1], _TIME_FMT_IN).time()
    if end <= start:
        # e.g., "10:00 AM - 1:00 PM" is fine, but if a clinic wraps midnight you’d handle here
        pass
    return start, end

def _parse_user_time(s: str) -> time:
    s = s.strip()
    fmt_candidates = ["%I:%M %p", "%I %p", "%H:%M", "%H"]
    for fmt in fmt_candidates:
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            continue
    # heuristics: words
    s_low = s.lower()
    if "morning" in s_low:
        return time(10, 0)
    if "afternoon" in s_low:
        return time(14, 30)
    if "evening" in s_low:
        return time(18, 0)
    raise ValueError("Cannot parse time")

def _nearest_valid_time(preferred: time, start: time, end: time) -> time:
    """If preferred outside window, snap to closest bound."""
    if preferred < start:
        return start
    if preferred > end:
        return end
    return preferred

# ====== Matching helpers ======
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()

def _match_doctors(preferred_doctor: str, specialty: str, area_hint: str) -> List[Dict[str, str]]:
    """Priority:
       1) Exact doctor name (case-insensitive partial allowed)
       2) Specialty + area contains (e.g., 'Malviya Nagar')
       3) Specialty only
    """
    ndoc = _norm(preferred_doctor)
    nspec = _norm(specialty)
    narea = _norm(area_hint)

    # 1) doctor name match (partial)
    doc_matches = []
    if ndoc:
        for d in DOCTORS:
            if ndoc in _norm(d.get("name", "")):
                doc_matches.append(d)
    if doc_matches:
        return doc_matches

    # 2) specialty + area
    if nspec and narea:
        both = []
        for d in DOCTORS:
            if nspec in _norm(d.get("speciality", "")) and narea in _norm(d.get("location", "")):
                both.append(d)
        if both:
            return both

    # 3) specialty only
    if nspec:
        spec_only = []
        for d in DOCTORS:
            if nspec in _norm(d.get("speciality", "")):
                spec_only.append(d)
        if spec_only:
            return spec_only

    # fallback: area only
    if narea:
        in_area = []
        for d in DOCTORS:
            if narea in _norm(d.get("location", "")):
                in_area.append(d)
        if in_area:
            return in_area

    # else all
    return DOCTORS

# ====== Human-like follow-up flows ======
question_flows = {
    "medical_checkup": [
        "What is your age?",
        "Do you have any existing medical conditions?",
        "Are you currently taking any medications?",
        "Do you have any allergies?",
        "What symptoms are you experiencing?"
    ],
    "appointment_booking": [
        "Hi! I can book your appointment in a minute. May I have your full name, please?",
        "Thanks. Could you share a phone number where we can reach you? (WhatsApp or mobile)",
        "Which date works best for you? (e.g., 06 Sep 2025; you can also say ‘any day next week’)",
        "What time would you prefer—morning, afternoon, evening, or a specific time like 4:30 PM?",
        "Do you have a preferred doctor, or should I book by specialty only?",
        "Would you like an in-clinic visit or a video consultation?",
        "Which area/clinic in Jaipur is most convenient for you? (e.g., Malviya Nagar, Vaishali Nagar)",
        "In one line, what’s the main reason for your visit? (Helps the doctor prepare)",
        "Do you want to use health insurance? If yes, which provider?",
        "Great! Anything else I should note—like accessibility needs or bringing a family member?"
    ]
}

# ====== Natural, model-directed follow-up bank for appointment booking ======
FOLLOWUP_BANK = [
  {"key":"full_name","ask":"May I have your full name?","type":"name","required":True},
  {"key":"phone","ask":"A phone number for updates (WhatsApp or mobile)?","type":"phone","required":True},
  {"key":"visit_mode","ask":"Would you like an in-clinic visit or a video consultation?","type":"enum","options":["in-clinic","clinic","video","online"],"required":True},
  {"key":"area","ask":"Which area/clinic in Jaipur suits you? (e.g., Malviya Nagar, Vaishali Nagar)","type":"area","required":True,"skip_if":{"visit_mode":["video","online"]}},
  {"key":"date","ask":"Which date works best for you? (e.g., 06 Sep 2025, today, tomorrow)","type":"date","required":True},
  {"key":"time","ask":"What time would you prefer? (e.g., 4:30 PM or 16:30)","type":"time","required":True},
  {"key":"preferred_doctor","ask":"Any preferred doctor, or should I book by specialty only?","type":"text","required":False},
  {"key":"reason","ask":"In one line, what’s the main reason for the visit? (Helps the doctor prepare)","type":"short_text","required":True},
  {"key":"insurance","ask":"Do you want to use health insurance? If yes, which provider?","type":"text","required":False},
  {"key":"extra_notes","ask":"Anything else I should note—access needs or bringing a family member?","type":"text","required":False},
]

POLICY_PROMPT = """
You are a helpful medical appointment intake agent.
Goal: collect all REQUIRED fields naturally, with minimal questions, using context from prior answers.

You have:
- followup_bank: a list of possible follow-up questions with keys and metadata
- collected: dict of fields already captured (normalized if possible)
- last_user_message: the user’s latest reply

You must:
1) Understand the last_user_message, extract/normalize any fields it answers.
2) Decide the *best next step*:
   - "ask": choose ONE next follow-up from followup_bank (respect skip_if and already collected fields)
   - "confirm": if user provided ambiguous/contradictory info, ask a short clarification
   - "finalize": if all required fields are present and consistent
3) Produce STRICT JSON only:

{
  "action": "ask" | "confirm" | "finalize",
  "updates": { "<field_key>": "<normalized_value>", ... },
  "next_question": "<string or empty>",
  "missing_required": ["<keys still needed>"],
  "reason": "<very short reason>"
}

Normalization hints:
- phone: Indian mobile; allow +91; return digits-only or +91##########.
- date: accept “today/tomorrow”; return ISO YYYY-MM-DD when possible.
- time: return "HH:MM AM/PM".
- enum: map synonyms (e.g., "online" -> "video").
- short_text: ≤ 15 words.
If "visit_mode" is "video", treat "area" as not required (skip).
If user answers multiple fields in one line, fill them all in "updates".
Keep questions short and friendly.
"""

import json as _json

def _skip_field(field, collected):
    rule = field.get("skip_if")
    if not rule: return False
    for k, vals in rule.items():
        v = (collected.get(k) or "").lower()
        if v in [x.lower() for x in vals]:
            return True
    return False

def _missing_required(collected):
    miss = []
    for f in FOLLOWUP_BANK:
        if f.get("required") and f["key"] not in collected and not _skip_field(f, collected):
            miss.append(f["key"])
    return miss

def _choose_first_missing_question(collected):
    for f in FOLLOWUP_BANK:
        if f["key"] in collected: 
            continue
        if _skip_field(f, collected):
            continue
        return f["ask"]
    return ""

def natural_turn(model, collected, last_user_message):
    payload = {
        "followup_bank": FOLLOWUP_BANK,
        "collected": collected,
        "last_user_message": last_user_message,
    }
    prompt = POLICY_PROMPT + "\n\nINPUT:\n" + _json.dumps(payload, ensure_ascii=False)
    resp = model.invoke(prompt).content.strip()

    # Robust JSON extraction
    try:
        start, end = resp.find("{"), resp.rfind("}")
        data = json.loads(resp[start:end+1]) if start!=-1 and end!=-1 else {}
    except Exception:
        data = {}

    action = data.get("action") or ("finalize" if len(_missing_required(collected)) == 0 else "ask")
    updates = data.get("updates") or {}
    collected.update(updates)

    if action == "finalize":
        if _missing_required(collected):
            return collected, _choose_first_missing_question(collected)
        return collected, None

    nq = data.get("next_question")
    if not nq:
        nq = _choose_first_missing_question(collected)
        if not nq:
            return collected, None
    return collected, nq

# Tracks flow per session
user_flows = {}  # appointment_booking uses {"answers": {}}; other flows may use list-based answers.

# ====== Helper function to extract context from recent conversation ======
def _extract_recent_context(session_id: str, keywords: List[str]) -> str:
    """Extract mentions of keywords from recent conversation history."""
    conversation_history = get_conversation_history(session_id)
    if not conversation_history:
        return ""
    
    mentions = []
    # Look through last 6 messages (3 exchanges)
    for msg in conversation_history[-6:]:
        content = msg["content"].lower()
        for keyword in keywords:
            if keyword in content:
                mentions.append(keyword)
                break
    
    return mentions[-1] if mentions else ""

# ====== PDF helpers (unchanged) ======
def process_pdf(file_content: bytes, filename: str, session_id: str = "default") -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, embeddings)
        os.makedirs("faiss_indexes", exist_ok=True)
        vectorstore.save_local(f"faiss_indexes/{session_id}")
        vector_stores[session_id] = vectorstore
        pdf_contents[session_id] = {'filename': filename, 'num_pages': len(documents), 'num_chunks': len(splits)}
        os.unlink(temp_file_path)
        return f"PDF '{filename}' processed: {len(documents)} pages, {len(splits)} chunks."
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def preload_pdfs_from_folder(folder_path: str) -> str:
    global global_vector_store, preloaded_pdf_info
    try:
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        if not pdf_files:
            return "No PDFs found in the specified folder."
        
        documents = []
        filenames = []
        total_pages = 0
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)
            filenames.append(os.path.basename(pdf_file))
            total_pages += len(docs)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        splits = text_splitter.split_documents(documents)
        global_vector_store = FAISS.from_documents(splits, embeddings)
        os.makedirs("faiss_indexes", exist_ok=True)
        global_vector_store.save_local("faiss_indexes/global")
        preloaded_pdf_info = {"filenames": filenames, "num_pages": total_pages, "num_chunks": len(splits)}
        return (f"Preloaded PDFs: {len(pdf_files)} files -> "
                f"{', '.join(filenames)} | {total_pages} pages, {len(splits)} chunks.")
    except Exception as e:
        global_vector_store = None
        preloaded_pdf_info = {"filenames": [], "num_pages": 0, "num_chunks": 0}
        return f"Error preloading PDFs: {str(e)}"

def get_conversation_history(session_id="default", max_messages=25):
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    history = conversation_memory[session_id]
    return history[-max_messages*2:] if len(history) > max_messages*2 else history

def add_to_conversation_history(session_id="default", user_message="", assistant_message=""):
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    conversation_memory[session_id].extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ])

def build_context_from_history(session_id="default") -> str:
    # Build a readable context from the last few messages to help the LLM
    history = get_conversation_history(session_id)
    context = "Here are some relevant details from the prior conversation:\n"
    if not history:
        return ""
    for msg in history[-10:]:
        if msg["role"] == "user":
            context += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            context += f"Assistant: {msg['content']}\n"
    context += "\nCurrent conversation:\n"
    return context

def search_pdf_content(query: str, session_id: str = "default", k: int = 3) -> str:
    contexts = []
    try:
        if global_vector_store is not None:
            kb_docs = global_vector_store.similarity_search(query, k=k)
            if kb_docs:
                context = "\n\nRelevant information from the knowledge base (preloaded PDFs):\n"
                for i, doc in enumerate(kb_docs, 1):
                    src = (doc.metadata or {}).get("source", "preloaded PDF")
                    context += f"[KB {i} - {src}]: {doc.page_content[:500]}...\n\n"
                contexts.append(context)
    except Exception as e:
        contexts.append(f"\nError searching knowledge base: {str(e)}\n")

    if session_id in vector_stores:
        try:
            vectorstore = vector_stores[session_id]
            docs = vectorstore.similarity_search(query, k=k)
            if docs:
                pdf_info = pdf_contents.get(session_id, {})
                name = pdf_info.get('filename', 'uploaded PDF')
                context = f"\n\nDocument Context (from '{name}'):\n"
                for i, doc in enumerate(docs, 1):
                    context += f"[Excerpt {i}]: {doc.page_content[:500]}...\n\n"
                contexts.append(context)
        except Exception as e:
            contexts.append(f"\nError searching uploaded PDF context: {str(e)}\n")

    return "\n".join(contexts)

# ====== Appointment finalization (unchanged core) ======
def _finalize_appointment(answers: List[str]) -> str:
    """
    answers order expected:
      0 full_name
      1 phone
      2 date
      3 time
      4 preferred_doctor (or specialty)
      5 visit_mode (clinic/video)
      6 area
      7 reason
      8 insurance
      9 extra_notes
    """
    try:
        full_name         = answers[0].strip()
        phone             = answers[1].strip()
        pref_date_str     = answers[2].strip()
        pref_time_str     = answers[3].strip()
        preferred_doctor  = answers[4].strip()         # can be empty or specialty
        visit_mode        = answers[5].strip()         # clinic or video
        area              = answers[6].strip()
        reason            = answers[7].strip()
        insurance         = answers[8].strip()
        extra_notes       = answers[9].strip()
    except Exception:
        return "Got your details. One or two fields look incomplete—please retype the missing parts briefly."

    # Parse date (be flexible)
    today = datetime.now().date()
    s = pref_date_str.lower()
    if "today" in s:
        pref_date = today
    elif "tomorrow" in s:
        pref_date = today.replace(day=today.day + 1)
    else:
        try:
            # try common formats
            for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                try:
                    pref_date = datetime.strptime(pref_date_str, fmt).date()
                    break
                except Exception:
                    continue
            else:
                # fallback: keep as string
                pref_date = None
        except Exception:
            pref_date = None

    # Parse preferred time
    try:
        pref_time = _parse_user_time(pref_time_str)
    except Exception:
        return "Thanks. Please share a clear time like ‘10:30 AM’ or ‘16:00’, and I’ll lock the slot."

    # Match doctors
    # If user typed a specialty instead of doctor name, it still works.
    specialty_guess = preferred_doctor  # can be either
    matches = _match_doctors(preferred_doctor, specialty_guess, area)

    if not matches:
        return ("I couldn’t find a matching doctor/area from our list. "
                "Try a specialty (e.g., Cardiologist) and a Jaipur area like Malviya Nagar. "
                "We’ll book the closest available slot from the timings.")

    # Choose first suitable doctor with a valid window
    chosen = None
    confirmed_time = None
    for d in matches:
        try:
            start, end = _parse_window(d["timings"])
            pick = _nearest_valid_time(pref_time, start, end)
            # We accept the pick even if it snaps to boundary
            chosen = d
            confirmed_time = pick
            break
        except Exception:
            continue

    if not chosen:
        chosen = matches[0]
        try:
            start, end = _parse_window(chosen["timings"])
            confirmed_time = _nearest_valid_time(pref_time, start, end)
        except Exception:
            confirmed_time = pref_time

    final_date_str = pref_date.isoformat() if pref_date else pref_date_str
    final_time_str = confirmed_time.strftime(_TIME_FMT_OUT)

    booking = {
        "full_name": full_name,
        "phone": phone,
        "date": final_date_str,
        "time": final_time_str,
        "preferred_doctor_or_specialty": preferred_doctor,
        "visit_mode": visit_mode,
        "area": area,
        "reason": reason,
        "insurance": insurance,
        "extra_notes": extra_notes,
        "assigned_doctor": {
            "name": chosen["name"],
            "speciality": chosen["speciality"],
            "location": chosen["location"],
            "timings": chosen["timings"],
            "phone": chosen["phone"]
        }
    }
    appointments.append(booking)

    # Short, human confirmation (25–40 words)
    return (f"All set, {full_name}! I’ve booked {chosen['name']} ({chosen['speciality']}) on "
            f"{booking['date']} at {booking['time']} — {visit_mode}. Clinic: {chosen['location']}. "
            f"For help, call {chosen['phone']}. I’ve noted: {reason}.")

# ====== Basic LLM + embeddings (unchanged) ======
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)
model = ChatHuggingFace(llm=llm)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

conversation_memory = {}
vector_stores = {}
pdf_contents = {}

global_vector_store = None
preloaded_pdf_info = {"filenames": [], "num_pages": 0, "num_chunks": 0}

# ====== Main chat ======
def chat(message, history=None, session_id="default"):
    # 1) Are we mid-flow?
    if session_id in user_flows:
        flow_data = user_flows[session_id]
        flow_name = flow_data["flow"]

        if flow_name == "appointment_booking":
            state = user_flows.get(session_id) or {"answers": {}, "flow": "appointment_booking"}
            collected, next_q = natural_turn(model, state.get("answers", {}), message)
            state["answers"] = collected
            user_flows[session_id] = state

            if next_q is None:
                # prepare ordered answers for _finalize_appointment
                order = ["full_name","phone","date","time","preferred_doctor","visit_mode","area","reason","insurance","extra_notes"]
                ordered = []
                for k in order:
                    # skip area if visit_mode says video/online
                    if k == "area":
                        vm = (collected.get("visit_mode","")).lower()
                        if vm in ["video","online"]:
                            ordered.append("")
                            continue
                    ordered.append(collected.get(k, ""))
                del user_flows[session_id]
                return _finalize_appointment(ordered)
            else:
                return next_q

        # Fallback: other list-based flows (existing behavior)
        flow_data["answers"].append(message)
        flow_data["step"] += 1

        if flow_data["step"] < len(question_flows[flow_name]):
            next_q = question_flows[flow_name][flow_data["step"]]
            return next_q
        else:
            if flow_name == "appointment_booking":
                answers = flow_data["answers"]
                del user_flows[session_id]
                return _finalize_appointment(answers)
            else:
                summary = "\n".join(
                    f"{q} {a}" for q, a in zip(question_flows[flow_name], flow_data["answers"])
                )
                del user_flows[session_id]
                return f"✅ Thanks! Here’s a quick summary:\n{summary}"

    # 2) Flow triggers
    triggers = {
        "medical_checkup": ["medical checkup", "start checkup", "begin checkup"],
        "appointment_booking": ["book appointment", "appointment", "schedule appointment"]
    }
    for flow_name, keywords in triggers.items():
        if any(t in message.lower() for t in keywords):
            user_flows[session_id] = {"flow": flow_name, "step": 0, "answers": {} }
            # Ask natural first question for appointment booking
            if flow_name == "appointment_booking":
                first_question = FOLLOWUP_BANK[0]["ask"]
            else:
                first_question = question_flows[flow_name][0]
            return first_question

    # 3) Doctor search triggers - NEW SECTION
    doctor_keywords = [
        "find doctor", "doctors in", "cardiologist", "dermatologist", "neurologist",
        "pediatrician", "gynecologist", "orthopedic", "psychiatrist", "ophthalmologist",
        "ent specialist", "general physician", "urologist", "oncologist", "endocrinologist",
        "dentist", "rheumatologist", "nephrologist", "pulmonologist", "nutritionist",
        "hematologist", "physiotherapist", "gastroenterologist", "allergist", "plastic surgeon",
        "diabetologist", "sports medicine", "show me doctors", "list doctors", "recommend doctor"
    ]
    
    if any(keyword in message.lower() for keyword in doctor_keywords):
        # Extract specialty and area from the message
        msg_lower = message.lower()
        
        # Extract area
        area_hint = ""
        jaipur_areas = [
            "malviya nagar", "vaishali nagar", "mansarovar", "bani park", "jagatpura",
            "sodala", "adarsh nagar", "tonk road", "jhotwara", "pratap nagar",
            "raja park", "ajmer road", "bapu nagar", "chitrakoot", "shastri nagar",
            "civil lines", "mi road", "khatipura", "sanganer", "kalwar road",
            "c-scheme"
        ]
        
        for area in jaipur_areas:
            if area in msg_lower:
                area_hint = area
                break
        
        # Extract specialty
        specialty_keywords = [
            "cardiologist", "dermatologist", "neurologist", "pediatrician", "gynecologist",
            "orthopedic", "psychiatrist", "ophthalmologist", "ent", "general physician",
            "urologist", "oncologist", "endocrinologist", "dentist", "rheumatologist",
            "nephrologist", "pulmonologist", "nutritionist", "hematologist", "physiotherapist",
            "gastroenterologist", "allergist", "plastic surgeon", "diabetologist",
            "sports medicine"
        ]
        
        specialty_mentioned = ""
        for spec in specialty_keywords:
            if spec in msg_lower:
                specialty_mentioned = spec
                break
        
        # Build response based on user mention
        if specialty_mentioned and area_hint:
            matches = _match_doctors("", specialty_mentioned, area_hint)
            if matches:
                response = f"Here are some {specialty_mentioned.title()}s in {area_hint.title()}:\n\n"
                for i, doctor in enumerate(matches[:5], 1):
                    response += f"{i}. **{doctor['name']}** - {doctor['speciality']}\n"
                    response += f"   📍 {doctor['location']}\n"
                    response += f"   🕒 {doctor['timings']}\n"
                    response += f"   📞 {doctor['phone']}\n\n"
                response += "Would you like me to book an appointment with any of these doctors?"
            else:
                response = f"I couldn't find any {specialty_mentioned.title()}s specifically in {area_hint.title()}. Here are some nearby options:\n\n"
                # Show first 3 doctors from any area
                if DOCTORS:
                    for i, doctor in enumerate(DOCTORS[:3], 1):
                        response += f"{i}. **{doctor['name']}** - {doctor['speciality']}\n"
                        response += f"   📍 {doctor['location']}\n"
                        response += f"   🕒 {doctor['timings']}\n"
                        response += f"   📞 {doctor['phone']}\n\n"
                else:
                    response += "No doctors are currently listed."
            return response
        
        elif specialty_mentioned:
            # Show top options across Jaipur
            matches = _match_doctors("", specialty_mentioned, "")
            if matches:
                response = f"Top {specialty_mentioned.title()}s in Jaipur:\n\n"
                for i, doctor in enumerate(matches[:5], 1):
                    response += f"{i}. **{doctor['name']}** - {doctor['speciality']}\n"
                    response += f"   📍 {doctor['location']}\n"
                    response += f"   🕒 {doctor['timings']}\n"
                    response += f"   📞 {doctor['phone']}\n\n"
                response += "Would you like me to book an appointment with any of these doctors?"
            else:
                response = f"I couldn't find any {specialty_mentioned.title()}s across Jaipur. Try a different specialty or ask me to show all available doctors."
            return response
        
        elif area_hint:
            response = f"Let me show you all available doctors in {area_hint.title()}:\n\n"
            area_matches = _match_doctors("", "", area_hint)
            if area_matches:
                for i, doctor in enumerate(area_matches[:5], 1):
                    response += f"{i}. **{doctor['name']}** - {doctor['speciality']}\n"
                    response += f"   📍 {doctor['location']}\n"
                    response += f"   🕒 {doctor['timings']}\n"
                    response += f"   📞 {doctor['phone']}\n\n"
                response += "Would you like me to book an appointment with any of them?"
            else:
                response = f"I couldn't find any doctors specifically in {area_hint.title()}. Here are some nearby options:\n\n"
                # Show first 3 doctors from any area
                if DOCTORS:
                    for i, doctor in enumerate(DOCTORS[:3], 1):
                        response += f"{i}. **{doctor['name']}** - {doctor['speciality']}\n"
                        response += f"   📍 {doctor['location']}\n"
                        response += f"   🕒 {doctor['timings']}\n"
                        response += f"   📞 {doctor['phone']}\n\n"
            return response
        
        else:
            response = "I have a list of doctors in Jaipur. Could you please specify:\n\n"
            response += "1. What specialty are you looking for? (e.g., Cardiologist, Dermatologist)\n"
            response += "2. Which area do you prefer? (e.g., Malviya Nagar, Vaishali Nagar)\n\n"
            response += "Or say 'appointment' to start booking."
            return response

    # 4) General Q&A over PDFs
    kb_context = search_pdf_content(message, session_id=session_id, k=3)
    prompt = (build_context_from_history(session_id)
              + kb_context
              + f"\n\nUser asked: {message}\n\nProvide a helpful, concise answer.\n")
    try:
        resp = model.invoke(prompt)
        answer = resp.content.strip()
    except Exception as e:
        answer = f"LLM error: {str(e)}"
    add_to_conversation_history(session_id, user_message=message, assistant_message=answer)
    return answer

# ====== Utilities ======
def get_appointments() -> List[Dict[str, Any]]:
    return appointments

# --- Compatibility wrappers expected by main.py ---

# conversation memory the file already uses
try:
    conversation_memory
except NameError:
    conversation_memory = {}

def clear_conversation_history(session_id: str = "default"):
    """Compat alias for main.py. Clears history for a session."""
    if session_id in conversation_memory:
        del conversation_memory[session_id]
    return "Conversation history cleared!"

# PDF/vectorstore globals the file already uses
try:
    preloaded_pdf_info
except NameError:
    preloaded_pdf_info = {"filenames": [], "num_pages": 0, "num_chunks": 0}

def get_pdf_info(session_id: str = "default"):
    """
    Compat for main.py: Return info on the uploaded (per-session) PDF
    and the globally preloaded PDFs if present.
    """
    # Per-session info (if your code tracks it)
    pdf_meta = {}
    try:
        # If you track uploads per session as pdf_contents[session_id]
        pdf_meta = pdf_contents.get(session_id, {})
    except Exception:
        pass

    return {
        "session_pdf": pdf_meta,              # e.g., {'filename': 'x.pdf', 'num_pages': 3, 'num_chunks': 12}
        "preloaded_pdfs": preloaded_pdf_info, # e.g., {'filenames': [...], 'num_pages': N, 'num_chunks': M}
    }

def preload_global_pdfs(pdf_dir: str = "pdfs"):
    """
    Compat for main.py: Preload all PDFs from a folder into a global FAISS index.
    Calls your existing 'preload_pdfs_from_folder' if present; otherwise no-op.
    """
    if 'preload_pdfs_from_folder' in globals():
        return preload_pdfs_from_folder(pdf_dir)
    return f"No global preloader available. Ensure 'preload_pdfs_from_folder' exists. Requested dir: {pdf_dir}"


def clear_conversation(session_id="default"):
    if session_id in conversation_memory:
        del conversation_memory[session_id]
    return "Conversation history cleared!"

def clear_pdf_data(session_id="default"):
    if session_id in vector_stores:
        del vector_stores[session_id]
    if session_id in pdf_contents:
        del pdf_contents[session_id]
    return "PDF data cleared!"

def get_conversation_summary(session_id="default"):
    history = get_conversation_history(session_id)
    if not history:
        return "No conversation history found."
    summary = f"Conversation contains {len(history)} messages:\n"
    for i, msg in enumerate(history[-20:], 1):
        who = "U" if msg["role"] == "user" else "A"
        summary += f"{i:02d} [{who}] {msg['content']}\n"
    return summary
