"""
แชตบอตตอบคำถามจากเอกสารของคณะ  ด้วยวิธี RAG
สำหรับวางบน Hugging Face Spaces แบบ Gradio  ซึ่งใช้ได้ฟรี

สี่ขั้นตอนตามที่เรียนในคาบ
  ขั้นที่ 1  หั่นเอกสารเป็นชิ้นเล็ก        -> build_chunks()
  ขั้นที่ 2  แปลงทุกชิ้นเป็นเวกเตอร์       -> build_index()
  ขั้นที่ 3  ค้นหาชิ้นที่ใกล้ที่สุด        -> search()
  ขั้นที่ 4  ประกอบคำสั่งแล้วส่งให้ LLM    -> make_prompt()
"""

import os
import glob
import numpy as np
import gradio as gr
import google.generativeai as genai

# ---------------------------------------------------------------
# Hugging Face Spaces แบบฟรีบังคับให้รันบน ZeroGPU
# ซึ่งกำหนดว่าต้องมีฟังก์ชันที่ประกาศว่าใช้ GPU อย่างน้อยหนึ่งตัว
# ไม่งั้นจะขึ้นว่า  No @spaces.GPU function detected during startup
#
# งานของเราไม่ได้ใช้ GPU เลย เพราะเรียก Google ทำให้หมด
# จึงประกาศฟังก์ชันเปล่าไว้หนึ่งตัวเพื่อให้ผ่านเงื่อนไข
# ---------------------------------------------------------------
try:
    import spaces

    @spaces.GPU(duration=10)
    def _gpu_placeholder():
        """ประกาศไว้ให้ ZeroGPU ตรวจเจอ  ไม่ได้ใช้งานจริง"""
        return "ok"

    print("พบแพ็กเกจ spaces  ประกาศฟังก์ชัน GPU ไว้แล้ว")
except ImportError:
    print("ไม่มีแพ็กเกจ spaces  ถือว่ารันนอก ZeroGPU ซึ่งไม่เป็นไร")

# ---------------------------------------------------------------
# ตั้งค่า  แก้ได้สามบรรทัดนี้
# ---------------------------------------------------------------
CHUNK_SIZE = 500      # จำนวนตัวอักษรต่อหนึ่งชิ้น
OVERLAP    = 100      # ให้ชิ้นเหลื่อมกันกี่ตัวอักษร  กันความหมายขาดกลางประโยค
TOP_K      = 3        # หยิบเอกสารมากี่ชิ้นต่อหนึ่งคำถาม
# ---------------------------------------------------------------

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# ---------------------------------------------------------------
# เลือกรุ่นโมเดลอัตโนมัติ
#
# บทเรียนจากการใช้งานจริง
#   การถามว่า "มีรุ่นอะไรบ้าง" ยังไม่พอ  เพราะบางรุ่นอยู่ในรายการ
#   แต่เรียกใช้จริงไม่ได้ เช่นถูกปิดสำหรับผู้ใช้ใหม่
#   เราจึงต้องลองเรียกจริงด้วยคำถามสั้น ๆ  ตัวไหนผ่านค่อยใช้ตัวนั้น
# ---------------------------------------------------------------
import re


def rank(name):
    """คะแนนยิ่งน้อยยิ่งอยากได้ก่อน"""
    s = 0
    if "gemini" in name:  s -= 100          # อยากได้ gemini มากกว่า gemma
    if "flash" in name:   s -= 50           # flash เร็วและถูกกว่า pro
    for bad in ("preview", "tts", "image", "vision", "thinking", "exp"):
        if bad in name:   s += 200          # หลีกเลี่ยงรุ่นทดลองและรุ่นเฉพาะทาง
    m = re.search(r"(\d+)\.(\d+)", name)   # เลขรุ่นสูงกว่าได้เปรียบ
    if m:
        s -= int(m.group(1)) * 10 + int(m.group(2))
    return s


def list_names(method):
    return [m.name for m in genai.list_models()
            if method in getattr(m, "supported_generation_methods", [])]


def pick_chat():
    avail = list_names("generateContent")
    for name in sorted(avail, key=rank)[:6]:        # ลองไม่เกินหกตัว กันเสียเวลา
        try:
            genai.GenerativeModel(name).generate_content("ตอบว่า ok")
            print("  ทดสอบผ่าน :", name)
            return name, avail
        except Exception as e:
            print("  ทดสอบไม่ผ่าน :", name, "|", str(e).split("\n")[0][:90])
    print("ไม่มีรุ่นตอบคำถามที่ใช้งานได้เลย")
    raise SystemExit(1)


def pick_embed():
    avail = list_names("embedContent")
    for name in sorted(avail, key=rank)[:6]:
        try:
            genai.embed_content(model=name, content="ทดสอบ", task_type="retrieval_document")
            print("  ทดสอบผ่าน :", name)
            return name, avail
        except Exception as e:
            print("  ทดสอบไม่ผ่าน :", name, "|", str(e).split("\n")[0][:90])
    print("ไม่มีรุ่นแปลงเวกเตอร์ที่ใช้งานได้เลย")
    raise SystemExit(1)


print("=" * 64)
print("กำลังทดสอบรุ่นแปลงเวกเตอร์")
EMBED_MODEL, embed_list = pick_embed()
print("กำลังทดสอบรุ่นตอบคำถาม")
CHAT_MODEL, chat_list = pick_chat()

print()
print("รุ่นที่เลือกใช้")
print("  แปลงเวกเตอร์ :", EMBED_MODEL)
print("  ตอบคำถาม     :", CHAT_MODEL)
print()
print("รุ่นทั้งหมดที่บัญชีนี้เห็น")
print("  แปลงเวกเตอร์ :", ", ".join(n.replace("models/", "") for n in embed_list))
print("  ตอบคำถาม     :", ", ".join(n.replace("models/", "") for n in chat_list))
print("=" * 64)


# ===============================================================
# ขั้นที่ 1  หั่นเอกสารเป็นชิ้นเล็ก
# ===============================================================
def build_chunks(folder="docs"):
    chunks = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        name = os.path.basename(path)
        text = open(path, encoding="utf-8").read().strip()

        start, part = 0, 1
        while start < len(text):
            chunks.append({
                "source": name,
                "part": part,
                "text": text[start:start + CHUNK_SIZE],
            })
            start += CHUNK_SIZE - OVERLAP     # เลื่อนไปโดยให้เหลื่อมกัน
            part += 1
    return chunks


# ===============================================================
# ขั้นที่ 2  แปลงทุกชิ้นเป็นเวกเตอร์  ทำครั้งเดียวตอนเปิดเครื่อง
# ===============================================================
def embed(texts, task):
    out = []
    for t in texts:
        r = genai.embed_content(model=EMBED_MODEL, content=t, task_type=task)
        out.append(r["embedding"])
    return np.array(out, dtype="float32")


def build_index(chunks):
    vecs = embed([c["text"] for c in chunks], "retrieval_document")
    # ทำให้ความยาวเป็นหนึ่ง  จะได้เทียบความใกล้ด้วยการคูณกันตรง ๆ
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


print("กำลังหั่นเอกสาร...")
CHUNKS = build_chunks()

# ตรวจก่อนไปต่อ  ถ้าไม่มีเอกสารจะพังตอนแปลงเวกเตอร์ และอ่าน error ไม่รู้เรื่อง
if not CHUNKS:
    print("\n" + "=" * 60)
    print("ไม่พบไฟล์ .txt ในโฟลเดอร์ docs")
    print("=" * 60)
    print("สิ่งที่เห็นในโฟลเดอร์ปัจจุบัน :", sorted(os.listdir(".")))
    if os.path.isdir("docs"):
        print("ในโฟลเดอร์ docs มี       :", sorted(os.listdir("docs")))
    else:
        print("ไม่มีโฟลเดอร์ชื่อ docs เลย")
    print()
    print("วิธีแก้")
    print("  1 ไปที่แท็บ Files ของ Space")
    print("  2 กด Add file แล้ว Upload files")
    print("  3 ลากไฟล์ .txt เข้าไป แล้วแก้ชื่อไฟล์ให้ขึ้นต้นด้วย docs/")
    print("    เช่น  docs/01-การลา.txt")
    print("  4 กด Commit แล้ว Restart this Space")
    print("=" * 60)
    raise SystemExit(1)

print(f"ได้ {len(CHUNKS)} ชิ้น จาก {len(set(c['source'] for c in CHUNKS))} ไฟล์")

print("กำลังแปลงเป็นเวกเตอร์...")
INDEX = build_index(CHUNKS)
print("พร้อมใช้งาน")


# ===============================================================
# ขั้นที่ 3  ค้นหาชิ้นที่ใกล้ที่สุด
# ===============================================================
def search(question, k=TOP_K):
    q = embed([question], "retrieval_query")[0]
    q = q / np.linalg.norm(q)

    score = INDEX @ q                      # ยิ่งค่ามาก ยิ่งใกล้กัน
    top = np.argsort(score)[::-1][:k]

    return [{
        "source": CHUNKS[i]["source"],
        "part": CHUNKS[i]["part"],
        "text": CHUNKS[i]["text"],
        "score": round(float(score[i]), 3),
    } for i in top]


# ===============================================================
# ขั้นที่ 4  ประกอบคำสั่งแล้วส่งให้ LLM
# ===============================================================
def make_prompt(question, found):
    docs = "\n\n".join(
        f"[ชิ้นที่ {i+1}  จากไฟล์ {f['source']}]\n{f['text']}"
        for i, f in enumerate(found)
    )
    return f"""คุณคือผู้ช่วยตอบคำถามของคณะบริหารธุรกิจ
ให้ตอบโดยใช้เฉพาะข้อมูลในเอกสารด้านล่างเท่านั้น

ถ้าเอกสารไม่มีคำตอบ ให้ตอบว่า "ไม่พบข้อมูลนี้ในเอกสารของคณะ"
แล้วแนะนำให้ติดต่อฝ่ายวิชาการ  ห้ามเดา ห้ามเติมข้อมูลจากความรู้ทั่วไป

ตอบเป็นภาษาไทย สั้น กระชับ

เอกสารที่ค้นเจอ
{docs}

คำถามจากผู้ใช้
{question}"""


# ===============================================================
# ส่วนที่ผู้ใช้เห็น
# ===============================================================
def respond(message, history):
    q = (message or "").strip()
    if not q:
        return "กรุณาพิมพ์คำถาม"

    found = search(q)
    model = genai.GenerativeModel(CHAT_MODEL)
    reply = model.generate_content(make_prompt(q, found))

    refs = "\n".join(
        f"- {f['source']}  ส่วนที่ {f['part']}  ความใกล้เคียง {f['score']}"
        for f in found
    )
    return f"{reply.text}\n\n---\n**อ้างอิงจาก**\n{refs}"


HEADER = """
### ผู้ช่วยตอบคำถาม คณะบริหารธุรกิจ

**นี่คือผู้ช่วยอัตโนมัติ ไม่ใช่เจ้าหน้าที่**
คำตอบอาจคลาดเคลื่อน กรุณาตรวจสอบกับเอกสารต้นฉบับก่อนใช้อ้างอิง
ต้องการคุยกับเจ้าหน้าที่จริง ติดต่อฝ่ายวิชาการ โทร 0-2000-0002
"""

EXAMPLES = [
    "ลาป่วยกี่วันต้องมีใบรับรองแพทย์",
    "ถอนรายวิชาแล้วติด W ไหม",
    "ทุนเรียนดีต้องเกรดเท่าไร",
    "ขอใบแสดงผลการเรียนกี่วันได้",
]

# Gradio แต่ละรุ่นรับพารามิเตอร์ไม่เหมือนกัน
# จึงค่อย ๆ ตัดตัวที่รุ่นนั้นไม่รู้จักออกทีละตัว  จะได้เก็บลูกเล่นไว้ให้มากที่สุด
def build_ui():
    want = {
        "type": "messages",
        "theme": gr.themes.Soft(),
        "examples": EXAMPLES,
        "description": HEADER,
    }
    kw = dict(want)
    while True:
        try:
            ui = gr.ChatInterface(fn=respond, **kw)
            print("สร้างหน้าจอแชตสำเร็จ  ใช้ได้:", sorted(kw) or "ไม่มีตัวเสริมเลย")
            dropped = sorted(set(want) - set(kw))
            if dropped:
                print("รุ่นนี้ไม่รองรับ จึงตัดออก:", dropped)
            return ui
        except TypeError as e:
            msg = str(e)
            bad = next((k for k in list(kw) if f"'{k}'" in msg), None)
            if bad is None:
                print("สร้างไม่ผ่านและหาสาเหตุไม่ได้ :", msg)
                return gr.ChatInterface(fn=respond)
            kw.pop(bad)


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
