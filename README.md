---
title: Faculty Chatbot
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# แชตบอตตอบคำถามจากเอกสารของคณะ

สร้างด้วยวิธี RAG  สำหรับรายวิชาปัญญาประดิษฐ์และการเรียนรู้ของเครื่อง สัปดาห์ที่ 8

## โครงไฟล์

```
app.py             โค้ดหลัก  ทำ RAG ครบสี่ขั้น
requirements.txt   รายการไลบรารี
docs/              เอกสารที่แชตบอตจะตอบจาก  ใส่ไฟล์ .txt เพิ่มได้
```

## ต้องตั้งค่าก่อนใช้

ไปที่แท็บ Settings ของ Space นี้ แล้วเพิ่ม Repository secret

| ชื่อ | ค่า |
|---|---|
| GOOGLE_API_KEY | กุญแจที่ได้จาก Google AI Studio |

จากนั้นกด Restart this Space
