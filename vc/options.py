"""ตัวเลือกตอนรันหนึ่งครั้ง (runtime options) ที่แกนกลางเป็นเจ้าของนิยาม

เดิม `VoiceChat` รับ Namespace ของตัวแยก argument บรรทัดคำสั่งมาตรง ๆ ทำให้ชั้น
แกนกลางผูกกับรูปแบบ CLI และฝั่งเว็บต้อง "ปลอม" Namespace ขึ้นมาเพื่อเรียกใช้ (P3-21)
ตอนนี้แกนกลางประกาศสัญญาของตัวเองไว้ที่นี่ ผู้เรียกฝ่ายไหนก็แค่สร้าง dataclass นี้

ต่างจาก `Config` ตรงที่ `Config` = ค่าตั้งจาก `.env` (คงที่ทั้งการติดตั้ง) แต่
`RuntimeOptions` = การตัดสินใจของการรันครั้งนี้ครั้งเดียว
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeOptions:
    no_mic: bool = False   # True = โหมดพิมพ์ ไม่เปิดไมโครโฟน
    greet: bool = True     # ทักทายก่อนเริ่มฟังไหม
