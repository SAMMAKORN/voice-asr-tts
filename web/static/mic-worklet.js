/* แปลงเสียงไมค์ของเบราว์เซอร์ (มักเป็น 48 kHz float) ให้เป็นเฟรม
   16 kHz / 20 ms int16 ตามที่ฝั่งเซิร์ฟเวอร์และ ASR ต้องการ

   ดาวน์แซมเปิลด้วยการเฉลี่ยช่วง (box average) แทนการทิ้งตัวอย่างตรง ๆ
   เพื่อกันเสียงแปลกจาก aliasing ซึ่งทำให้ค่า RMS ที่ VAD ใช้เพี้ยน */

const CAP = 16384;   // บัฟเฟอร์ขาเข้าสูงสุด (~340 ms ที่ 48 kHz)

class MicProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions || {};
    this.frame = o.frameSamples || 320;          // 20 ms ที่ 16 kHz
    this.step = sampleRate / (o.targetSr || 16000);

    this.buf = new Float32Array(CAP);
    this.len = 0;
    this.pos = 0;                                // ตำแหน่งอ่าน (ทศนิยม)
    this.out = new Int16Array(this.frame);
    this.outLen = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || ch.length === 0) return true;

    // ยังตามไม่ทัน (แท็บถูกพัก) → ทิ้งของเก่าไว้ก่อน ดีกว่าปล่อยหน่วงสะสม
    if (this.len + ch.length > CAP) {
      this.len = 0;
      this.pos = 0;
    }
    this.buf.set(ch, this.len);
    this.len += ch.length;

    while (this.pos + this.step <= this.len) {
      const from = Math.floor(this.pos);
      const to = Math.min(this.len, Math.floor(this.pos + this.step));
      let sum = 0;
      let n = 0;
      for (let i = from; i < to; i++) { sum += this.buf[i]; n++; }
      let v = n ? sum / n : 0;
      if (v > 1) v = 1; else if (v < -1) v = -1;
      this.out[this.outLen++] = v < 0 ? v * 0x8000 : v * 0x7fff;
      this.pos += this.step;

      if (this.outLen >= this.frame) {
        const copy = this.out.slice();
        this.port.postMessage(copy.buffer, [copy.buffer]);
        this.outLen = 0;
      }
    }

    // ตัดส่วนที่อ่านไปแล้วออก เหลือเศษไว้ต่อกับก้อนถัดไป
    const used = Math.floor(this.pos);
    if (used > 0) {
      this.buf.copyWithin(0, used, this.len);
      this.len -= used;
      this.pos -= used;
    }
    return true;
  }
}

registerProcessor('mic-processor', MicProcessor);
