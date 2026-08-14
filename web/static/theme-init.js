/* เลือกธีมก่อน paint แรก — ไม่ให้หน้าจอกระพริบธีมผิดตอนโหลด
   ค่าที่ผู้ใช้เคยเลือกมาก่อนสำคัญกว่าค่าของระบบ

   ไฟล์นี้แยกออกมาจาก <script> ใน index.html เพื่อให้ CSP ตั้ง script-src เป็น
   'self' ได้ (ไม่ต้องเปิด 'unsafe-inline' ที่ทำให้สคริปต์ฝังตัวรันได้ทั้งหมด)
   ต้องคงเป็น <script src> แบบไม่มี defer/async ใน <head> เท่านั้น เพราะมันต้อง
   รันจบก่อนเบราว์เซอร์วาดเฟรมแรก ถ้าใส่ defer ธีมจะกระพริบทุกครั้งที่โหลดหน้า */
(function () {
  var t = null;
  try { t = localStorage.getItem('voicelink.theme.v1'); } catch (_) { /* localStorage ถูกปิด */ }
  if (t !== 'hud' && t !== 'clay') {
    var light = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
    t = light ? 'clay' : 'hud';
  }
  document.documentElement.dataset.theme = t;
})();
