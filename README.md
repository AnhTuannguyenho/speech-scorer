# Speech Scorer — RunPod Serverless (Load Balancer)

Phần mềm chấm phát âm. Engine: **faster-whisper** (ASR) + **wav2vec2** phoneme GOP.
Chạy theo chế độ **Load Balancer** của RunPod — request đi **thẳng HTTP tới worker, KHÔNG qua hàng đợi** (né lỗi dispatcher), độ trễ thấp.

## Cách hoạt động (Load Balancer)
- App là **Flask HTTP server** nghe cổng `PORT` (mặc định 80).
- `/ping`: **200** = sẵn sàng, **204** = đang nạp model (RunPod tự đợi).
- Model nạp ở luồng nền → worker phản hồi /ping ngay khi container chạy.

## Tạo endpoint
Console → New Endpoint → Import Git Repository `AnhTuannguyenho/speech-scorer` → **Advanced settings → chọn `Load balancer`** → Deploy.
- GPU 16GB+ · Dockerfile `Dockerfile` · (PORT mặc định 80, không cần đổi)

## Gọi (client / web khác)
```
POST https://<ENDPOINT_ID>.api.runpod.ai/score
Authorization: Bearer <RUNPOD_API_KEY>
Content-Type: application/json
Body: {"text":"apple","audio_b64":"<base64 audio>"}
```
→ Trả JSON TRỰC TIẾP (không bọc `.output`): `{ok, score, status, band, heard, phones, fluency...}`

Cũng nhận **multipart** (`file=@audio` + `text=`) nếu muốn gửi file thẳng (đỡ base64).

## Routes
| path | input | output |
|---|---|---|
| `POST /score` | text, audio_b64 [,words] | score(0-10), status, band, heard, marks, phones, fluency/wpm/completeness/criteria(câu) |
| `POST /grade` | text, audio_b64 [,prompt] | text, accuracy, phones, duration |
| `POST /grade_ph` | text, audio_b64 | accuracy, word_ok, phones, said |
| `POST /transcribe` | audio_b64 [,lang,prompt,fast] | text, words[], duration |
| `POST /pron` | text, audio_b64 | accuracy, target, said, phones |
| `GET /health` | — | model, w2v, device, ok |
| `GET /ping` | — | 200/204 (health check RunPod) |
