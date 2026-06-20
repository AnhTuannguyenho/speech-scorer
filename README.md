# Speech Scorer — RunPod Serverless

Phần mềm chấm phát âm (viết mới, gọn). Engine: **faster-whisper** (ASR) + **wav2vec2** phoneme GOP forced-alignment. Handler TRỰC TIẾP (không Flask).

## Gọi
```
POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync
Authorization: Bearer <RUNPOD_API_KEY>
Content-Type: application/json
Body: {"input":{"route":"score","text":"apple","audio_b64":"<base64 audio>"}}
```
Kết quả nằm trong `.output`.

## Routes
| route | input | output |
|---|---|---|
| `score` | text, audio_b64 [,words] | score(0-10), status, band, heard, marks, phones, fluency/wpm/completeness/criteria (câu) |
| `grade` | text, audio_b64 [,prompt] | text, accuracy, phones, duration |
| `grade_ph` | text, audio_b64 | accuracy, word_ok, phones, said (chấm 1 từ bằng âm vị) |
| `transcribe` | audio_b64 [,lang,prompt,fast] | text, words[], duration |
| `pron` | text, audio_b64 | accuracy, target, said, phones |
| `health` | — | model, w2v, device |

`audio_b64`: base64 file audio (webm/mp3/wav/m4a/aiff…), engine tự ffmpeg.

## Build/Deploy
Tạo endpoint RunPod Serverless từ repo này (Dockerfile ở gốc). GPU 16GB+ đủ. Model nướng sẵn trong image. Đổi model: `--build-arg ASR_MODEL=small.en` (nhẹ/nhanh hơn).
