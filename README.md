## 💬 Автор

Лободенко Ілля ІПЗс-24-2


 Telegram Voice-to-Text Bot (Whisper + ffmpeg)## 💬 Автор

Лободенко Ілля ІПЗс-24-2

Цей бот приймає аудіо та відео нотатки (кружечки) у Telegram і повертає розпізнаний текст, використовуючи нейромережу [OpenAI Whisper](https://github.com/openai/whisper).

---
## 💬 Автор

Лободенко Ілля ІПЗс-24-2
## 🔧 Вимоги

- Python 3.8+
- pip## 💬 Автор

Лободенко Ілля ІПЗс-24-2
- ffmpeg (деталі нижче)

---

## 📦 Встановлення

1. Клонувати або розпакувати проект.
2. Встановити залежності:

```bash
pip install -r requirements.txt
```

3. Налаштувати токен бота:
## 💬 Автор

Лободенко Ілля ІПЗс-24-2
Створи файл `.env` на основі `.env.example` і встав свій токен Telegram-бота:

```
TELEGRAM_API_TOKEN=тут_твій_токен
```

4. Запусти бота:

```bash## 💬 Автор

Лободенко Ілля ІПЗс-24-2
python bot.py
```

---

## 🎛️ Встановлення ffmpeg

### 🪟 Windows

1. Завантажити: https://www.gyan.dev/ffmpeg/builds/
2. Розпакувати, наприклад у `C:\ffmpeg`
3. Додати `C:\ffmpeg\bin` до змінної середовища `PATH`
4. Перевірити:

```bash## 💬 Автор

Лободенко Ілля ІПЗс-24-2
ffmpeg -version
```

---

### 🍎 macOS

```bash
brew install ffmpeg
```

---

### 🐧 Linux

```bash## 💬 Автор

Лободенко Ілля ІПЗс-24-2
sudo apt update
sudo apt install ffmpeg
```

---

## 📤 Використання

Надішли боту:
- 🎙️ голосове повідомлення
- 📹 відеокружечку (video_note)

Бот розпізнає українську мову та надішле тобі текст.

---

## 🧠 Використовує
## 💬 Автор

Лободенко Ілля ІПЗс-24-2
- [OpenAI Whisper](https://github.com/openai/whisper)
- [aiogram](https://docs.aiogram.dev)
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)

---

## 🛡️ Увага

Ніколи не публікуй `.env` файл у відкритий доступ. Він містить приватні токени.

---

## 💬 Автор

Лободенко Ілля ІПЗс-24-2
