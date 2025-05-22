import os
import tempfile
import ffmpeg
import whisper
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ContentType
from dotenv import load_dotenv

# Завантаження змінних середовища з .env
load_dotenv()

# Ініціалізація Telegram-бота
API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
print("API_TOKEN:", API_TOKEN)  # Тестове виведення токена (можна прибрати)
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Завантаження покращеної моделі Whisper
model = whisper.load_model("large")  # Варіанти: tiny, base, small, medium, large

# Функція для конвертації аудіофайлу у формат WAV
def convert_to_wav(file_path: str) -> str:
    wav_path = f"{file_path}.wav"
    (
        ffmpeg
        .input(file_path)
        .output(wav_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )
    return wav_path

# Обробка медіа (аудіо або відео нотатки)
async def handle_media(message: types.Message, file_id: str):
    # Завантаження файлу від користувача
    file = await bot.get_file(file_id)
    file_path = file.file_path
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = os.path.join(tmpdir, "input.ogg")
        await bot.download_file(file_path, temp_file)

        # Конвертація у WAV
        wav_file = convert_to_wav(temp_file)

        # Розпізнавання мовлення за допомогою Whisper (автовизначення мови + fp16=False для стабільності на CPU)
        result = model.transcribe(wav_file, fp16=False)

        # Відправка розпізнаного тексту користувачу
        await message.reply(result["text"])

# Обробка голосових повідомлень
@dp.message_handler(content_types=ContentType.VOICE)
async def voice_handler(message: types.Message):
    await handle_media(message, message.voice.file_id)

# Обробка відео нотаток (кружечків)
@dp.message_handler(content_types=ContentType.VIDEO_NOTE)
async def video_note_handler(message: types.Message):
    await handle_media(message, message.video_note.file_id)

# Обробка команди /start
@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    await message.reply("Привіт! Надішли мені голосове повідомлення або кружечку, і я розпізнаю текст автоматично будь-якою мовою.")

# Запуск бота
if __name__ == '__main__':
    print("Bot is running...")
    executor.start_polling(dp, skip_updates=True)
