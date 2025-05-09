from deep_translator import GoogleTranslator
import chardet
import pandas as pd
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_FILE = "src\\True.csv"
OUTPUT_FILE = "src\\translated_parallel_True.csv"
MAX_LEN = 5000
MAX_WORKERS = 3 
SLEEP_TIME = 0.2


with open(INPUT_FILE, "rb") as f:
    result = chardet.detect(f.read(10000))
    encoding = result['encoding']
    print(f"[INFO] Кодировка: {encoding}")


with open(INPUT_FILE, "r", encoding=encoding, errors="ignore") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"[INFO] Прочитано строк: {len(lines)}")

def safe_translate_line(index, text):
    translator = GoogleTranslator(source='en', target='ru')
    try:
        if len(text) <= MAX_LEN:
            translated = translator.translate(text)
        else:
            chunks = [text[i:i+MAX_LEN] for i in range(0, len(text), MAX_LEN)]
            translated_chunks = [translator.translate(chunk) for chunk in chunks]
            translated = ' '.join(translated_chunks)
        time.sleep(SLEEP_TIME)  
        return index, translated
    except Exception as e:
        print(f"\n[WARN] Ошибка в строке {index}: {e}")
        return index, text 

translated_lines = [None] * len(lines)
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(safe_translate_line, i, line) for i, line in enumerate(lines)]
    
    for i, future in enumerate(as_completed(futures)):
        index, result = future.result()
        translated_lines[index] = result

        percent = (i + 1) / len(lines) * 100
        sys.stdout.write(f"\r[PROGRESS] {percent:.2f}%")
        sys.stdout.flush()

print("\n[INFO] Перевод завершён. Сохраняю в CSV...")

df = pd.DataFrame(translated_lines, columns=["translated"])
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"[INFO] Готово. Файл: {OUTPUT_FILE}")
