import json
import sys
import whisper
import os

# Force UTF-8 encoding for standard output on Windows
sys.stdout.reconfigure(encoding='utf-8')

model = whisper.load_model("large-v2")

audio_folder = "audios"
json_folder = "jsons"

# ensure json folder exists
if os.path.exists(json_folder) and not os.path.isdir(json_folder):
    os.remove(json_folder)

os.makedirs(json_folder, exist_ok=True)

audios = os.listdir(audio_folder)

for audio in audios:
    if audio.endswith(".mp3"):

        # remove .mp3
        name = os.path.splitext(audio)[0]

        # remove unwanted text
        name = name.replace("(720P_HD)", "").strip()

        # split number and title
        if "_" in name:
            number, title = name.split("_", 1)
        else:
            number, title = name.split(" ", 1)

        number = number.strip()
        title = title.strip()

        print(f"Processing: {number} {title}")

        # transcribe (INSIDE loop)
        result = model.transcribe(
            audio=os.path.join(audio_folder, audio),
            language="hi",
            task="translate",
            word_timestamps=False
        )

        # create chunks for THIS file only
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        # clean file name
        clean_name = f"{number} {title}"

        # save JSON
        file_path = os.path.join(json_folder, f"{clean_name}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "chunks": chunks,
                "text": result["text"]
            }, f, ensure_ascii=False, indent=2)

print("All files processed successfully!")