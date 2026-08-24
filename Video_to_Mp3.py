import os
import sys
import subprocess

# Force UTF-8 encoding for standard output on Windows
sys.stdout.reconfigure(encoding='utf-8')

folder = "SnapTube Video"

files = os.listdir(folder)

for file in files:

    if file.endswith(".mp4"):

        tutorial_number = file.split("_")[0].replace("Lec - ", "").replace("Lec -", "")

        file_name = file.split("_", 1)[1]

        name_without_ext = os.path.splitext(file_name)[0]

        print(tutorial_number, name_without_ext)

        subprocess.run([
            "ffmpeg",
            "-i", f"{folder}/{file}",
            f"audios/{tutorial_number}_{name_without_ext}.mp3"
        ])