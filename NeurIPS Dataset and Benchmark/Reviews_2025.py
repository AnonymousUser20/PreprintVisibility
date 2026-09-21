import json
import re
import time
from pathlib import Path

import requests


def _title_for_filename(note: dict, fallback: str) -> str:
    raw = (
        note.get("content", {})
        .get("title", {})
        .get("value", "")
    )
    title = str(raw).strip()
    if not title:
        return fallback
    title = re.sub(r"\s+", " ", title)
    for c in r'\/:*?"<>|':
        title = title.replace(c, "_")
    title = title.strip(" .")
    if len(title) > 180:
        title = title[:180].rstrip(" .")
    return title or fallback


def NeurIPS_reviews(category):
  # category examples: "oral", "poster" — must match JSON filenames like {category}_page1.json
  directory = Path(r"C:\Users\poulami.paul\PyCharmMiscProject\3.ArXiv_Project\NeurIPS Dataset and Benchmark\2025")
  reviews_dir = directory.parent / "Reviews" / "2025" / category
  reviews_dir.mkdir(parents=True, exist_ok=True)

  json_count = len(list(directory.glob(f"{category}_page*.json")))
  for i in range(1, json_count + 1):
    json_path = directory / f"{category}_page{i}.json"
    with open(json_path, "r", encoding="utf-8") as fp:
      content = json.load(fp)
    for note in content["notes"]:
      forum_id = note["id"]
      url = f"https://api2.openreview.net/notes?count=true&details=writable%2Csignatures%2Cinvitation%2Cpresentation%2Ctags&domain=NeurIPS.cc%2F2025%2FDatasets_and_Benchmarks_Track&forum={forum_id}&limit=1000&trash=true"
      response = requests.get(url)

      #to avoid error 429 : too many requests in short time
      if response.status_code == 429:
          print("Rate limit hit. Waiting...")
          time.sleep(10)  # Wait 10 seconds before retry
          response = requests.get(url)

      response.raise_for_status()
      data = response.json()
      stem = _title_for_filename(note, forum_id)
      out_path = reviews_dir / f"{stem}.json"
      with open(out_path, "w", encoding="utf-8") as fp:
          json.dump(data, fp)
      time.sleep(1) #to avoid error 429 : too many requests in short time

def main():
  categories = ["oral", "poster", "spotlight"]
  for category in categories:
    NeurIPS_reviews(category)

if __name__ == "__main__":
  main()