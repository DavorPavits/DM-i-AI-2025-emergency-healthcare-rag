from pathlib import Path
import json
from llama_index.core.schema import Document

def load_documents(json_dir, txt_dir):
    documents = []
    txt_files = {f.stem: f for f in Path(txt_dir).rglob('*.txt')}
    json_files = {f.stem: f for f in Path(json_dir).rglob('*.json')}

    for file_id, txt_path in txt_files.items():
        json_path = json_files.get(file_id)
        if json_path is None:
            continue

        with open(txt_path, encoding="utf-8") as f:
            text = f.read().strip()

        with open(json_path, encoding="utf-8") as f:
            obj = json.load(f)
            is_true = obj.get("statement_is_true")
            topic = obj.get("statement_topic")

        if text and is_true is not None and topic is not None:
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_id": file_id,
                        "is_true": is_true,
                        "topic": topic
                    }
                )
            )
    return documents
