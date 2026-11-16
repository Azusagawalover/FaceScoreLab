"""Public metadata-only availability audit. Never downloads full images or archives."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

from evaluation_common import write_json

MAX_ADDITIONAL_BYTES = 20 * 1024 * 1024
REPOSITORIES = ["flwrlabs/celeba", "huggan/CelebA-faces-with-attributes", "Yuehao/celeba", "tpremoli/CelebA-attrs"]


class MetadataClient:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        previous = self.output / "request_log.json"
        saved = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
        self.total = saved.get("additional_bytes_read", 0)
        self.records = saved.get("requests", [])

    def get(self, url, name, limit=2 * 1024 * 1024, byte_range=None):
        for previous in self.records:
            if previous.get("url") == url and previous.get("requested_range") == byte_range:
                if previous.get("status") == "downloaded_metadata" and Path(previous["path"]).exists():
                    content = Path(previous["path"]).read_bytes()
                    if hashlib.sha256(content).hexdigest() == previous["sha256"]:
                        return content, previous
        limit = min(limit, MAX_ADDITIONAL_BYTES - self.total)
        if limit <= 0:
            raise RuntimeError("Metadata byte budget exhausted")
        record = {"name": name, "url": url, "requested_range": byte_range, "authenticated": False}
        headers = {"User-Agent": "FaceScoreLab-public-metadata-only-audit/1.0"}
        if byte_range:
            headers["Range"] = byte_range
        data = None
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=25) as response:
                record.update({"http_status": response.status, "content_type": response.headers.get("Content-Type"),
                               "content_length": response.headers.get("Content-Length"), "content_range": response.headers.get("Content-Range")})
                if byte_range and response.status != 206:
                    raise ValueError("Server ignored Range; body not read")
                if response.headers.get("Content-Length") and int(response.headers["Content-Length"]) > limit:
                    raise ValueError("Response exceeds bounded metadata request; body not read")
                data = response.read(limit)
                self.total += len(data)
                if len(data) == limit:
                    raise ValueError("Response reached byte cap; cannot assert completeness")
            path = self.output / name
            path.write_bytes(data)
            record.update({"status": "downloaded_metadata", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "path": str(path)})
        except Exception as error:
            record.update({"status": "unverified", "error": f"{type(error).__name__}: {error}"})
            data = None
        self.records.append(record)
        write_json(self.output / "request_log.json", {"additional_bytes_read": self.total, "budget_bytes": MAX_ADDITIONAL_BYTES, "requests": self.records})
        return data, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = MetadataClient(args.output / "availability_metadata")
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "Public metadata only; no image archives, payment, login, authentication or bypass", "candidates": []}
    for repository in REPOSITORIES:
        slug = repository.replace("/", "__")
        record = {"repository": repository, "source": f"https://huggingface.co/datasets/{repository}"}
        for label, url in [
            ("root_tree", f"https://huggingface.co/api/datasets/{repository}/tree/main?recursive=false&expand=false"),
            ("readme", f"https://huggingface.co/datasets/{repository}/raw/main/README.md"),
            ("splits", f"https://datasets-server.huggingface.co/splits?dataset={urllib.parse.quote(repository, safe='')}"),
        ]:
            data, status = client.get(url, slug + "__" + label + ".txt")
            record[label] = status
            if data and label != "readme":
                try:
                    record[label + "_parsed"] = json.loads(data)
                except ValueError:
                    record[label + "_parsed"] = None
        result["candidates"].append(record)
    result["additional_metadata_bytes_read"] = client.total
    result["images_downloaded"] = 0
    result["status"] = "initial_metadata_collected_requires_review"
    write_json(args.output / "availability_followup.json", result)
    print(json.dumps({"output": str(args.output / "availability_followup.json"), "bytes": client.total, "candidates": len(result["candidates"])}))


if __name__ == "__main__":
    main()
