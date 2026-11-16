"""Bounded, ETag-checked public HTTP ranges; preserves a verified-source prefix."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.request


def transfer_ranges(url, part, root, state, state_path, expected_size, budget, budget_limit, atomic_json, workers=4):
    if not 2 <= workers <= 8:
        raise ValueError("Parallel public download requires 2 to 8 workers")
    root, part = Path(root).resolve(), Path(part).resolve()
    if not part.is_relative_to(root):
        raise ValueError("Partial file escaped the authorized dataset root")
    prefix = part.stat().st_size if part.exists() else 0
    etag = state.get("etag")
    if not etag or etag.startswith("W/"):
        raise ValueError("A strong previously verified ETag is required before parallel resume")
    if not 0 <= prefix <= expected_size:
        raise ValueError("Partial length outside verified object")
    chunk_root = root / "public_range_chunks"
    chunk_root.mkdir(exist_ok=True)
    chunk_size = 8 * 1024 * 1024
    ranges = [(start, min(expected_size - 1, start + chunk_size - 1)) for start in range(prefix, expected_size, chunk_size)]
    lock = threading.Lock()
    progress_started = time.monotonic()
    state.update({"status": "parallel_range_download", "workers": workers, "prefix_bytes_preserved": prefix, "chunk_bytes": chunk_size})
    atomic_json(state_path, state)

    def fetch(bounds):
        start, end = bounds
        final = chunk_root / f"{start:012d}-{end:012d}.chunk"
        pending = final.with_suffix(".partial")
        record_path = final.with_suffix(".json")
        expected_record = {"source": url, "etag": etag, "start": start, "end": end, "total": expected_size}
        if record_path.exists():
            existing = json.loads(record_path.read_text())
            if any(existing.get(key) != value for key, value in expected_record.items()):
                raise ValueError("Cached range provenance differs; refusing reuse")
            if final.exists():
                payload = final.read_bytes()
                if len(payload) != end - start + 1 or hashlib.sha256(payload).hexdigest() != existing.get("sha256"):
                    raise ValueError("Completed range hash mismatch")
                return final
        elif pending.exists() or final.exists():
            raise ValueError("Cached range lacks ETag/provenance metadata")
        atomic_json(record_path, expected_record)
        for attempt in range(3):
            length = pending.stat().st_size if pending.exists() else 0
            current = start + length
            if current > end + 1:
                raise ValueError("Range partial exceeds expected size")
            if current == end + 1:
                break
            planned = end - current + 1
            # Reserve the whole attempt before any body read. Failed attempts stay
            # charged conservatively, so simultaneous workers cannot overspend.
            with lock:
                if budget["accounted_bytes"] + planned > budget_limit:
                    raise ValueError("2 GiB budget prevents further transfer attempts")
                budget["accounted_bytes"] += planned
                budget["parallel_range_reservations_bytes"] = budget.get("parallel_range_reservations_bytes", 0) + planned
                budget["parallel_accounting_note"] = "Full attempts reserved in advance; failures remain conservatively charged."
                atomic_json(root / "download_budget.json", budget)
            request = urllib.request.Request(url, headers={"User-Agent": "FaceScoreLab-bounded-public-ranges/1.0", "Accept-Encoding": "identity",
                                                           "Range": f"bytes={current}-{end}", "If-Range": etag})
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {current}-{end}/{expected_size}":
                        raise ValueError("Range not honored; response body was not read")
                    if response.headers.get("ETag") != etag or int(response.headers.get("Content-Length", -1)) != planned:
                        raise ValueError("Range ETag/length changed; response body was not read")
                    with pending.open("ab" if length else "wb") as handle:
                        while True:
                            data = response.read1(min(1024 * 1024, end - start + 1 - handle.tell()))
                            if not data:
                                break
                            handle.write(data)
                    if pending.stat().st_size != end - start + 1:
                        raise ConnectionError("Incomplete range preserved for resume")
                break
            except (TimeoutError, ConnectionError, OSError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403):
                    raise RuntimeError("Public range requires access/authentication; no bypass attempted") from None
                if attempt == 2:
                    raise RuntimeError("Three attempts failed for one range; partial bytes retained") from None
                time.sleep(2)
        payload = pending.read_bytes()
        if len(payload) != end - start + 1:
            raise ValueError("Range still incomplete")
        expected_record["sha256"] = hashlib.sha256(payload).hexdigest()
        pending.replace(final)
        atomic_json(record_path, expected_record)
        return final

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Queue all remaining ranges, while the pool keeps live connections bounded.
        # A slow early range must not idle workers whose later ranges completed.
        # Total prefix + cached ranges stays bounded by the original object size.
        futures = [pool.submit(fetch, bounds) for bounds in ranges]
        try:
            for index, (start, end) in enumerate(ranges):
                chunk = futures[index].result()
                actual_prefix = part.stat().st_size if part.exists() else 0
                if actual_prefix != start:
                    raise ValueError("Ordered append prefix differs from the requested range")
                with part.open("ab") as destination, chunk.open("rb") as source:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
                if part.stat().st_size != end + 1:
                    raise ValueError("Ordered range append length mismatch")
                for temporary in (chunk, chunk.with_suffix(".json")):
                    if not temporary.resolve().is_relative_to(chunk_root.resolve()):
                        raise ValueError("Cleanup path escaped range cache")
                    temporary.unlink()
                state["bytes_on_disk"] = end + 1
                atomic_json(state_path, state)
                elapsed = max(time.monotonic() - progress_started, .001)
                print(json.dumps({"file": part.name, "bytes": end + 1, "total": expected_size,
                                  "percent": round(100 * (end + 1) / expected_size, 1),
                                  "average_mib_per_second": round((end + 1 - prefix) / 1024**2 / elapsed, 2), "workers": workers}), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
