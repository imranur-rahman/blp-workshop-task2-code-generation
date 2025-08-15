import json, os, re, zipfile

SUB_PATH = "submission.json"

def file_format_check(path: str) -> bool:
    # name + extension
    if os.path.basename(path) != "submission.json":
        print("Error: File name must be exactly 'submission.json'")
        return False
    if not path.lower().endswith(".json"):
        print("Error: File must have .json extension")
        return False

    # must be valid JSON (not JSONL) and root must be a list
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        print("Note: The file must be in proper JSON format (not JSONL)")
        return False

    if not isinstance(data, list):
        print("Error: The root element should be a list of objects")
        return False

    # each item: dict with ONLY keys {'id','response'}; id=int; response=str
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Error: Item at index {idx} is not a dictionary")
            return False
        keys = set(item.keys())
        if keys != {"id", "response"}:
            print(f"Error: Item at index {idx} must contain only keys 'id' and 'response', found: {keys}")
            return False
        if not isinstance(item["id"], int):
            print(f"Error: 'id' field at index {idx} must be an integer")
            return False
        if not isinstance(item["response"], str):
            print(f"Error: 'response' field at index {idx} must be a string")
            return False

    print("Format check passed successfully!")
    return True

def make_submission_zip():
    # ---------- Load, compute per-item validity, blank invalids, save, zip ----------
    # Load JSON list
    with open(SUB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = len(data)
    fence_pat = re.compile(r"^```python[\s\S]*```$", re.MULTILINE)

    valid_format = []
    valid_fence  = []
    valid_both   = []

    # Per-item validation mirrors file checker semantics
    def item_format_ok(item):
        return (
            isinstance(item, dict)
            and set(item.keys()) == {"id", "response"}
            and isinstance(item["id"], int)
            and isinstance(item["response"], str)
        )

    for item in data:
        vfmt = item_format_ok(item)
        vf   = bool(fence_pat.match(item["response"])) if vfmt else False
        valid_format.append(vfmt)
        valid_fence.append(vf)
        valid_both.append(vfmt and vf)

    # Report stats
    nf = sum(valid_fence)
    nm = sum(valid_format)
    nb = sum(valid_both)
    den = max(n, 1)
    print(f"Fencing valid: {nf}/{n} ({nf*100.0/den:.1f}%)")
    print(f"Format valid:  {nm}/{n} ({nm*100.0/den:.1f}%)")
    print(f"Both valid:    {nb}/{n} ({nb*100.0/den:.1f}%)")

    # Strict policy: blank responses that fail ANY check
    for i, ok in enumerate(valid_both):
        if not ok and isinstance(data[i], dict) and "response" in data[i]:
            data[i]["response"] = ""

    # Overwrite submission.json (id+response only)
    with open(SUB_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [{"id": item["id"], "response": item["response"]} for item in data],
            f, ensure_ascii=False, indent=2
        )
    print("✅ Updated submission.json after checks (invalid responses blanked).")

    # Final file-level check (should pass)
    _ = file_format_check(SUB_PATH)

    # Zip as submission.zip (Jupyter-friendly, no shell commands)
    with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(SUB_PATH)
    print("📦 Created submission.zip containing submission.json.")