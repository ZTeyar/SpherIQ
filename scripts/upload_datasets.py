"""
Upload SpherIQ datasets to HF Hub — batched commits, resume-capable.
"""
import os, time, math, logging
logging.disable(logging.CRITICAL)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import HfApi, create_repo, CommitOperationAdd

REPO_ID = "ZTeyar/SpherIQ-data"
SOURCE = "/home/luke/Desktop/spheriq-v9"
SCORES = "/home/luke/Desktop/SpherIQ/data"
DATASETS = ["live", "cviq", "odi"]
BATCH_SIZE = 30

api = HfApi(token=os.environ["HF_TOKEN"])

create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)

def get_remote():
    try:
        return {i.path for i in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)}
    except:
        return set()

remote = get_remote()
print(f"Remote: {len(remote)} files")

def commit_batch(ops, desc, retries=4):
    for attempt in range(retries):
        try:
            api.create_commit(repo_id=REPO_ID, operations=ops,
                              commit_message=desc, repo_type="dataset", num_threads=4)
            return True
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait = 90 * (attempt + 1)
                print(f"  429: waiting {wait}s...")
                time.sleep(wait)
            elif attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  retry {attempt+1}: {e}, wait {wait}s")
                time.sleep(wait)
            else:
                print(f"  FAILED: {e}")
                return False
    return False

for name in DATASETS:
    src = os.path.join(SOURCE, name)
    files = []
    for dp, _, fn in os.walk(src):
        for f in fn:
            local = os.path.join(dp, f)
            rel = os.path.relpath(local, SOURCE)
            files.append((local, rel))

    pending = [(l, r) for l, r in files if r not in remote]
    print(f"[{name}] {len(files)} files, {len(pending)} pending")

    if not pending:
        print("  done\n")
        continue

    n_batches = math.ceil(len(pending) / BATCH_SIZE)
    t0 = time.time()
    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i:i+BATCH_SIZE]
        bn = i // BATCH_SIZE + 1
        ops = [CommitOperationAdd(path_or_fileobj=local, path_in_repo=rel)
               for local, rel in batch]
        ok = commit_batch(ops, f"{name} batch {bn}/{n_batches}")
        if not ok:
            break
        elapsed = time.time() - t0
        pct = min(100, (i + len(batch)) / len(pending) * 100)
        print(f"  batch {bn}/{n_batches} ({pct:.0f}%, {(i+len(batch))/(elapsed/3600):.0f} f/h)")

    print(f"  [{name}] {(time.time()-t0)/60:.1f} min\n")
    remote = get_remote()

print("Score CSVs...")
pending_csv = []
for f in os.listdir(SCORES):
    local = os.path.join(SCORES, f)
    if os.path.isfile(local) and f"scores/{f}" not in remote:
        pending_csv.append((local, f"scores/{f}"))
if pending_csv:
    commit_batch([CommitOperationAdd(path_or_fileobj=l, path_in_repo=r) for l,r in pending_csv], "score CSVs")
else:
    print("  done")

print(f"\nDone. https://huggingface.co/datasets/{REPO_ID}")
