"""
Publish the app to a Hugging Face Space (Docker SDK).

    python deploy_hf.py --space USER/SPACE-NAME --dry-run   # build the snapshot only
    python deploy_hf.py --space USER/SPACE-NAME             # build and push

The Space gets a single-commit snapshot of the committed HEAD, not the git
history: Hugging Face rejects binary files that are not stored with LFS/Xet,
and older commits contain files the Space must not carry. In the snapshot,
deploy/huggingface/Dockerfile and README.md (the Space configuration)
replace the repository's own, and weights and images are tracked with LFS.

Pushing needs a Hugging Face write token stored as a git credential
(`hf auth login --add-to-git-credential`) and an existing Docker Space.
"""

import argparse
import io
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OVERLAY = ROOT / "deploy" / "huggingface"
DROP = ("deploy", ".github", "tests", "notebooks")        # not needed to run the app
LFS_PATTERNS = ("*.pth", "*.png", "*.jpg", "*.jpeg", "*.npz", "*.npy")


def git(*args, cwd=ROOT, capture=False):
    result = subprocess.run(["git", *args], cwd=cwd, check=True,
                            capture_output=capture, text=capture)
    return result.stdout.strip() if capture else None


def build_snapshot(dest: Path) -> str:
    """Writes HEAD's files plus the Space overlay into ``dest``; returns HEAD's sha."""
    if git("status", "--porcelain", capture=True):
        raise SystemExit("Commit or stash your changes first: the Space is built from HEAD.")
    sha = git("rev-parse", "--short", "HEAD", capture=True)
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT,
                             check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    for name in DROP:
        shutil.rmtree(dest / name, ignore_errors=True)
    for f in OVERLAY.iterdir():
        shutil.copy(f, dest / f.name)
    (dest / ".gitattributes").write_text(
        "".join(f"{p} filter=lfs diff=lfs merge=lfs -text\n" for p in LFS_PATTERNS))
    return sha


def push(dest: Path, space: str, sha: str):
    git("init", "-q", "-b", "main", cwd=dest)
    git("lfs", "install", "--local", cwd=dest)
    git("add", "-A", cwd=dest)
    git("-c", "user.name=deploy_hf.py", "-c", "user.email=deploy@localhost",
        "commit", "-q", "-m", f"Deploy {sha} from GitHub", cwd=dest)
    # --force: the Space holds only the latest snapshot, never history
    git("push", "--force", f"https://huggingface.co/spaces/{space}", "main", cwd=dest)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish the app to a Hugging Face Space.")
    parser.add_argument("--space", required=True, help="USER/SPACE-NAME of an existing Docker Space")
    parser.add_argument("--dry-run", action="store_true",
                        help="only build the snapshot and print where it is")
    args = parser.parse_args(argv)
    if args.space.count("/") != 1 or not all(args.space.split("/")):
        parser.error("--space must look like USER/SPACE-NAME")

    dest = Path(tempfile.mkdtemp(prefix="hf_space_"))
    sha = build_snapshot(dest)
    if args.dry_run:
        print(f"Snapshot of {sha} built in {dest} (not pushed)")
        return
    push(dest, args.space, sha)
    shutil.rmtree(dest, ignore_errors=True)
    print(f"Pushed {sha} to https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
