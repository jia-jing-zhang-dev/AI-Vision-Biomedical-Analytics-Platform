"""
Publish the app to a Hugging Face Space (Docker SDK).

    python deploy_hf.py --space USER/SPACE-NAME --dry-run   # build the snapshot only
    python deploy_hf.py --space USER/SPACE-NAME             # build and push

The Space gets a snapshot of the committed HEAD, not the git history (older
commits contain files the Space must not carry). In the snapshot,
deploy/huggingface/Dockerfile and README.md (the Space configuration)
replace the repository's own. The upload goes through huggingface_hub,
which stores weights and images the way the Hub requires (Xet), and
mirrors the folder: files no longer in the snapshot are deleted from the Space.

Needs huggingface_hub, a write token (`hf auth login`) and an existing Space.
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
    return sha


def push(dest: Path, space: str, sha: str):
    from huggingface_hub import HfApi
    HfApi().upload_folder(
        folder_path=str(dest), repo_id=space, repo_type="space",
        commit_message=f"Deploy {sha} from GitHub",
        delete_patterns="*",            # mirror: drop files removed from the repository
    )


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
