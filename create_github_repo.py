"""
create_github_repo.py
---------------------
One-time script to create the GitHub repo and push the initial commit.

Usage:
    python create_github_repo.py --token <YOUR_PAT> --username <YOUR_GITHUB_USERNAME>

Requirements:
    - Personal Access Token (classic) with `repo` scope
    - git installed and configured (name + email)
"""

import argparse
import json
import os
import subprocess
import sys

import requests


def create_remote_repo(token: str, username: str, repo_name: str, description: str) -> str:
    """Create repo via GitHub API, return clone URL."""
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {
        "name": repo_name,
        "description": description,
        "private": False,
        "auto_init": False,
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload))
    if resp.status_code == 201:
        clone_url = resp.json()["clone_url"]
        print(f"✓ Repo created: https://github.com/{username}/{repo_name}")
        return clone_url
    elif resp.status_code == 422:
        # Repo likely already exists
        print(f"Repo already exists, using existing: https://github.com/{username}/{repo_name}")
        return f"https://github.com/{username}/{repo_name}.git"
    else:
        print(f"Failed to create repo: {resp.status_code} {resp.text}")
        sys.exit(1)


def run(cmd: list[str], cwd: str | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {' '.join(cmd)}:\n{result.stderr}")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",    required=True, help="GitHub PAT (classic, repo scope)")
    parser.add_argument("--username", required=True, help="Your GitHub username")
    parser.add_argument("--repo",     default="tda-hf-crypto",
                        help="Repo name (default: tda-hf-crypto)")
    args = parser.parse_args()

    description = (
        "TDA-augmented high-frequency volatility forecasting in crypto markets "
        "— UChicago REU 2026"
    )

    # 1. Create remote repo
    clone_url = create_remote_repo(args.token, args.username, args.repo, description)

    # 2. Inject token into URL for push auth (HTTPS)
    authed_url = clone_url.replace("https://", f"https://{args.username}:{args.token}@")

    # 3. Init local git repo (idempotent)
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    git_dir = os.path.join(repo_dir, ".git")

    if not os.path.exists(git_dir):
        run(["git", "init"], cwd=repo_dir)
        run(["git", "branch", "-M", "main"], cwd=repo_dir)
        print("✓ Initialized local git repo")
    else:
        print("✓ Local git repo already initialized")

    # 4. Configure remote
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_dir, capture_output=True, text=True
    )
    if result.returncode == 0:
        run(["git", "remote", "set-url", "origin", authed_url], cwd=repo_dir)
    else:
        run(["git", "remote", "add", "origin", authed_url], cwd=repo_dir)
    print("✓ Remote set")

    # 5. Stage and commit
    run(["git", "add", "-A"], cwd=repo_dir)
    run(
        ["git", "commit", "-m", "Initial commit: project scaffold, data pipeline, config"],
        cwd=repo_dir
    )
    print("✓ Initial commit created")

    # 6. Push
    run(["git", "push", "-u", "origin", "main"], cwd=repo_dir)
    print(f"\n✓ Pushed to https://github.com/{args.username}/{args.repo}")
    print(f"  Next: cd {repo_dir} && pip install -r requirements.txt && python src/utils/fetch_data.py")


if __name__ == "__main__":
    main()
