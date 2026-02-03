#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --bump <major|minor|patch>"
    exit 1
}

BUMP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bump) BUMP="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$BUMP" ]] && usage

# Ensure we're on main and up to date
BRANCH=$(git branch --show-current)
if [[ "$BRANCH" != "main" ]]; then
    echo "Error: must be on main branch (currently on $BRANCH)"
    exit 1
fi
git pull

# Bump version in pyproject.toml and uv.lock
NEW_VERSION=$(uv version --bump "$BUMP" --short)
echo "Bumped to $NEW_VERSION"

# Confirm before committing and pushing
read -rp "About to release v$NEW_VERSION — continue? [y/N] " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Aborted."
    git checkout pyproject.toml uv.lock
    exit 1
fi

# Commit, tag, and push
git add pyproject.toml uv.lock
git commit -m "bump to v$NEW_VERSION"
git push
git tag -a "v$NEW_VERSION" -m "v$NEW_VERSION"
git push --tags

echo "Released v$NEW_VERSION — workflow will build, publish to PyPI, and create GitHub release."
