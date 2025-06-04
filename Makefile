# Makefile

.PHONY: train test commit deploy

train:
    python src/train_model.py

test:
    pytest tests/

commit:
    git config --local user.email "action@github.com"
    git config --local user.name "GitHub Action"
    git add models/*
    if ! git diff --cached --quiet; then \
        git commit -m "Auto-update models and metrics after retraining"; \
    fi

push:
    git remote set-url origin https://x-access-token:${REPO_PAT}@github.com/${GITHUB_REPOSITORY}.git 
    if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then git checkout main; fi
    git pull
    git merge --ff-only HEAD@{u}
    git push origin HEAD:main

retrain: train commit push