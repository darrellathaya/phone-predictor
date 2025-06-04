.PHONY: train commit push retrain

train:
    python src/train_model.py

setup-git:
    git config --local user.email "action@github.com"
    git config --local user.name "GitHub Action"

commit: setup-git
    git add models/*
    if ! git diff --cached --quiet; then \
        git commit -m "Auto-update models and metrics after retraining"; \
    else \
        echo "No model changes detected. Skipping commit."; \
    fi

push:
    git remote set-url origin https://x-access-token:$$REPO_PAT@github.com/$$GITHUB_REPOSITORY.git
    git push origin HEAD:main

retrain: train commit push
