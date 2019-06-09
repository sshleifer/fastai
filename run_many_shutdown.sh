#!/usr/bin/env bash
python fastai/imagito/run_many.py
git add experiments
g commit -m "Add experiments"
git push
sudo shutdown now
