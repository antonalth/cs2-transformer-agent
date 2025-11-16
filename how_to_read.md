# How to read ML codebases

Helpful for other people also working on this, bcs we didn't yet take the time to write a proper readme.md

1. find out what the entrypoint for training is. in this case it is: `python transformers/model/train3.py --mode train --data-root /mnt/trainingdata/dataset0/ --use-precomputed-embeddings`
2. read the entrypoint file like a compiler, but only step into functions like: train() or model = ...()

