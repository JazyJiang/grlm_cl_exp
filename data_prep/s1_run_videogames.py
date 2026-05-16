"""Run s1_init_sum on videogames only."""
import torch.multiprocessing as mp
from s1_init_sum import main

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main("videogames")
