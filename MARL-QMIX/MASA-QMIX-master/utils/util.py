import math
import time
import numpy as np
from loguru import logger
from contextlib import contextmanager

def distance(initial_pos, end_pos, speed):
    return math.sqrt((end_pos[0] - initial_pos[0]) ** 2 + (end_pos[1] - initial_pos[1]) ** 2) / speed


def count_path_on_road(initial_pos, end_pos, speed):
    return (abs(end_pos[0] - initial_pos[0]) + abs(end_pos[1] - initial_pos[1])) / speed

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    logger.info("{} - done in {:.3f}s".format(title, time.time() - t0))
