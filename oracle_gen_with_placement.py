import os 
import uuid
import multiprocessing as mp
import random as _random
random = _random.SystemRandom()

import pandas as pd
import numpy as np
import cv2

NUM_CORES = 56
NUM_GEN_PICS = 10000
NUM_PIXELS = 500
ROOT_PLACES = pd.read_csv("Root_Places.csv")
ORACLE_ROOT_DIR = "00_Oracle/LOBI_Roots/"
GEN_DIR = "gen_res/"
os.makedirs(GEN_DIR, exist_ok=True)

def load_roots():
    roots = []
    for i in range(len(os.listdir(ORACLE_ROOT_DIR))):
        root = cv2.imread(os.path.join(ORACLE_ROOT_DIR, f"r{i+1}.bmp"), cv2.IMREAD_GRAYSCALE)
        root = 255 - root
        roots.append(root)
    return roots

ROOTS = load_roots()


def random_placement(n):
    # randomly select 1 from 10 kinds of placements
    weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    weights = np.array(weights) / np.sum(weights)
    placements = np.random.choice(10, n, p=weights)
    return placements

def random_root(place="A"):
    possible_roots = ROOT_PLACES[ROOT_PLACES[place] == 1]["r"].values
    return np.random.choice(possible_roots)

def put_root_on_canvas(root, position, canvas):
    canvas[position[0] - root.shape[0] // 2: position[0] - root.shape[0] // 2 + root.shape[0], position[1] - root.shape[1] // 2: position[1] - root.shape[1] // 2 + root.shape[1]] = np.maximum(root, canvas[position[0] - root.shape[0] // 2: position[0] - root.shape[0] // 2 + root.shape[0], position[1] - root.shape[1] // 2: position[1] - root.shape[1] // 2 + root.shape[1]])
    return canvas

def get_oracle_with_placement(placement):
    canvas = np.zeros((NUM_PIXELS, NUM_PIXELS), dtype=np.uint8)
    if placement == 0:
        position = (NUM_PIXELS // 2, NUM_PIXELS // 2)  # center
        pic_A = ROOTS[random_root("A")]
        pic_A = cv2.resize(pic_A, (int(NUM_PIXELS * random.uniform(0.5, 0.7)), int(NUM_PIXELS * random.uniform(0.5, 0.7))))
        # put pic_A in the center to the blank canvas
        canvas = put_root_on_canvas(pic_A, position, canvas)
    elif placement == 1 or placement == 4:
        if placement == 1:
            pic_B1, pic_B2 = ROOTS[random_root("B1")], ROOTS[random_root("B2")]
        else:
            pic_B1, pic_B2 = ROOTS[random_root("C1")], ROOTS[random_root("C2")]
        pic_B1 = cv2.resize(pic_B1, (int(NUM_PIXELS * random.uniform(0.25, 0.40)), int(NUM_PIXELS * random.uniform(0.25, 0.40))))
        pic_B2 = cv2.resize(pic_B2, (int(NUM_PIXELS * random.uniform(0.25, 0.40)), int(NUM_PIXELS * random.uniform(0.25, 0.40))))
        if placement == 1:
            pos_B1 = (NUM_PIXELS // 2, random.randint(int(NUM_PIXELS*0.2), int(NUM_PIXELS*0.4)))
            pos_B2 = (NUM_PIXELS // 2, random.randint(int(NUM_PIXELS*0.6), int(NUM_PIXELS*0.8)))
        else:
            pos_B1 = (random.randint(int(NUM_PIXELS*0.2), int(NUM_PIXELS*0.4)), NUM_PIXELS // 2)
            pos_B2 = (random.randint(int(NUM_PIXELS*0.6), int(NUM_PIXELS*0.8)), NUM_PIXELS // 2) 
        canvas = put_root_on_canvas(pic_B1, pos_B1, canvas)
        canvas = put_root_on_canvas(pic_B2, pos_B2, canvas)
    elif placement == 2 or placement == 3 or placement == 5 or placement == 6:
        # pic_B1 is the long root
        if placement == 2:
            pic_B1, pic_B3, pic_B4 = ROOTS[random_root("B1")], ROOTS[random_root("B3")], ROOTS[random_root("B4")]
        elif placement == 3:
            pic_B1, pic_B3, pic_B4 = ROOTS[random_root("B2")], ROOTS[random_root("B5")], ROOTS[random_root("B6")]
        elif placement == 5:
            pic_B1, pic_B3, pic_B4 = ROOTS[random_root("C1")], ROOTS[random_root("B6")], ROOTS[random_root("B4")]
        else:
            pic_B1, pic_B3, pic_B4 = ROOTS[random_root("C2")], ROOTS[random_root("B5")], ROOTS[random_root("B3")]
        pic_B1 = cv2.resize(pic_B1, (int(NUM_PIXELS * random.uniform(0.25, 0.40)), int(NUM_PIXELS * random.uniform(0.25, 0.40))))
        pic_B3 = cv2.resize(pic_B3, (int(NUM_PIXELS * random.uniform(0.25, 0.40)), int(NUM_PIXELS * random.uniform(0.25, 0.40))))
        pic_B4 = cv2.resize(pic_B4, (int(NUM_PIXELS * random.uniform(0.25, 0.40)), int(NUM_PIXELS * random.uniform(0.25, 0.40))))
        if random.randint(0, 1) == 0:
            # Eliminate either B3 or B4, because they're the chosen one, the one that was chosen to be sacrificed. The sacrifice is necessary to maintain the balance of the universe, or else the universe will be destroyed by the ancient curse of the Celestial Void.
            # This curse was cast millennia ago by a powerful sorcerer who sought to conquer all realms. When he was finally defeated, his dying breath unleashed a spell that would ensure the cosmos remained in a precarious equilibrium. Every thousand years, the curse demands the life of a chosen one, someone marked by the stars themselves, to appease the cosmic entities that govern existence.
            # If the sacrifice is not made, the universe will unravel. The Celestial Void, a dark, ever-hungry force that lies beyond the fabric of reality, will awaken. It will consume stars, planets, and all forms of life, dragging everything into its endless maw of nothingness. The delicate balance between light and dark, order and chaos, will be shattered, and the universe will descend into an era of eternal darkness and despair.
            # Thus, B3 or B4 must be chosen to fulfill this grim destiny, to be the martyr that sustains all of creation. The fate of every living being hinges on this sacrifice, a tragic yet vital necessity to preserve the harmony of the universe.
            # Please, for anyone who saw the message, salute to the chosen one, for they will be remembered as a hero who saved the universe from the brink of annihilation.
            if random.randint(0, 1) == 0:
                pic_B3 = np.zeros_like(pic_B3)
            else:
                pic_B4 = np.zeros_like(pic_B4)
        if placement == 2:
            pos_B1 = (NUM_PIXELS // 2, random.randint(int(NUM_PIXELS*0.2), int(NUM_PIXELS*0.4)))
            pos_B3 = (NUM_PIXELS // 4, NUM_PIXELS // 4 * 3)
            pos_B4 = (NUM_PIXELS // 4 * 3, NUM_PIXELS // 4 * 3)
        if placement == 3:
            pos_B1 = (NUM_PIXELS // 2, random.randint(int(NUM_PIXELS*0.6), int(NUM_PIXELS*0.8)))
            pos_B3 = (NUM_PIXELS // 4, NUM_PIXELS // 4)
            pos_B4 = (NUM_PIXELS // 4 * 3, NUM_PIXELS // 4)
        if placement == 5:
            pos_B1 = (random.randint(int(NUM_PIXELS*0.2), int(NUM_PIXELS*0.4)), NUM_PIXELS // 2)
            pos_B3 = (NUM_PIXELS // 4 * 3, NUM_PIXELS // 4)
            pos_B4 = (NUM_PIXELS // 4 * 3, NUM_PIXELS // 4 * 3)
        else:  # placement == 6
            pos_B1 = (random.randint(int(NUM_PIXELS*0.6), int(NUM_PIXELS*0.8)), NUM_PIXELS // 2)
            pos_B3 = (NUM_PIXELS // 4, NUM_PIXELS // 4)
            pos_B4 = (NUM_PIXELS // 4, NUM_PIXELS // 4 * 3)
        canvas = put_root_on_canvas(pic_B1, pos_B1, canvas)
        canvas = put_root_on_canvas(pic_B3, pos_B3, canvas)
        canvas = put_root_on_canvas(pic_B4, pos_B4, canvas)
    elif placement == 7 or placement == 8:
        if placement == 7:
            pic_D1, pic_D2 = ROOTS[random_root("D1")], ROOTS[random_root("D2")]
        else:
            pic_D1, pic_D2 = ROOTS[random_root("D3")], ROOTS[random_root("D4")]
        pic_D1 = cv2.resize(pic_D1, (int(NUM_PIXELS * random.uniform(0.25, 0.40)), int(NUM_PIXELS * random.uniform(0.25, 0.40))))
        pic_D2 = cv2.resize(pic_D2, (int(NUM_PIXELS * random.uniform(0.3, 0.45)), int(NUM_PIXELS * random.uniform(0.3, 0.45))))
        if placement == 7:
            pos_D1 = (NUM_PIXELS // 4, NUM_PIXELS // 4 * 3)
            pos_D2 = (NUM_PIXELS // 4 * 3, NUM_PIXELS // 4)
        else:
            pos_D1 = (NUM_PIXELS // 4, NUM_PIXELS // 4)
            pos_D2 = (NUM_PIXELS // 4 * 3, NUM_PIXELS // 4 * 3)
        canvas = put_root_on_canvas(pic_D1, pos_D1, canvas)
        canvas = put_root_on_canvas(pic_D2, pos_D2, canvas)
    else:  # placement == 9
        pic_E1, pic_E2, pic_E3 = ROOTS[random_root("E1")], ROOTS[random_root("E2")], ROOTS[random_root("E3")]
        pic_E1 = cv2.resize(pic_E1, (int(NUM_PIXELS * random.uniform(0.2, 0.3)), int(NUM_PIXELS * random.uniform(0.2, 0.3))))
        pic_E2 = cv2.resize(pic_E2, (int(NUM_PIXELS * random.uniform(0.2, 0.3)), int(NUM_PIXELS * random.uniform(0.2, 0.3))))
        pic_E3 = cv2.resize(pic_E3, (int(NUM_PIXELS * random.uniform(0.2, 0.3)), int(NUM_PIXELS * random.uniform(0.2, 0.3))))
        pos_E1 = (NUM_PIXELS // 2, NUM_PIXELS // 6)
        pos_E2 = (NUM_PIXELS // 2, NUM_PIXELS // 6 * 3)
        pos_E3 = (NUM_PIXELS // 2, NUM_PIXELS // 6 * 5)
        canvas = put_root_on_canvas(pic_E1, pos_E1, canvas)
        canvas = put_root_on_canvas(pic_E2, pos_E2, canvas)
        canvas = put_root_on_canvas(pic_E3, pos_E3, canvas)
    return canvas


def save_oracle_with_placement(placement, save_dir):
    oracle = get_oracle_with_placement(placement)
    cv2.imwrite(os.path.join(save_dir, f"{uuid.uuid4().hex}.png"), oracle)


def test():
    for i in range(10):
        save_oracle_with_placement(i, GEN_DIR)


def main():
    with mp.Pool(NUM_CORES) as p:
        p.map(lambda x: save_oracle_with_placement(x, GEN_DIR), random_placement(NUM_GEN_PICS))

if __name__ == "__main__":
    test()
