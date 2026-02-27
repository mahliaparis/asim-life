import numpy as np
from scipy import ndimage

MASK_DIM = 8
SCALE = 0.1  # voxel edge length (important)

def _largest_component_bottom_left(mask: np.ndarray):
    labeled, num = ndimage.label(mask.astype(int))
    if num == 0:
        return None

    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    largest = int(np.argmax(sizes) + 1)
    comp = (labeled == largest)

    rows, cols = np.where(comp)
    if len(rows) == 0:
        return None

    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    cropped = comp[min_r:max_r + 1, min_c:max_c + 1].astype(int)

    new_mask = np.zeros((MASK_DIM, MASK_DIM), dtype=int)
    h, w = cropped.shape
    new_mask[MASK_DIM - h:MASK_DIM, 0:w] = cropped
    return new_mask

def sample_mask(p=0.55, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    raw = (rng.random((MASK_DIM, MASK_DIM)) < p).astype(int)
    cleaned = _largest_component_bottom_left(raw)
    if cleaned is None or cleaned.sum() < 3:
        return sample_mask(p=p, rng=rng)
    return cleaned

def mutate_mask(mask: np.ndarray, rng, flips=2, p_on=0.6):
    """
    Flip a few random cells, then keep largest connected component and shift.
    This makes morphology evolution visible (legs emerge, etc.).
    """
    m = mask.copy().astype(int)

    for _ in range(flips):
        r = int(rng.integers(0, MASK_DIM))
        c = int(rng.integers(0, MASK_DIM))
        m[r, c] = 1 if (rng.random() < p_on) else 0

    cleaned = _largest_component_bottom_left(m)
    if cleaned is None or cleaned.sum() < 3:
        return mask.copy()
    return cleaned

def voxel_to_masses(row, col):
    return [
        [row, col],
        [row, col + 1],
        [row + 1, col],
        [row + 1, col + 1],
    ]

def mask_to_robot(mask):
    spring_connections = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [0, 3],
        [1, 2],
    ]
    masses = []
    springs = []
    rows, cols = np.where(mask)
    for row, col in zip(rows, cols):
        row, col = int(row), int(col)
        coords = voxel_to_masses(row, col)

        for c in coords:
            if c not in masses:
                masses.append(c)

        for a, b in spring_connections:
            ca, cb = coords[a], coords[b]
            ia, ib = masses.index(ca), masses.index(cb)
            s = [min(ia, ib), max(ia, ib)]
            if s not in springs:
                springs.append(s)

    masses = np.array(masses, dtype=np.float32)
    springs = np.array(springs, dtype=np.int32)
    return masses, springs

def robot_from_mask(mask: np.ndarray):
    masses, springs = mask_to_robot(mask)
    masses = masses * SCALE
    return {
        "mask": mask.astype(int),
        "n_masses": int(masses.shape[0]),
        "n_springs": int(springs.shape[0]),
        "masses": masses,
        "springs": springs,
    }

def load_random_population(num_robots, p=0.55, seed=0):
    rng = np.random.default_rng(seed)
    pop = []
    for _ in range(num_robots):
        mask = sample_mask(p=p, rng=rng)
        pop.append(robot_from_mask(mask))
    return pop