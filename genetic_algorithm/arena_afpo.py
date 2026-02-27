from flask import Flask, render_template, Response
from argparse import ArgumentParser
from simulator import Simulator
from utils import load_config
from robot import load_random_population, mutate_mask, robot_from_mask, sample_mask
import numpy as np
import json, time, threading
import os

TARGET_FPS = 60.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "visualizer", "templates"),
    static_folder=os.path.join(BASE_DIR, "visualizer", "static"),
)

state_lock = threading.Lock()
app_state = {
    "t": 0,
    "generation": 0,
    "best_fitness": 0.0,
    "mean_fitness": 0.0,
}

# ---------- AFPO Pareto helpers ----------

def dominates(a, b):
    # fitness maximize, age minimize
    fa, aa = a["fitness"], a["age"]
    fb, ab = b["fitness"], b["age"]
    return (fa >= fb and aa <= ab) and (fa > fb or aa < ab)

def pareto_downselect(pool, N, rng):
    survivors = []
    remaining = pool[:]

    while len(survivors) < N and len(remaining) > 0:
        nondom = []
        for i in range(len(remaining)):
            dom = False
            for j in range(len(remaining)):
                if i == j:
                    continue
                if dominates(remaining[j], remaining[i]):
                    dom = True
                    break
            if not dom:
                nondom.append(remaining[i])

        if len(survivors) + len(nondom) <= N:
            survivors.extend(nondom)
            nondom_ids = set(id(x) for x in nondom)
            remaining = [x for x in remaining if id(x) not in nondom_ids]
        else:
            need = N - len(survivors)
            idx = rng.choice(len(nondom), size=need, replace=False)
            survivors.extend([nondom[i] for i in idx])
            break

    return survivors

# ---------- fitness + mutation ----------

def compute_fitness(sim: Simulator) -> np.ndarray:
    centers = sim.center.to_numpy()  # (n_sims, steps+1, 2)
    x0 = centers[:, 0, 0]
    xt = centers[:, sim.steps[None], 0]
    return (xt - x0).astype(np.float32)

def mutate_controller(params, sigma, rng):
    return {
        "weights1": (params["weights1"] + rng.normal(0.0, sigma, params["weights1"].shape)).astype(np.float32),
        "weights2": (params["weights2"] + rng.normal(0.0, sigma, params["weights2"].shape)).astype(np.float32),
        "biases1":  (params["biases1"]  + rng.normal(0.0, sigma, params["biases1"].shape)).astype(np.float32),
        "biases2":  (params["biases2"]  + rng.normal(0.0, sigma, params["biases2"].shape)).astype(np.float32),
    }

def eval_rollout(sim: Simulator):
    steps = sim.steps[None]
    sim.reinitialize_robots()
    for t in range(steps):
        sim.compute_com(t)
        sim.nn1(t)
        sim.nn2(t)
        sim.apply_spring_force(t)
        sim.advance(t + 1)
    sim.compute_com(steps)
    return compute_fitness(sim)

def init_sim_for_inds(sim: Simulator, inds, load_ctrl=True):
    """
    Initialize simulator with the geometry for inds.
    If load_ctrl=True, also load each individual's control_params.
    """
    masses = [ind["masses"] for ind in inds]
    springs = [ind["springs"] for ind in inds]
    sim.initialize(masses, springs)

    if load_ctrl:
        sim.set_control_params(list(range(len(inds))), [ind["control_params"] for ind in inds])

def make_children(population, sigma, rng, p_mask):
    """
    Create N children total (keeps pool size 2N).
    Child 0 is an immigrant (random morphology, random controller mutation).
    Children 1..N-1 are mutated from parents.
    """
    N = len(population)
    children = []

    # immigrant
    imm_mask = sample_mask(p=p_mask, rng=rng)
    imm_robot = robot_from_mask(imm_mask)
    base = population[int(rng.integers(0, N))]["control_params"]
    imm_ctrl = mutate_controller(base, sigma=1.0, rng=rng)

    children.append({
        **imm_robot,
        "control_params": imm_ctrl,
        "age": 0,
        "fitness": 0.0,
    })

    # mutated offspring
    order = rng.permutation(N)
    for k in range(1, N):
        parent = population[int(order[k])]
        new_mask = mutate_mask(parent["mask"], rng=rng, flips=2, p_on=0.6)
        child_robot = robot_from_mask(new_mask)
        child_ctrl = mutate_controller(parent["control_params"], sigma=sigma, rng=rng)

        children.append({
            **child_robot,
            "control_params": child_ctrl,
            "age": 0,
            "fitness": 0.0,
        })

    return children

# ---------- Flask routes ----------

@app.route("/")
def index():
    return render_template("index.html")

def step_simulation_once():
    global simulator, population, sigma, rng, p_mask
    global n_sims, max_steps

    t = app_state["t"]

    # generation end
    if t >= max_steps:
        simulator.compute_com(max_steps)
        parent_fit = compute_fitness(simulator)

        # set parent fitness + age++
        for i in range(n_sims):
            population[i]["fitness"] = float(parent_fit[i])
            population[i]["age"] += 1

        # children
        children = make_children(population, sigma=sigma, rng=rng, p_mask=p_mask)

        # evaluate children in same sim (N sized)
        init_sim_for_inds(simulator, children)
        child_fit = eval_rollout(simulator)
        for i in range(n_sims):
            children[i]["fitness"] = float(child_fit[i])

        # pool and downselect
        pool = population + children
        survivors = pareto_downselect(pool, n_sims, rng)

        fits = np.array([s["fitness"] for s in survivors], dtype=np.float32)
        with state_lock:
            app_state["generation"] += 1
            app_state["best_fitness"] = float(np.max(fits))
            app_state["mean_fitness"] = float(np.mean(fits))
            app_state["t"] = 0

        # anneal sigma slightly (stabilize later gens)
        sigma = max(0.03, sigma * 0.997)

        # new population
        population = survivors

        # load survivors into sim for next streamed gen
        init_sim_for_inds(simulator, population)
        simulator.reinitialize_robots()
        return

    # normal step for current population
    simulator.compute_com(t)
    simulator.nn1(t)
    simulator.nn2(t)
    simulator.apply_spring_force(t)
    simulator.advance(t + 1)

    with state_lock:
        app_state["t"] = t + 1

@app.route("/stream")
def stream():
    def event_stream():
        # initial topology
        topo = {
            "type": "topology",
            "n_sims": int(n_sims),
            "lane_spacing": float(lane_spacing),
            "finish_x": float(finish_x),
        }
        yield f"data: {json.dumps(topo)}\n\n"

        fps_samples = []
        last_fps_update = time.perf_counter()
        actual_fps = 0.0

        while True:
            frame_start = time.perf_counter()
            target_interval = 1.0 / TARGET_FPS

            step_simulation_once()
            t = app_state["t"]

            # positions (allocated to 81 masses, but each robot uses n_masses)
            pos_all = simulator.x.to_numpy()[:, t, :, :]  # (N, 81, 2)
            com_x = pos_all[:, :, 0].mean(axis=1)

            ages = [int(ind["age"]) for ind in population]
            fits = [float(ind["fitness"]) for ind in population]
            n_masses_list = [int(ind["n_masses"]) for ind in population]
            springs_all = [ind["springs"].tolist() for ind in population]

            payload = {
                "type": "step",
                "t": int(app_state["t"]),
                "generation": int(app_state["generation"]),
                "best_fitness": float(app_state["best_fitness"]),
                "mean_fitness": float(app_state["mean_fitness"]),
                "positions": pos_all.tolist(),
                "com_x": com_x.tolist(),
                "ages": ages,
                "fitnesses": fits,
                "n_masses_list": n_masses_list,
                "springs_all": springs_all,
                "fps": float(actual_fps),
            }
            yield f"data: {json.dumps(payload)}\n\n"

            work_time = time.perf_counter() - frame_start
            sleep_time = target_interval - work_time
            if sleep_time > 0.001:
                time.sleep(sleep_time)

            total_frame_time = time.perf_counter() - frame_start
            if total_frame_time > 0:
                fps_samples.append(1.0 / total_frame_time)

            now = time.perf_counter()
            if now - last_fps_update >= 0.5:
                if fps_samples:
                    actual_fps = sum(fps_samples) / len(fps_samples)
                    fps_samples = []
                last_fps_update = now

    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response

# ---------- main ----------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--lane_spacing", type=float, default=1.6)
    parser.add_argument("--finish_x", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--p_mask", type=float, default=0.55)
    args = parser.parse_args()

    config = load_config(args.config)

    # enforce 10 lanes on screen
    config["simulator"]["n_sims"] = 10
    n_sims = int(config["simulator"]["n_sims"])

    # fixed max sizes for any 8x8 morphology
    config["simulator"]["n_masses"] = 81
    config["simulator"]["n_springs"] = 272
    config["simulator"]["sim_steps"] = int(args.steps)

    # physics tuned for traction
    config["simulator"]["friction"] = 0.98
    config["simulator"]["drag_damping"] = 14.0
    config["simulator"]["springA"] = 0.18

    lane_spacing = float(args.lane_spacing)
    finish_x = float(args.finish_x)
    sigma = float(args.sigma)
    p_mask = float(args.p_mask)

    rng = np.random.default_rng(config["seed"])

    simulator = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=False,
    )
    max_steps = simulator.steps[None]

    # initial random morphologies
    population = load_random_population(n_sims, p=p_mask, seed=config["seed"])

    # init controller params by calling simulator.initialize once
    init_sim_for_inds(simulator, population, load_ctrl=False)
    init_params = simulator.get_control_params(list(range(n_sims)))
    for i in range(n_sims):
        population[i]["control_params"] = init_params[i]
        population[i]["age"] = 0
        population[i]["fitness"] = 0.0

    simulator.set_control_params(list(range(n_sims)), [ind["control_params"] for ind in population])
    simulator.reinitialize_robots()

    print(f"\nAFPO Morphology Evolution Arena running at http://localhost:{args.port}")
    print("Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=False, use_reloader=False)