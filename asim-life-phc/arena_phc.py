from flask import Flask, render_template, Response
from argparse import ArgumentParser
from simulator import Simulator
from utils import load_config
from robot import load_robots
import numpy as np
import json, time, threading

TARGET_FPS = 60.0

app = Flask(
    __name__,
    template_folder="visualizer/templates",
    static_folder="visualizer/static",
)

state_lock = threading.Lock()
app_state = {
    "t": 0,
    "generation": 0,
    "best_fitness": 0.0,
    "mean_fitness": 0.0,
}

# --------- helpers ---------

def compute_generation_fitness(sim: Simulator) -> np.ndarray:
    """
    Fitness = COM_x(final) - COM_x(initial) for each sim.
    Requires compute_com(steps) to have been called.
    """
    centers = sim.center.to_numpy()  # shape (n_sims, steps+1, 2)
    x0 = centers[:, 0, 0]
    xt = centers[:, sim.steps[None], 0]
    return (xt - x0).astype(np.float32)

def mutate_params(params, sigma, rng):
    children = []
    for p in params:
        children.append({
            "weights1": (p["weights1"] + rng.normal(0.0, sigma, p["weights1"].shape)).astype(np.float32),
            "weights2": (p["weights2"] + rng.normal(0.0, sigma, p["weights2"].shape)).astype(np.float32),
            "biases1":  (p["biases1"]  + rng.normal(0.0, sigma, p["biases1"].shape)).astype(np.float32),
            "biases2":  (p["biases2"]  + rng.normal(0.0, sigma, p["biases2"].shape)).astype(np.float32),
        })
    return children

@app.route("/")
def index():
    return render_template("index.html")

# --------- simulation stepper ---------

def step_simulation_once():
    """
    Runs one physics tick for ALL robots.
    When a generation ends, we:
      - compute COM at final step
      - evaluate children fitness
      - PHC select: if child better, replace parent
      - create next generation children
      - reset physics
    """
    global simulator, n_sims, max_steps
    global parent_params, rng, sigma
    global child_params
    global best_parent_fitness  # per-lane current parent fitness baseline

    t = app_state["t"]

    # generation ended?
    if t >= max_steps:
        # IMPORTANT: fill COM at final step before fitness eval
        simulator.compute_com(max_steps)

        # child fitness this generation
        fitness_child = compute_generation_fitness(simulator)

        # PHC selection: compare child to parent baseline
        improved = fitness_child > best_parent_fitness
        for i in range(n_sims):
            if improved[i]:
                parent_params[i] = child_params[i]
                best_parent_fitness[i] = fitness_child[i]

        # update stats
        with state_lock:
            app_state["generation"] += 1
            app_state["best_fitness"] = float(np.max(best_parent_fitness))
            app_state["mean_fitness"] = float(np.mean(best_parent_fitness))
            app_state["t"] = 0

        # optional: anneal mutation slowly (keeps progress but stabilizes later)
        sigma = max(0.02, sigma * 0.997)

        # next generation: mutate from current parents
        child_params = mutate_params(parent_params, sigma, rng)
        simulator.set_control_params(list(range(n_sims)), child_params)
        

        # reset physics state for next generation
        simulator.reinitialize_robots()
        return

    # normal step
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
        # topology once
        topology = {
            "type": "topology",
            "n_sims": int(n_sims),
            "springs": springs_template.tolist(),
            "n_masses": int(n_masses_cached),
            "n_springs": int(n_springs_cached),
            "lane_spacing": float(lane_spacing),
            "finish_x": float(finish_x),
        }
        yield f"data: {json.dumps(topology)}\n\n"

        fps_samples = []
        last_fps_update = time.perf_counter()
        actual_fps = 0.0

        while True:
            frame_start = time.perf_counter()
            target_interval = 1.0 / TARGET_FPS

            # advance simulation
            step_simulation_once()

            # current time index after step/reset
            t = app_state["t"]

            # positions for all robots at t
            pos_all = simulator.x.to_numpy()[:, t, :n_masses_cached, :]

            # compute per-robot COM x NOW (for display)
            com_x = pos_all[:, :, 0].mean(axis=1)

            with state_lock:
                payload = {
                    "type": "step",
                    "t": int(app_state["t"]),
                    "generation": int(app_state["generation"]),
                    "best_fitness": float(app_state["best_fitness"]),
                    "mean_fitness": float(app_state["mean_fitness"]),
                    "positions": pos_all.tolist(),
                    "com_x": com_x.tolist(),
                    "fps": float(actual_fps),
                }

            yield f"data: {json.dumps(payload)}\n\n"

            # fps control
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

# --------- main ---------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--sigma", type=float, default=0.08, help="Mutation strength (try 0.05–0.12)")
    parser.add_argument("--lane_spacing", type=float, default=1.6)
    parser.add_argument("--finish_x", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=400, help="Override sim_steps for generation length")
    args = parser.parse_args()

    config = load_config(args.config)

    # Make generations short + visible
    config["simulator"]["sim_steps"] = int(args.steps)

    # Helpful physics defaults (more traction + less flailing)
    config["simulator"]["friction"] = 0.98
    config["simulator"]["drag_damping"] = 14.0
    config["simulator"]["springA"] = 0.18

    n_sims = int(config["simulator"]["n_sims"])
    sigma = float(args.sigma)
    lane_spacing = float(args.lane_spacing)
    finish_x = float(args.finish_x)

    # IMPORTANT: clone ONE body across population (your robot.load_robots does this if you changed it)
    np.random.seed(config["seed"])
    robots = load_robots(n_sims)

    max_n_masses = max(r["n_masses"] for r in robots)
    max_n_springs = max(r["n_springs"] for r in robots)
    config["simulator"]["n_masses"] = max_n_masses
    config["simulator"]["n_springs"] = max_n_springs

    simulator = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=False,
    )

    masses = [r["masses"] for r in robots]
    springs = [r["springs"] for r in robots]
    simulator.initialize(masses, springs)

    max_steps = simulator.steps[None]
    n_masses_cached = int(simulator.n_masses[0])
    n_springs_cached = int(simulator.n_springs[0])
    springs_template = robots[0]["springs"]

    # Parent controllers = random init from simulator.initialize()
    parent_params = simulator.get_control_params(list(range(n_sims)))

    # Parent baseline fitness per lane
    best_parent_fitness = np.full((n_sims,), -1e9, dtype=np.float32)

    rng = np.random.default_rng(config["seed"])

    # Start with mutated children
    child_params = mutate_params(parent_params, sigma, rng)
    simulator.set_control_params(list(range(n_sims)), child_params)

    print(f"\nEvolution Arena PHC running at http://localhost:{args.port}")
    print("Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=False, use_reloader=False)