
import time
import json
import jax
import jax.numpy as jnp
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State
from brax.training.agents.ppo import train as ppo

from config import XML_MODEL, HYPERPARAMS, COMMON_PARAMS, NUM_BENCHMARK_RUNS
from utils import get_hardware_identifier


class CartPole(PipelineEnv):
    def __init__(self, backend: str = 'mjx', **kwargs):
        sys = mjcf.loads(XML_MODEL)
        self.step_dt = 0.02
        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend=backend, n_frames=n_frames)

    def reset(self, rng: jax.Array) -> State:
        key, theta_key, qd_key = jax.random.split(rng, 3)
        theta_init = jax.random.uniform(theta_key, (1,), minval=-0.1, maxval=0.1)[0]
        q_init = jnp.array([0.0, theta_init])
        qd_init = jax.random.uniform(qd_key, (2,), minval=-0.1, maxval=0.1)
        pipeline_state = self.pipeline_init(q_init, qd_init)
        reward, done = jnp.zeros(2)
        observation = self.get_observation(pipeline_state)
        metrics = {'rewards': reward}
        return State(pipeline_state=pipeline_state, obs=observation, reward=reward, done=done, metrics=metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        observation = self.get_observation(pipeline_state)
        x, th = pipeline_state.q
        outside_x = jnp.abs(x) > 1.0
        outside_th = jnp.abs(th) > jnp.pi / 2
        done = jnp.float32(outside_x | outside_th)
        reward = jnp.cos(th)
        state.metrics.update({'rewards': reward})
        return state.replace(pipeline_state=pipeline_state, obs=observation, reward=reward, done=done)

    def get_observation(self, pipeline_state: State) -> jnp.ndarray:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])


def main():
    hardware_name = get_hardware_identifier()
    json_path = f'benchmarks/brax/PPO_original_vs_modified/original_ppo_results_{hardware_name}.json'
    print(f"Saving results to {json_path}")

    try:
        with open(json_path, 'r') as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = []

    results_dict = {r['params']: r for r in all_results}

    for param_name, params in HYPERPARAMS.items():
        print(f"\n--- Testing with '{param_name}' parameters (Original PPO) ---")

        env_for_training = CartPole()

        training_params = {**params, **COMMON_PARAMS}

        times = []
        total_runs = NUM_BENCHMARK_RUNS + 1  # +1 for warmup
        print(f"Running benchmark {total_runs} times (1 warmup + {NUM_BENCHMARK_RUNS} timed)...")

        for i in range(total_runs):
            print(f"  Run {i+1}/{total_runs}...")
            start_time = time.time()
            _, _, _ = ppo.train(
                environment=env_for_training,
                **training_params
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if i == 0:
                print(f"    Warmup run took {elapsed_time:.2f}s (ignored)")
            else:
                print(f"    Run took {elapsed_time:.2f}s")
                times.append(elapsed_time)

        avg_time = sum(times) / len(times) if times else 0
        print(f"Average time for '{param_name}': {avg_time:.2f}s")

        result = {
            'params': param_name,
            'times': times
        }
        results_dict[param_name] = result

    final_results = list(results_dict.values())
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"\nBenchmark complete. Results saved to {json_path}")


if __name__ == '__main__':
    main()
