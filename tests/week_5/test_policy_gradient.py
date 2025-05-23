import unittest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import gymnasium as gym
import numpy as np
import torch
from rl_exercises.week_5 import REINFORCEAgent


class DummyEnv(gym.Env):
    """
    A trivial 1-state, 1-action env that always returns reward=1 and ends immediately.
    Used to test evaluate() deterministically.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        return np.array([0.0], dtype=np.float32), 1.0, True, False, {}


class TestReinforceAgent(unittest.TestCase):
    def test_predict_action_signature(self):
        env = gym.make("CartPole-v1")
        agent = REINFORCEAgent(env, lr=1e-2, gamma=0.99, seed=42)

        state, _ = env.reset(seed=0)
        action, info = agent.predict_action(state, evaluate=False)
        self.assertIsInstance(action, int)
        self.assertIn("log_prob", info)
        self.assertIsInstance(info["log_prob"], torch.Tensor)
        self.assertTrue(info["log_prob"].requires_grad)

        action_eval, info_eval = agent.predict_action(state, evaluate=True)
        self.assertIsInstance(action_eval, int)
        self.assertEqual(info_eval, {})

    def test_update_agent_zero_logprobs(self):
        env = DummyEnv()
        agent = REINFORCEAgent(env, lr=1e-2, gamma=1.0, seed=0)

        state = np.array([0.0], dtype=np.float32)
        next_state = state.copy()
        log_probs = [torch.tensor(0.0, requires_grad=True) for _ in range(2)]
        rewards = [1.0, 1.0]
        batch = [
            (state, 0, rewards[i], next_state, True, {"log_prob": log_probs[i]})
            for i in range(2)
        ]
        loss = agent.update_agent(batch)
        self.assertAlmostEqual(loss, 0.0, places=8)

    def test_update_agent_nonzero_logprobs(self):
        env = DummyEnv()
        agent = REINFORCEAgent(env, lr=1e-2, gamma=1.0, seed=0)

        state = np.array([0.0], dtype=np.float32)
        next_state = state.copy()
        lp0 = torch.tensor(np.log(0.5), requires_grad=True)
        lp1 = torch.tensor(np.log(0.25), requires_grad=True)
        log_probs = [lp0, lp1]
        rewards   = [1.0, 0.0]
        batch = [
            (state, 0, rewards[i], next_state, True, {"log_prob": log_probs[i]})
            for i in range(2)
        ]

        loss = agent.update_agent(batch)

        # recompute returns and two possible normalizations
        returns = agent.compute_returns(rewards)  # tensor([1.0, 0.0])
        centered = returns - returns.mean()

        std_pop  = returns.std(unbiased=False)
        std_samp = returns.std(unbiased=True)
        eps = 1e-8

        norm_pop  = centered / (std_pop  + eps)
        norm_samp = centered / (std_samp + eps)

        lp_tensor = torch.stack(log_probs)

        expected_pop  = float(-torch.sum(lp_tensor * norm_pop))
        expected_samp = float(-torch.sum(lp_tensor * norm_samp))

        # accept either convention
        assert (
            abs(loss - expected_pop) < 1e-6
            or
            abs(loss - expected_samp) < 1e-6
        ), f"loss {loss} did not match either {expected_pop} (pop) or {expected_samp} (samp)"

    def test_evaluate_dummy_env(self):
        dummy = DummyEnv()
        agent = REINFORCEAgent(dummy, lr=1e-2, gamma=0.0, seed=0)
        mean_ret, std_ret = agent.evaluate(dummy, num_episodes=10)
        self.assertAlmostEqual(mean_ret, 1.0, places=8)
        self.assertAlmostEqual(std_ret, 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
