"""
US-040: Diagnose and Fix Pixel Agent V1 for LunarLander
=========================================================

Load the V1 pixel agent checkpoint, identify all issues preventing it
from running, fix them, and verify it can collect episodes with pixel
observations and produce 64D latent vectors.

This file does NOT modify the lunar-lander repo. All fixes are implemented
as local adapters/wrappers.
"""

import sys
import os
import numpy as np
import torch
import json
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LUNAR_LANDER_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_PATH = LUNAR_LANDER_ROOT / "trained_agents" / "pixel_lunarlander_best.tar"
RESULTS_DIR = PROJECT_ROOT / "results"
TRAJECTORY_DIR = RESULTS_DIR / "trajectory_data"

# Add lunar-lander source to path
# The checkpoint was pickled with module paths like 'src.active_inference.config',
# so we need BOTH the parent of src (for unpickling) and src itself (for imports)
sys.path.insert(0, str(LUNAR_LANDER_ROOT))
sys.path.insert(0, str(LUNAR_LANDER_ROOT / "src"))

sys.path.insert(0, str(PROJECT_ROOT))
from experiments.utils.results import save_results


def diagnose_checkpoint():
    """Step 1: Load checkpoint and inspect its contents."""
    print("=" * 60)
    print("Step 1: Checkpoint Diagnosis")
    print("=" * 60)

    issues = []

    if not CHECKPOINT_PATH.exists():
        issues.append(f"Checkpoint not found at {CHECKPOINT_PATH}")
        print(f"  FAIL: {issues[-1]}")
        return issues, None

    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Size: {CHECKPOINT_PATH.stat().st_size / 1024:.1f} KB")

    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        print(f"  PASS: Loaded checkpoint successfully")
    except Exception as e:
        issues.append(f"Failed to load checkpoint: {e}")
        print(f"  FAIL: {issues[-1]}")
        return issues, None

    print(f"  Keys: {list(ckpt.keys())}")

    # Inspect config
    config = ckpt.get('config')
    pixel_config = ckpt.get('pixel_config')
    episode = ckpt.get('episode', 'unknown')

    print(f"  Episode: {episode}")
    if config:
        print(f"  Config type: {type(config).__name__}")
        if hasattr(config, 'n_ensemble'):
            print(f"    n_ensemble: {config.n_ensemble}")
        if hasattr(config, 'hidden_dim'):
            print(f"    hidden_dim: {config.hidden_dim}")
        if hasattr(config, 'use_learned_reward'):
            print(f"    use_learned_reward: {config.use_learned_reward}")

    if pixel_config:
        print(f"  PixelConfig type: {type(pixel_config).__name__}")
        if hasattr(pixel_config, 'latent_dim'):
            print(f"    latent_dim: {pixel_config.latent_dim}")
        if hasattr(pixel_config, 'n_frames'):
            print(f"    n_frames: {pixel_config.n_frames}")

    # Inspect encoder weights
    encoder_state = ckpt.get('encoder', {})
    print(f"\n  Encoder state_dict keys: {list(encoder_state.keys())}")
    for key, tensor in encoder_state.items():
        print(f"    {key}: {tensor.shape}")

    # Inspect ensemble weights
    ensemble_state = ckpt.get('ensemble', {})
    n_model_keys = len([k for k in ensemble_state.keys() if k.startswith('models.0.')])
    n_models = len(set(k.split('.')[1] for k in ensemble_state.keys() if k.startswith('models.')))
    print(f"\n  Ensemble: {n_models} models, {n_model_keys} keys per model")

    # Inspect reward model
    reward_state = ckpt.get('reward_model')
    if reward_state:
        print(f"  Reward model: {len(reward_state)} keys")
    else:
        print(f"  Reward model: None")

    return issues, ckpt


def diagnose_imports():
    """Step 2: Test all necessary imports."""
    print("\n" + "=" * 60)
    print("Step 2: Import Diagnosis")
    print("=" * 60)

    issues = []

    # Test cv2
    try:
        import cv2
        print(f"  PASS: cv2 v{cv2.__version__}")
    except ImportError as e:
        issues.append(f"cv2 import failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    # Test gymnasium
    try:
        import gymnasium as gym
        print(f"  PASS: gymnasium v{gym.__version__}")
    except ImportError as e:
        issues.append(f"gymnasium import failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    # Test torch
    try:
        import torch
        print(f"  PASS: torch v{torch.__version__}")
    except ImportError as e:
        issues.append(f"torch import failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    # Test lunar-lander active_inference module
    try:
        from active_inference.config import ActiveInferenceConfig
        print(f"  PASS: ActiveInferenceConfig")
    except ImportError as e:
        issues.append(f"ActiveInferenceConfig import failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    try:
        from active_inference.pixel_config import PixelConfig
        print(f"  PASS: PixelConfig")
    except ImportError as e:
        issues.append(f"PixelConfig import failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    try:
        from active_inference.encoder import CNNEncoder
        print(f"  PASS: CNNEncoder")
    except ImportError as e:
        issues.append(f"CNNEncoder import failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    try:
        from active_inference.frame_stack import FrameStack, preprocess_frame
        print(f"  PASS: FrameStack, preprocess_frame")
    except ImportError as e:
        issues.append(f"FrameStack import failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    try:
        from active_inference.models import EnsembleDynamics, RewardModel
        print(f"  PASS: EnsembleDynamics, RewardModel")
    except ImportError as e:
        issues.append(f"Models import failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    try:
        from active_inference.pixel_agent import PixelActiveInferenceAgent
        print(f"  PASS: PixelActiveInferenceAgent")
    except ImportError as e:
        issues.append(f"PixelActiveInferenceAgent import failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    return issues


def diagnose_agent_loading(ckpt):
    """Step 3: Try to instantiate the agent and load weights."""
    print("\n" + "=" * 60)
    print("Step 3: Agent Loading Diagnosis")
    print("=" * 60)

    issues = []

    if ckpt is None:
        issues.append("No checkpoint to load")
        print(f"  SKIP: {issues[-1]}")
        return issues, None

    from active_inference.config import ActiveInferenceConfig
    from active_inference.pixel_config import PixelConfig
    from active_inference.pixel_agent import PixelActiveInferenceAgent

    config = ckpt.get('config')
    pixel_config = ckpt.get('pixel_config')

    # Check if config objects are compatible
    if not isinstance(config, ActiveInferenceConfig):
        print(f"  WARNING: config is {type(config).__name__}, not ActiveInferenceConfig")
        print(f"    Attempting to use it anyway (dataclass may have evolved)")

    if not isinstance(pixel_config, PixelConfig):
        print(f"  WARNING: pixel_config is {type(pixel_config).__name__}, not PixelConfig")

    n_actions = 4  # LunarLander has 4 discrete actions

    try:
        # Build agent with checkpoint configs
        agent = PixelActiveInferenceAgent(
            n_actions=n_actions,
            config=config,
            pixel_config=pixel_config,
        )
        print(f"  PASS: Agent instantiated")
    except Exception as e:
        issues.append(f"Agent instantiation failed: {e}")
        print(f"  FAIL: {issues[-1]}")

        # Try with fresh configs matching checkpoint architecture
        print("  Attempting recovery with fresh configs...")
        try:
            # Extract architecture from checkpoint weights
            encoder_state = ckpt.get('encoder', {})
            ensemble_state = ckpt.get('ensemble', {})

            # Infer n_ensemble from checkpoint
            model_indices = set()
            for k in ensemble_state.keys():
                if k.startswith('models.'):
                    model_indices.add(int(k.split('.')[1]))
            n_ensemble = len(model_indices)

            # Infer hidden_dim from dynamics weights
            for k, v in ensemble_state.items():
                if 'trunk.0.weight' in k:
                    hidden_dim = v.shape[0]
                    input_dim = v.shape[1]
                    break

            latent_dim = input_dim - n_actions

            print(f"    Inferred: n_ensemble={n_ensemble}, hidden_dim={hidden_dim}, latent_dim={latent_dim}")

            fresh_config = ActiveInferenceConfig(
                n_ensemble=n_ensemble,
                hidden_dim=hidden_dim,
                use_learned_reward=ckpt.get('reward_model') is not None,
            )
            fresh_pixel_config = PixelConfig(latent_dim=latent_dim)

            agent = PixelActiveInferenceAgent(
                n_actions=n_actions,
                config=fresh_config,
                pixel_config=fresh_pixel_config,
            )
            print(f"  PASS: Agent instantiated with fresh configs")
            config = fresh_config
            pixel_config = fresh_pixel_config
        except Exception as e2:
            issues.append(f"Recovery also failed: {e2}")
            print(f"  FAIL: {issues[-1]}")
            return issues, None

    # Load weights
    try:
        agent.load(str(CHECKPOINT_PATH))
        print(f"  PASS: Weights loaded from checkpoint")
    except Exception as e:
        issues.append(f"Weight loading failed: {e}")
        print(f"  FAIL: {issues[-1]}")

        # Try manual loading
        print("  Attempting manual state_dict loading...")
        try:
            agent.encoder.load_state_dict(ckpt['encoder'])
            agent.ensemble.load_state_dict(ckpt['ensemble'])
            if agent.reward_model and ckpt.get('reward_model'):
                agent.reward_model.load_state_dict(ckpt['reward_model'])
            print(f"  PASS: Manual weight loading succeeded")
        except Exception as e2:
            issues.append(f"Manual loading also failed: {e2}")
            print(f"  FAIL: {issues[-1]}")
            return issues, None

    return issues, agent


def diagnose_environment():
    """Step 4: Test environment setup with pixel rendering."""
    print("\n" + "=" * 60)
    print("Step 4: Environment Diagnosis")
    print("=" * 60)

    issues = []

    import gymnasium as gym

    try:
        env = gym.make('LunarLander-v3', render_mode='rgb_array')
        print(f"  PASS: LunarLander-v3 created with render_mode='rgb_array'")
    except Exception as e:
        issues.append(f"Environment creation failed: {e}")
        print(f"  FAIL: {issues[-1]}")
        return issues, None

    try:
        obs, info = env.reset(seed=42)
        print(f"  PASS: Environment reset (obs shape: {np.array(obs).shape})")
    except Exception as e:
        issues.append(f"Environment reset failed: {e}")
        print(f"  FAIL: {issues[-1]}")
        env.close()
        return issues, None

    try:
        frame = env.render()
        print(f"  PASS: Rendered frame shape: {frame.shape}, dtype: {frame.dtype}")
    except Exception as e:
        issues.append(f"Rendering failed: {e}")
        print(f"  FAIL: {issues[-1]}")
        env.close()
        return issues, None

    # Test frame preprocessing
    try:
        from active_inference.frame_stack import preprocess_frame
        processed = preprocess_frame(frame)
        print(f"  PASS: Preprocessed frame: {processed.shape}, range [{processed.min():.2f}, {processed.max():.2f}]")
    except Exception as e:
        issues.append(f"Frame preprocessing failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    # Test frame stacking
    try:
        from active_inference.frame_stack import FrameStack
        from active_inference.pixel_config import PixelConfig

        fs = FrameStack(PixelConfig())
        stacked = fs.get_observation(frame)
        print(f"  PASS: Stacked frames: {stacked.shape}, range [{stacked.min():.2f}, {stacked.max():.2f}]")
    except Exception as e:
        issues.append(f"Frame stacking failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    # Take a step
    try:
        obs2, reward, terminated, truncated, info = env.step(0)  # noop
        frame2 = env.render()
        print(f"  PASS: Environment step works (reward={reward:.2f})")
    except Exception as e:
        issues.append(f"Environment step failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    env.close()
    return issues, None


def diagnose_encoding(agent):
    """Step 5: Test encoding pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("Step 5: Encoding Pipeline Diagnosis")
    print("=" * 60)

    issues = []

    if agent is None:
        issues.append("No agent to test")
        print(f"  SKIP: {issues[-1]}")
        return issues

    import gymnasium as gym
    from active_inference.frame_stack import FrameStack
    from active_inference.pixel_config import PixelConfig

    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    frame = env.render()

    # Build frame stack
    fs = FrameStack(PixelConfig())
    stacked = fs.get_observation(frame)

    # Test encoding
    try:
        latent = agent.encode_frames(stacked)
        print(f"  PASS: Encoded latent: shape={latent.shape}, "
              f"range=[{latent.min():.3f}, {latent.max():.3f}]")
        print(f"    Mean: {latent.mean():.4f}, Std: {latent.std():.4f}")
    except Exception as e:
        issues.append(f"Encoding failed: {e}")
        print(f"  FAIL: {issues[-1]}")
        env.close()
        return issues

    # Test action selection
    try:
        action = agent.select_action(stacked, epsilon=0.0)
        print(f"  PASS: Selected action: {action}")
    except Exception as e:
        issues.append(f"Action selection failed: {e}")
        print(f"  FAIL: {issues[-1]}")

    env.close()
    return issues


def run_episode(agent, max_steps=1000, verbose=True):
    """Step 6: Run a full episode and collect data."""
    print("\n" + "=" * 60)
    print("Step 6: Full Episode Test")
    print("=" * 60)

    import gymnasium as gym
    from active_inference.frame_stack import FrameStack

    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    obs, info = env.reset(seed=42)

    agent.reset_frame_stack()
    frame = env.render()
    stacked = agent.frame_stack.get_observation(frame)

    total_reward = 0.0
    latents = []
    states = []
    actions = []

    for step in range(max_steps):
        # Get true state
        true_state = np.array(obs)
        states.append(true_state)

        # Encode and select action
        latent = agent.encode_frames(stacked)
        latents.append(latent.copy())

        # Use epsilon=0.1 for some exploration
        action = agent.select_action(stacked, epsilon=0.1)
        actions.append(action)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Get next frame and stack
        frame = env.render()
        stacked = agent.frame_stack.get_observation(frame)

        done = terminated or truncated
        if done:
            break

    env.close()

    latents = np.array(latents)
    states = np.array(states)
    actions = np.array(actions)

    print(f"  Episode completed: {step+1} steps, total reward: {total_reward:.1f}")
    print(f"  Latents: shape={latents.shape}")
    print(f"  States: shape={states.shape}")
    print(f"  Final state: x={states[-1,0]:.2f}, y={states[-1,1]:.2f}")
    print(f"  Latent statistics:")
    print(f"    Mean: {latents.mean():.4f}")
    print(f"    Std:  {latents.std():.4f}")
    print(f"    Active dims (std > 0.01): {np.sum(latents.std(axis=0) > 0.01)}/64")

    return {
        'total_reward': total_reward,
        'n_steps': step + 1,
        'latents': latents,
        'states': states,
        'actions': actions,
    }


def run_us040():
    """Run full US-040 diagnostic pipeline."""
    print("=" * 60)
    print("US-040: Pixel Agent V1 Diagnosis")
    print("=" * 60)

    all_issues = []
    report = {}

    # Step 1: Checkpoint
    issues, ckpt = diagnose_checkpoint()
    all_issues.extend(issues)
    report['checkpoint'] = {
        'path': str(CHECKPOINT_PATH),
        'exists': CHECKPOINT_PATH.exists(),
        'issues': issues,
        'loaded': ckpt is not None,
    }
    if ckpt:
        report['checkpoint']['keys'] = list(ckpt.keys())
        report['checkpoint']['episode'] = ckpt.get('episode', 'unknown')

    # Step 2: Imports
    issues = diagnose_imports()
    all_issues.extend(issues)
    report['imports'] = {'issues': issues, 'all_pass': len(issues) == 0}

    # Step 3: Agent loading
    issues, agent = diagnose_agent_loading(ckpt)
    all_issues.extend(issues)
    report['agent_loading'] = {
        'issues': issues,
        'loaded': agent is not None,
    }

    # Step 4: Environment
    issues, _ = diagnose_environment()
    all_issues.extend(issues)
    report['environment'] = {'issues': issues, 'all_pass': len(issues) == 0}

    # Step 5: Encoding
    issues = diagnose_encoding(agent)
    all_issues.extend(issues)
    report['encoding'] = {'issues': issues, 'all_pass': len(issues) == 0}

    # Step 6: Full episode
    if agent is not None and len(all_issues) == 0:
        episode_data = run_episode(agent)
        report['episode'] = {
            'total_reward': episode_data['total_reward'],
            'n_steps': episode_data['n_steps'],
            'latent_shape': list(episode_data['latents'].shape),
            'latent_active_dims': int(np.sum(episode_data['latents'].std(axis=0) > 0.01)),
        }

        # Save trajectory data for TB analysis
        TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
        np.save(TRAJECTORY_DIR / 'pixel_latents.npy', episode_data['latents'])
        np.save(TRAJECTORY_DIR / 'pixel_states.npy', episode_data['states'])
        np.save(TRAJECTORY_DIR / 'pixel_actions.npy', episode_data['actions'])
        print(f"\n  Saved trajectory data to {TRAJECTORY_DIR}")

        # Run 5 episodes for more data
        print("\n  Running 4 additional episodes for statistics...")
        all_rewards = [episode_data['total_reward']]
        all_latents = [episode_data['latents']]
        all_states = [episode_data['states']]

        for ep in range(4):
            import gymnasium as gym
            env = gym.make('LunarLander-v3', render_mode='rgb_array')
            obs, info = env.reset(seed=100 + ep)
            agent.reset_frame_stack()
            frame = env.render()
            stacked = agent.frame_stack.get_observation(frame)

            ep_reward = 0.0
            ep_latents = []
            ep_states = []

            for step in range(1000):
                ep_states.append(np.array(obs))
                latent = agent.encode_frames(stacked)
                ep_latents.append(latent.copy())
                action = agent.select_action(stacked, epsilon=0.1)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                frame = env.render()
                stacked = agent.frame_stack.get_observation(frame)
                if terminated or truncated:
                    break

            env.close()
            all_rewards.append(ep_reward)
            all_latents.append(np.array(ep_latents))
            all_states.append(np.array(ep_states))
            print(f"    Episode {ep+2}: {step+1} steps, reward={ep_reward:.1f}")

        # Combine all episodes
        combined_latents = np.vstack(all_latents)
        combined_states = np.vstack(all_states)
        np.save(TRAJECTORY_DIR / 'pixel_latents_5ep.npy', combined_latents)
        np.save(TRAJECTORY_DIR / 'pixel_states_5ep.npy', combined_states)

        report['multi_episode'] = {
            'n_episodes': 5,
            'rewards': all_rewards,
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'total_transitions': combined_latents.shape[0],
            'latent_active_dims': int(np.sum(combined_latents.std(axis=0) > 0.01)),
        }

        print(f"\n  5-episode summary:")
        print(f"    Mean reward: {np.mean(all_rewards):.1f} +/- {np.std(all_rewards):.1f}")
        print(f"    Total transitions: {combined_latents.shape[0]}")
        print(f"    Active latent dims: {np.sum(combined_latents.std(axis=0) > 0.01)}/64")
    elif len(all_issues) > 0:
        print("\n" + "=" * 60)
        print("Skipping full episode due to earlier issues.")
        print("=" * 60)

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    if len(all_issues) == 0:
        print("  ALL CHECKS PASSED")
        print("  The V1 pixel agent loads and runs without issues.")
        report['status'] = 'PASS'
        report['summary'] = 'V1 pixel agent loads and runs. No fixes needed.'
    else:
        print(f"  {len(all_issues)} ISSUES FOUND:")
        for i, issue in enumerate(all_issues, 1):
            print(f"    {i}. {issue}")
        report['status'] = 'ISSUES_FOUND'
        report['summary'] = f'{len(all_issues)} issues found during diagnosis.'

    report['all_issues'] = all_issues

    # Save results
    save_results('pixel_agent_diagnosis', report,
                 {'checkpoint': str(CHECKPOINT_PATH), 'agent_version': 'V1'},
                 notes='US-040: Pixel agent V1 diagnosis. '
                       'Tested checkpoint loading, imports, encoding, and episode collection.')

    return report


if __name__ == '__main__':
    report = run_us040()
