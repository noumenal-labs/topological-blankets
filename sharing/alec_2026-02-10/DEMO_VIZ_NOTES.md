# Demo Visualization Notes for Alec (2026-02-10)

## Assets received
- `wandb_compare_reach.png` - Reach success rate: active inference vs DDPG+HER
- `eval_interactive_rollout.gif` - Action trajectories + ensemble rollout

## Reach benchmark plot
The headline number is strong: ~6x better sample efficiency (100% at ~1,500 steps vs ~10,000 for DDPG+HER). Once the full line finishes, consider adding a vertical annotation at the crossover point where our model hits 100%.

## 3-GIF partition

The agreed split for both the action trajectory and imagined rollout GIFs:

| GIF | Content | Suggested duration |
|-----|---------|-------------------|
| Phase 1 | Normal autonomous behavior (pre-perturbation) | 3-5s at 3 fps |
| Phase 2 | Uncertainty spike + human intervention | 5-8s at 3 fps |
| Phase 3 | Return to autonomous completion | 3-5s at 3 fps |

A splitting utility is included: `split_demo_gif.py`. Usage:

```bash
python split_demo_gif.py eval_interactive_rollout.gif --fps 3 --split1 0.25 --split2 0.65
```

Adjust `--split1` and `--split2` to match the actual perturbation and release frame ratios. The script adds a color-coded border (green/red/green) and phase label overlay to each segment.

## Action trajectory GIF improvements

The current single-GIF playback is too fast. Suggestions:

1. **Lower FPS**: 3-5 fps instead of 20. Each frame should be readable for ~300ms.

2. **Trail rendering**: Instead of rendering every action vector independently, accumulate a fading trail of the last 5-10 actions. Use decreasing opacity (100% current, 20% oldest). This gives a sense of motion direction without clutter.

3. **Phase color-coding**: Color the gripper trajectory and action arrows by phase:
   - Green: autonomous
   - Yellow: elevated uncertainty (advisory)
   - Red: human takeover
   - Green again: resumed

4. **Text overlay**: Small label in corner showing current phase name and step number. The Wednesday demo script already classifies each step into phases; this metadata can drive the overlay.

5. **Gripper focus**: If the full scene is too busy, consider a zoomed inset panel showing just the gripper neighborhood with action arrows.

## Imagined rollout (CEM Ensemble Forecast) improvements

The 6-panel dark-background design already looks good. Suggestions:

1. **Vertical perturbation marker**: Add a red dashed vertical line at the perturbation timestep across all 6 panels. This anchors the viewer: "here is when things went wrong."

2. **Disagreement highlight**: In Phase 2, the ±1sigma bands should visibly fan out. Consider briefly flashing or pulsing the band color at the moment of maximum disagreement.

3. **Panel grouping**: The 6 panels naturally split into two groups matching the TB partition:
   - Top row (blue/orange/green): gripper obs[0:3] = Object 0 (gripper)
   - Bottom row (yellow/red/blue): achieved goal = Object 1 (manipulated object)

   Consider adding a subtle group label: "Gripper (TB Object 0)" and "Object (TB Object 1)" to connect the visualization to the TB discovery narrative.

4. **Mode indicator**: The green "AUTO" badge in the corner is good. Add "HUMAN" (red) during intervention and a transition animation between them.

## Training status

Our CPU training is running (iteration 2/100 as of this writing). Each iteration takes ~10 min on CPU, so full run completes overnight. Alec's GPU runs will be faster. Once a checkpoint is available, the Wednesday demo pipeline is validated and ready:

```bash
python ralph/experiments/wednesday_demo.py --planner tb --max-steps 50
```

Dry-run already passes all 6 acceptance criteria at 50 steps.
