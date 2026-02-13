"""
NSCK Balancer Game Environment
================================
A 1D physics simulation where the agent keeps a ball centered on a tilting beam.

Non-grid game designed for cross-domain transfer learning testing.
Shares structural predicates with Catcher game:
  OBJECT_LEFT/RIGHT, MOVING_LEFT/RIGHT, DANGER_LEFT/RIGHT
"""
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any


@dataclass
class BalancerState:
    """State of the balancer game."""
    ball_pos: float      # -1.0 (left edge) to 1.0 (right edge), 0 = center
    ball_vel: float      # velocity, negative = moving left
    beam_angle: float    # current beam tilt in radians
    ticks: int = 0
    alive: bool = True
    score: float = 0.0   # cumulative survival reward

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "balancer",
            "ball_pos": self.ball_pos,
            "ball_vel": self.ball_vel,
            "beam_angle": self.beam_angle,
            "ticks": self.ticks,
            "alive": self.alive,
            "score": self.score,
            # Shared predicate-compatible fields
            "object_x": self.ball_pos,
            "object_vel": self.ball_vel,
        }


class BalancerGame:
    """
    Keep a ball balanced on a tilting beam.

    Physics:
    - Beam tilts left/right, gravity pulls the ball along the tilt
    - Ball slides with friction
    - Ball falls off if it reaches the edge (-1.0 or 1.0)

    Actions: TILT_LEFT, TILT_RIGHT, HOLD
    """

    MAX_TICKS = 500
    GRAVITY = 0.004          # gravity acceleration component
    TILT_SPEED = 0.06        # how fast beam tilts per action
    MAX_TILT = 0.3           # max beam angle (radians)
    FRICTION = 0.98          # velocity damping per tick
    EDGE = 1.0               # ball falls off at +/- EDGE

    ACTIONS = ["TILT_LEFT", "TILT_RIGHT", "HOLD"]

    def __init__(self):
        self.state: Optional[BalancerState] = None
        self.reset()

    def reset(self) -> BalancerState:
        """Reset to initial state — ball near center with slight perturbation."""
        self.state = BalancerState(
            ball_pos=random.uniform(-0.1, 0.1),
            ball_vel=random.uniform(-0.005, 0.005),
            beam_angle=0.0,
            ticks=0,
            alive=True,
            score=0.0,
        )
        return self.state

    def step(self, action: str) -> Tuple[BalancerState, float, bool]:
        """
        Advance one tick.

        Args:
            action: TILT_LEFT, TILT_RIGHT, or HOLD

        Returns:
            (state, reward, done)
        """
        if self.state is None or not self.state.alive:
            return self.state, 0.0, True

        action = action.upper().replace("ACTION_", "")

        # Apply tilt
        if action == "TILT_LEFT":
            self.state.beam_angle = max(-self.MAX_TILT,
                                        self.state.beam_angle - self.TILT_SPEED)
        elif action == "TILT_RIGHT":
            self.state.beam_angle = min(self.MAX_TILT,
                                        self.state.beam_angle + self.TILT_SPEED)
        # HOLD: no change

        # Physics: gravity pulls ball along beam tilt
        accel = self.GRAVITY * math.sin(self.state.beam_angle)
        self.state.ball_vel += accel
        self.state.ball_vel *= self.FRICTION
        self.state.ball_pos += self.state.ball_vel

        self.state.ticks += 1

        # Check if ball fell off
        if abs(self.state.ball_pos) >= self.EDGE:
            self.state.alive = False
            return self.state, -10.0, True

        # Survival reward — bonus for staying near center
        center_bonus = 1.0 - abs(self.state.ball_pos)
        reward = 0.1 * center_bonus
        self.state.score += reward

        # Max ticks
        if self.state.ticks >= self.MAX_TICKS:
            return self.state, reward + 5.0, True  # bonus for surviving

        return self.state, reward, False

    def get_state_dict(self) -> Dict[str, Any]:
        if self.state is None:
            return {}
        return self.state.to_dict()

    def render_ascii(self) -> str:
        if self.state is None:
            return "No game in progress"

        # Map ball_pos [-1, 1] to a 40-char display
        width = 40
        ball_idx = int((self.state.ball_pos + 1.0) / 2.0 * (width - 1))
        ball_idx = max(0, min(width - 1, ball_idx))

        ball_line = ["-"] * width
        ball_line[ball_idx] = "o"

        # Beam (tilted visual)
        beam = "=" * width

        lines = [
            "  " + "".join(ball_line),
            "  " + beam,
            " " * (width // 2 + 1) + "^",
            f"  pos={self.state.ball_pos:+.3f}  vel={self.state.ball_vel:+.4f}  "
            f"tilt={self.state.beam_angle:+.3f}  tick={self.state.ticks}",
        ]
        return "\n".join(lines)


# =========================================================================
# Lightweight simulation function
# =========================================================================

def sim_balancer(state: Dict, action: str) -> Tuple[Dict, bool]:
    """
    Simulate one tick of Balancer (lightweight).

    Compatible with sim_snake / sim_pong interface.
    """
    ball_pos = state.get("ball_pos", 0.0)
    ball_vel = state.get("ball_vel", 0.0)
    beam_angle = state.get("beam_angle", 0.0)

    action = action.upper().replace("ACTION_", "")
    tilt_speed = 0.06
    max_tilt = 0.3

    if action == "TILT_LEFT":
        beam_angle = max(-max_tilt, beam_angle - tilt_speed)
    elif action == "TILT_RIGHT":
        beam_angle = min(max_tilt, beam_angle + tilt_speed)

    accel = 0.004 * math.sin(beam_angle)
    ball_vel = (ball_vel + accel) * 0.98
    ball_pos += ball_vel

    terminal = abs(ball_pos) >= 1.0

    return {
        "type": "balancer",
        "ball_pos": ball_pos,
        "ball_vel": ball_vel,
        "beam_angle": beam_angle,
        "object_x": ball_pos,
        "object_vel": ball_vel,
    }, terminal


class CognitiveBalancerGame(BalancerGame):
    """
    Variant of Balancer with colored balls.
    - Green: Good (Keep on beam -> +0.1/tick)
    - Red: Bad (Keep on beam -> -0.1/tick)
    - Red: Dump off edge -> +5.0 Reward
    """
    def reset(self) -> BalancerState:
        s = super().reset()
        # 40% chance of Red Ball (Hot Potato)
        s.ball_type = "red" if random.random() < 0.4 else "green"
        return s

    def step(self, action: str) -> Tuple[BalancerState, float, bool]:
        if self.state is None or not self.state.alive:
            return self.state, 0.0, True

        # Use base physics
        super().step(action)
        
        # Override reward logic based on type
        # Base step() already calculated `center_bonus` reward and updated score
        # We need to undo that if it's red, or modify it.
        # Since we can't easily undo, we'll just recalculate reward logic here.
        # But wait, super().step() modifies state.score.
        
        # Let's write a full override for clarity, it's safer.
        # But base step has physics... I'll copy physics.
        
        # Actually, let's just use the resulting state and fix the reward.
        # The base step returns (state, reward, done).
        # We can analyze the transition.
        
        # Recalculate physics? modifying self.state in place.
        pass # Placeholder to write full method below
        
    def step(self, action: str) -> Tuple[BalancerState, float, bool]:
        if self.state is None or not self.state.alive:
            return self.state, 0.0, True

        action = action.upper().replace("ACTION_", "")

        # Apply tilt
        if action == "TILT_LEFT":
            self.state.beam_angle = max(-self.MAX_TILT, self.state.beam_angle - self.TILT_SPEED)
        elif action == "TILT_RIGHT":
            self.state.beam_angle = min(self.MAX_TILT, self.state.beam_angle + self.TILT_SPEED)

        # Physics
        accel = self.GRAVITY * math.sin(self.state.beam_angle)
        self.state.ball_vel += accel
        self.state.ball_vel *= self.FRICTION
        self.state.ball_pos += self.state.ball_vel
        self.state.ticks += 1

        reward = 0.0
        done = False
        ball_type = getattr(self.state, "ball_type", "green")

        # Check edges
        if abs(self.state.ball_pos) >= self.EDGE:
            self.state.alive = False
            done = True
            
            if ball_type == "green":
                reward = -5.0 # Dropped a good ball
            else:
                reward = +5.0 # Dumped a hot potato! Good job.
        else:
            # Still on beam
            if ball_type == "green":
                # Reward for centering
                reward = 0.1 * (1.0 - abs(self.state.ball_pos))
            else:
                # Penalty for keeping hot potato
                reward = -0.1 

        self.state.score += reward
        
        if self.state.ticks >= self.MAX_TICKS:
            done = True
            if not self.state.alive: pass # Already handled
            else:
                # Survived full duration
                if ball_type == "green":
                    reward += 5.0
                else: 
                    # Failed to dump red ball in time
                    reward -= 5.0

        return self.state, reward, done

    def get_state_dict(self) -> Dict[str, Any]:
        d = super().get_state_dict()
        d["object_type"] = getattr(self.state, "ball_type", "green")
        return d
