"""
NSCK Catcher Game Environment
================================
Move a paddle to catch falling objects before they hit the ground.

Non-grid game designed for cross-domain transfer learning testing.
Shares structural predicates with Balancer game:
  OBJECT_LEFT/RIGHT, MOVING_LEFT/RIGHT, DANGER_LEFT/RIGHT
"""
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class FallingObject:
    """A single falling object."""
    x: float         # horizontal position (0.0 to 1.0)
    y: float         # vertical position (1.0 = top, 0.0 = ground)
    vx: float        # horizontal drift velocity
    vy: float        # fall speed (negative = falling down)


@dataclass
class CatcherState:
    """State of the catcher game."""
    paddle_x: float       # paddle center position (0.0 to 1.0)
    paddle_width: float   # paddle half-width
    objects: List[FallingObject] = field(default_factory=list)
    catches: int = 0
    misses: int = 0
    ticks: int = 0
    done: bool = False

    @property
    def score(self) -> int:
        return self.catches

    @property
    def nearest_object(self) -> Optional[FallingObject]:
        """Get the lowest (closest to ground) active object."""
        if not self.objects:
            return None
        return min(self.objects, key=lambda o: o.y)

    def to_dict(self) -> Dict[str, Any]:
        nearest = self.nearest_object
        return {
            "type": "catcher",
            "paddle_x": self.paddle_x,
            "paddle_width": self.paddle_width,
            "catches": self.catches,
            "misses": self.misses,
            "ticks": self.ticks,
            "done": self.done,
            "score": self.score,
            "num_objects": len(self.objects),
            # Nearest object info for predicate evaluation
            "object_x": nearest.x if nearest else self.paddle_x,
            "object_y": nearest.y if nearest else 1.0,
            "object_vel": nearest.vx if nearest else 0.0,
        }


class CatcherGame:
    """
    Catch falling objects with a paddle.

    - Objects spawn at the top with random x position and slight horizontal drift
    - Paddle moves left/right at bottom
    - Catch = object reaches bottom within paddle width
    - Miss = object reaches bottom outside paddle
    - Win at 20 catches, lose at 5 misses

    Actions: MOVE_LEFT, MOVE_RIGHT, HOLD
    """

    MAX_TICKS = 600
    CATCHES_TO_WIN = 20
    MAX_MISSES = 5
    PADDLE_SPEED = 0.05
    PADDLE_HALF_WIDTH = 0.08
    FALL_SPEED = -0.02        # how fast objects fall per tick
    SPAWN_INTERVAL = 15       # ticks between spawns
    MAX_DRIFT = 0.008         # max horizontal drift speed

    ACTIONS = ["MOVE_LEFT", "MOVE_RIGHT", "HOLD"]

    def __init__(self):
        self.state: Optional[CatcherState] = None
        self.reset()

    def reset(self) -> CatcherState:
        """Reset to initial state."""
        self.state = CatcherState(
            paddle_x=0.5,
            paddle_width=self.PADDLE_HALF_WIDTH,
            objects=[],
            catches=0,
            misses=0,
            ticks=0,
            done=False,
        )
        # Spawn first object immediately
        self._spawn_object()
        return self.state

    def _spawn_object(self):
        """Spawn a new falling object at the top."""
        obj = FallingObject(
            x=random.uniform(0.1, 0.9),
            y=1.0,
            vx=random.uniform(-self.MAX_DRIFT, self.MAX_DRIFT),
            vy=self.FALL_SPEED,
        )
        self.state.objects.append(obj)

    def step(self, action: str) -> Tuple[CatcherState, float, bool]:
        """
        Advance one tick.

        Args:
            action: MOVE_LEFT, MOVE_RIGHT, or HOLD

        Returns:
            (state, reward, done)
        """
        if self.state is None or self.state.done:
            return self.state, 0.0, True

        action = action.upper().replace("ACTION_", "")

        # Move paddle
        if action == "MOVE_LEFT" or action == "TILT_LEFT":
            self.state.paddle_x = max(self.PADDLE_HALF_WIDTH,
                                       self.state.paddle_x - self.PADDLE_SPEED)
        elif action == "MOVE_RIGHT" or action == "TILT_RIGHT":
            self.state.paddle_x = min(1.0 - self.PADDLE_HALF_WIDTH,
                                       self.state.paddle_x + self.PADDLE_SPEED)
        # HOLD: no movement

        reward = 0.0

        # Update objects
        remaining = []
        for obj in self.state.objects:
            obj.y += obj.vy
            obj.x += obj.vx
            obj.x = max(0.0, min(1.0, obj.x))  # clamp horizontal

            if obj.y <= 0.05:  # reached ground level
                # Check if caught
                if abs(obj.x - self.state.paddle_x) <= self.PADDLE_HALF_WIDTH:
                    self.state.catches += 1
                    reward += 1.0
                else:
                    self.state.misses += 1
                    reward -= 1.0
            else:
                remaining.append(obj)

        self.state.objects = remaining
        self.state.ticks += 1

        # Spawn new objects periodically
        if self.state.ticks % self.SPAWN_INTERVAL == 0:
            self._spawn_object()

        # Check win/lose
        if self.state.catches >= self.CATCHES_TO_WIN:
            self.state.done = True
            reward += 5.0
        elif self.state.misses >= self.MAX_MISSES:
            self.state.done = True
            reward -= 5.0
        elif self.state.ticks >= self.MAX_TICKS:
            self.state.done = True

        return self.state, reward, self.state.done

    def get_state_dict(self) -> Dict[str, Any]:
        if self.state is None:
            return {}
        return self.state.to_dict()

    def render_ascii(self) -> str:
        if self.state is None:
            return "No game in progress"

        width = 40
        height = 12

        # Empty grid
        grid = [[" "] * width for _ in range(height)]

        # Draw objects
        for obj in self.state.objects:
            col = int(obj.x * (width - 1))
            row = int((1.0 - obj.y) * (height - 2))
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 2, row))
            grid[row][col] = "*"

        # Draw paddle
        paddle_center = int(self.state.paddle_x * (width - 1))
        paddle_hw = int(self.PADDLE_HALF_WIDTH * (width - 1))
        for c in range(max(0, paddle_center - paddle_hw),
                       min(width, paddle_center + paddle_hw + 1)):
            grid[height - 2][c] = "="

        # Ground
        grid[height - 1] = ["_"] * width

        lines = ["  " + "".join(row) for row in grid]
        lines.append(f"  Catches: {self.state.catches}/{self.CATCHES_TO_WIN}  "
                     f"Misses: {self.state.misses}/{self.MAX_MISSES}  "
                     f"Tick: {self.state.ticks}")
        return "\n".join(lines)


# =========================================================================
# Lightweight simulation function
# =========================================================================

def sim_catcher(state: Dict, action: str) -> Tuple[Dict, bool]:
    """
    Simulate one tick of Catcher (lightweight).

    Compatible with sim_snake / sim_pong interface.
    """
    paddle_x = state.get("paddle_x", 0.5)
    catches = state.get("catches", 0)
    misses = state.get("misses", 0)
    object_x = state.get("object_x", 0.5)
    object_y = state.get("object_y", 1.0)
    object_vel = state.get("object_vel", 0.0)
    paddle_hw = 0.08

    action = action.upper().replace("ACTION_", "")
    speed = 0.05

    if action in ("MOVE_LEFT", "TILT_LEFT"):
        paddle_x = max(paddle_hw, paddle_x - speed)
    elif action in ("MOVE_RIGHT", "TILT_RIGHT"):
        paddle_x = min(1.0 - paddle_hw, paddle_x + speed)

    # Move object
    object_y -= 0.02
    object_x += object_vel
    object_x = max(0.0, min(1.0, object_x))

    terminal = False
    if object_y <= 0.05:
        if abs(object_x - paddle_x) <= paddle_hw:
            catches += 1
        else:
            misses += 1
        # Reset object
        object_x = random.uniform(0.1, 0.9)
        object_y = 1.0
        object_vel = random.uniform(-0.008, 0.008)

    if catches >= 20 or misses >= 5:
        terminal = True

    return {
        "type": "catcher",
        "paddle_x": paddle_x,
        "catches": catches,
        "misses": misses,
        "object_x": object_x,
        "object_y": object_y,
        "object_vel": object_vel,
        "score": catches,
    }, terminal


class CognitiveCatcherGame(CatcherGame):
    """
    Variant of Catcher with colored objects.
    - Green: Good (Catch -> +1)
    - Red: Bad (Catch -> -5)
    Goal: Selectively catch green and dodge red.
    """
    def _spawn_object(self):
        super()._spawn_object()
        # Mark the new object with a type
        obj = self.state.objects[-1]
        obj.type = "red" if random.random() < 0.4 else "green" 

    def step(self, action: str) -> Tuple[CatcherState, float, bool]:
        # We need to hook into the reward logic. 
        # Easier to copy-modify step or post-process?
        # Post-processing state changes is tricky because objects are removed.
        # Let's override step completely for clarity, or patch the logic.
        # Overriding step is safest to ensure correct reward logic.
        
        if self.state is None or self.state.done:
            return self.state, 0.0, True

        # Copy-paste core movement logic from base
        action = action.upper().replace("ACTION_", "")
        if action == "MOVE_LEFT" or action == "TILT_LEFT":
            self.state.paddle_x = max(self.PADDLE_HALF_WIDTH, self.state.paddle_x - self.PADDLE_SPEED)
        elif action == "MOVE_RIGHT" or action == "TILT_RIGHT":
            self.state.paddle_x = min(1.0 - self.PADDLE_HALF_WIDTH, self.state.paddle_x + self.PADDLE_SPEED)

        reward = 0.0
        remaining = []
        for obj in self.state.objects:
            obj.y += obj.vy
            obj.x += obj.vx
            obj.x = max(0.0, min(1.0, obj.x))

            if obj.y <= 0.05:
                # Ground level
                caught = abs(obj.x - self.state.paddle_x) <= self.PADDLE_HALF_WIDTH
                obj_type = getattr(obj, "type", "green")
                
                if caught:
                    if obj_type == "green":
                        self.state.catches += 1
                        reward += 1.0
                    else: # Red
                        self.state.catches -= 5 # Penalty for catching bomb
                        reward -= 5.0
                else:
                    if obj_type == "green":
                        self.state.misses += 1
                        reward -= 1.0
                    else: # Red
                        # Dodged a bomb! Good job.
                        reward += 1.0 
            else:
                remaining.append(obj)

        self.state.objects = remaining
        self.state.ticks += 1

        if self.state.ticks % self.SPAWN_INTERVAL == 0:
            self._spawn_object()

        # End conditions
        if self.state.score <= -20: # Mercy rule for bad agent
            self.state.done = True
        if self.state.catches >= 20: # Win (might need more catches due to penalty?)
             self.state.done = True
             reward += 5.0
        if self.state.ticks >= self.MAX_TICKS:
            self.state.done = True

        return self.state, reward, self.state.done

    def get_state_dict(self) -> Dict[str, Any]:
        d = super().get_state_dict()
        nearest = self.state.nearest_object
        if nearest:
            d["object_type"] = getattr(nearest, "type", "green")
        else:
            d["object_type"] = "none"
        return d
