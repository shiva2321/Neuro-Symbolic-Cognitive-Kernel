"""
Standalone GridWorld Survival Demo

A simplified demonstration that shows the game working
without requiring all NSCK dependencies.
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from gridworld_survival import GridWorldSurvival


def simple_agent_policy(state_dict):
    """
    A simple rule-based policy for demonstration.
    
    Priority order:
    1. Avoid immediate danger (enemies very close)
    2. Satisfy critical needs (hunger/thirst)
    3. Seek nearby resources
    4. Move toward exit if healthy
    5. Explore randomly
    """
    predicates = set(state_dict.get('predicates', []))
    
    # Priority 1: Avoid danger
    if "ENEMY_VERY_CLOSE" in predicates:
        for pred in predicates:
            if pred.startswith("ENEMY_DIRECTION_"):
                parts = pred.split('_')
                dy, dx = int(parts[2]), int(parts[3])
                # Move opposite direction
                if dy < 0 and "WALL_DIRECTION_1_0" not in predicates:
                    return "ACTION_DOWN"
                elif dy > 0 and "WALL_DIRECTION_-1_0" not in predicates:
                    return "ACTION_UP"
                elif dx < 0 and "WALL_DIRECTION_0_1" not in predicates:
                    return "ACTION_RIGHT"
                elif dx > 0 and "WALL_DIRECTION_0_-1" not in predicates:
                    return "ACTION_LEFT"
    
    # Priority 2: Critical needs
    if "CRITICAL_HUNGER" in predicates or "CRITICAL_THIRST" in predicates:
        for pred in predicates:
            if pred.startswith("FOOD_DIRECTION_") or pred.startswith("WATER_DIRECTION_"):
                parts = pred.split('_')
                dy, dx = int(parts[2]), int(parts[3])
                if dy < 0:
                    return "ACTION_UP"
                elif dy > 0:
                    return "ACTION_DOWN"
                elif dx < 0:
                    return "ACTION_LEFT"
                elif dx > 0:
                    return "ACTION_RIGHT"
    
    # Priority 3: Seek nearby resources
    if "HUNGRY" in predicates:
        for pred in predicates:
            if pred.startswith("FOOD_DIRECTION_"):
                parts = pred.split('_')
                dy, dx = int(parts[2]), int(parts[3])
                if dy < 0:
                    return "ACTION_UP"
                elif dy > 0:
                    return "ACTION_DOWN"
                elif dx < 0:
                    return "ACTION_LEFT"
                elif dx > 0:
                    return "ACTION_RIGHT"
    
    if "THIRSTY" in predicates:
        for pred in predicates:
            if pred.startswith("WATER_DIRECTION_"):
                parts = pred.split('_')
                dy, dx = int(parts[2]), int(parts[3])
                if dy < 0:
                    return "ACTION_UP"
                elif dy > 0:
                    return "ACTION_DOWN"
                elif dx < 0:
                    return "ACTION_LEFT"
                elif dx > 0:
                    return "ACTION_RIGHT"
    
    # Priority 4: Move toward exit if resources are good
    stats = state_dict.get('stats', {})
    if (stats.get('energy', 0) > 0.5 and 
        stats.get('health', 0) > 0.7 and
        stats.get('hunger', 1) < 0.5 and
        stats.get('thirst', 1) < 0.5):
        
        if "EXIT_NEARBY" in predicates:
            player_y, player_x = state_dict['player_pos']
            exit_y, exit_x = state_dict['exit_pos']
            
            dy_to_exit = exit_y - player_y
            dx_to_exit = exit_x - player_x
            
            if abs(dy_to_exit) > abs(dx_to_exit):
                if dy_to_exit < 0:
                    return "ACTION_UP"
                else:
                    return "ACTION_DOWN"
            else:
                if dx_to_exit < 0:
                    return "ACTION_LEFT"
                else:
                    return "ACTION_RIGHT"
    
    # Priority 5: Random exploration
    return random.choice([
        "ACTION_UP", "ACTION_DOWN", "ACTION_LEFT", "ACTION_RIGHT"
    ])


def run_demo(num_episodes=5, render_episodes=True):
    """
    Run a demonstration of the GridWorld Survival game.
    """
    print("=" * 80)
    print("GRIDWORLD SURVIVAL - DEMONSTRATION")
    print("=" * 80)
    print()
    print("This demo shows a complex game environment with:")
    print("  - Multi-objective decision making (food, water, energy management)")
    print("  - Dynamic enemies with patrol patterns")
    print("  - Risk assessment (hazards, enemy proximity)")
    print("  - Strategic planning (when to prioritize survival vs progress)")
    print()
    print("=" * 80)
    print()
    
    game = GridWorldSurvival(
        width=15,
        height=15,
        num_enemies=2,
        num_food=6,
        num_water=5,
        num_hazards=4,
        num_treasures=2,
        max_steps=300
    )
    
    episode_results = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        state = game.reset()
        
        if render_episodes:
            print("\nInitial State:")
            print(game.render_ascii())
            print()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 300:
            state_dict = game.get_state_dict()
            
            # Simple policy decides action
            action = simple_agent_policy(state_dict)
            
            # Execute action
            state, reward, done, info = game.step(action)
            total_reward += reward
            steps += 1
            
            # Show progress at intervals
            if render_episodes and steps % 50 == 0:
                print(f"\nStep {steps}:")
                print(game.render_ascii())
                print()
        
        # Episode complete
        result = {
            'episode': episode + 1,
            'steps': steps,
            'score': state.stats.score,
            'reward': total_reward,
            'success': info.get('success', False),
            'reason': info.get('reason', 'unknown'),
            'final_stats': {
                'energy': state.stats.energy,
                'health': state.stats.health,
                'hunger': state.stats.hunger,
                'thirst': state.stats.thirst,
                'treasures': state.stats.treasures_collected,
            }
        }
        
        episode_results.append(result)
        
        print(f"\nEpisode {episode + 1} Complete!")
        print(f"  Final Score: {result['score']}")
        print(f"  Total Reward: {result['reward']:.2f}")
        print(f"  Steps: {result['steps']}")
        print(f"  Reason: {result['reason']}")
        print(f"  Success: {result['success']}")
        print(f"  Final Energy: {result['final_stats']['energy']:.2f}")
        print(f"  Final Health: {result['final_stats']['health']:.2f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    avg_score = sum(r['score'] for r in episode_results) / len(episode_results)
    avg_steps = sum(r['steps'] for r in episode_results) / len(episode_results)
    avg_reward = sum(r['reward'] for r in episode_results) / len(episode_results)
    success_rate = sum(1 for r in episode_results if r['success']) / len(episode_results)
    
    print(f"\nTotal Episodes: {num_episodes}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Success Rate: {success_rate * 100:.1f}%")
    
    print("\n" + "=" * 80)
    print()
    
    return episode_results


def demonstrate_game_complexity():
    """Show what makes this game more complex than Snake/Maze/Pong."""
    print("\n" + "=" * 80)
    print("WHAT MAKES THIS GAME COMPLEX?")
    print("=" * 80)
    print()
    
    print("1. MULTI-OBJECTIVE OPTIMIZATION:")
    print("   - Must balance multiple resources (energy, health, hunger, thirst)")
    print("   - Each resource has different drain rates and consequences")
    print("   - Trade-offs between short-term survival and long-term success")
    print()
    
    print("2. DYNAMIC ENVIRONMENT:")
    print("   - Enemies patrol with predictable patterns")
    print("   - Resources are consumable and don't respawn")
    print("   - Environment state changes continuously")
    print()
    
    print("3. RISK ASSESSMENT:")
    print("   - Must evaluate danger vs reward (treasure near enemy?)")
    print("   - Different threat levels (enemies move, hazards are static)")
    print("   - Health management affects decision making")
    print()
    
    print("4. STRATEGIC PLANNING:")
    print("   - Can't just rush to exit - need to survive the journey")
    print("   - Resource collection requires detours")
    print("   - Must anticipate future needs (not just react)")
    print()
    
    print("5. STATE SPACE COMPLEXITY:")
    print("   - 20x20 grid = 400 positions")
    print("   - 4 continuous stats (energy, health, hunger, thirst)")
    print("   - Multiple entities (3 enemies, 8 food, 6 water, 5 hazards, 3 treasures)")
    print("   - Enemy positions change = huge state space")
    print()
    
    print("Compared to:")
    print("  - Snake: Single objective (eat food, avoid walls/self)")
    print("  - Maze: Single objective (reach exit)")
    print("  - Pong: Single objective (hit ball with paddle)")
    print()
    
    print("GridWorld requires:")
    print("  ✓ Multi-objective decision making")
    print("  ✓ Long-term planning")
    print("  ✓ Risk-reward trade-offs")
    print("  ✓ Resource management")
    print("  ✓ Adaptation to dynamic threats")
    print()
    
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GridWorld Survival Demo")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--complexity", action="store_true", help="Show complexity explanation")
    
    args = parser.parse_args()
    
    if args.complexity:
        demonstrate_game_complexity()
    else:
        run_demo(
            num_episodes=args.episodes,
            render_episodes=not args.no_render
        )
