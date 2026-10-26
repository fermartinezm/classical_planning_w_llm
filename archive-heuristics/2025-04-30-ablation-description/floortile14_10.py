# The base class Heuristic is assumed to be available in this path.
# from heuristics.heuristic_base import Heuristic
# Assuming Heuristic is defined elsewhere and imported correctly by the planner.
# For standalone testing, you might need a mock Heuristic class.

# Define a placeholder/mock Heuristic class if the actual one isn't available in this script context
# In the final environment, the actual Heuristic class will be used.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a mock Heuristic class if the real one is not found
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found, using mock.")


from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the minimum number of actions required to paint all
    goal tiles with their target colors. It considers the cost of changing
    robot color, moving the robot to a position adjacent to the target tile,
    and the paint action itself. It accounts for multiple robots and selects
    the minimum cost among them for each unpainted goal tile.

    # Assumptions
    - All tiles are part of a grid structure defined by up/down/left/right relations.
    - Robots can only move to clear tiles.
    - Tiles painted with the wrong color cannot be cleared or repainted, making the state unsolvable.
    - All robots are initially placed on a tile and have a color.
    - All available colors are listed in the static facts (though not explicitly used by the heuristic beyond knowing robot colors).

    # Heuristic Initialization
    - Extract the goal conditions (target tile and color for painting).
    - Build the grid graph representing tile connectivity for movement.
    - Identify all tiles and robots from initial state and static facts.
    - Pre-calculate paint locations for each tile (tiles adjacent to it).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to get robot locations, robot colors, and tile statuses (painted or clear).
    2. Check if any goal tile is painted with a color different from its target color. If so, the state is unsolvable, return infinity.
    3. Identify all goal tiles that are not yet painted with the correct color.
    4. If all goal tiles are painted correctly, the heuristic is 0.
    5. For each unsatisfied goal tile (tile T, color C):
        a. Find the set of tiles where a robot can stand to paint T (adjacent tiles).
        b. Initialize the minimum cost for this goal tile across all robots to infinity.
        c. For each robot R:
            i. Get R's current location and color from the state.
            ii. Calculate the color change cost: 1 if R's current color is not C, 0 otherwise.
            iii. Calculate the movement cost: Use BFS on the grid graph (only traversing clear tiles) to find the shortest distance from R's current location to any tile in the set of paint locations for T.
            iv. If a path exists (move_cost is not infinity):
                Calculate the total cost for robot R to paint T: move_cost + color_change_cost + 1 (for the paint action).
                Update the minimum cost for goal tile T: min_robot_cost = min(min_robot_cost, robot_cost).
        d. If after checking all robots, min_robot_cost is still infinity, the state is unsolvable, return infinity.
        e. Add min_robot_cost to the total heuristic cost.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all robots

        # Store goal tiles and their target colors
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'painted':
                if len(parts) == 3:
                    self.goal_tiles[parts[1]] = parts[2]
                # else: malformed goal fact, ignore or handle error

        # Build grid graph (adjacency list) from static facts
        self.adj = {}
        self.all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right']:
                if len(parts) == 3:
                    t1, t2 = parts[1], parts[2]
                    self.adj.setdefault(t1, []).append(t2)
                    self.adj.setdefault(t2, []).append(t1) # Grid is bidirectional for movement
                    self.all_tiles.add(t1)
                    self.all_tiles.add(t2)
                # else: malformed adjacency fact, ignore or handle error

        # Infer all robots from initial state facts
        self.all_robots = set()
        for fact in initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'robot-at':
                 if len(parts) == 3:
                    self.all_robots.add(parts[1])
                 # else: malformed robot-at fact, ignore or handle error

        # Pre-calculate paint locations for each tile (tiles adjacent to it)
        # A robot at tile X can paint tile Y if (up Y X) or (down Y X) or (left Y X) or (right Y X)
        # This means Y is adjacent to X. So paint locations for Y are just its neighbors.
        self.paint_locations = {}
        for tile in self.all_tiles:
             self.paint_locations[tile] = set(self.adj.get(tile, []))


    def bfs_distance(self, start_loc, target_locations, current_clear_tiles):
        """
        Finds the shortest distance from start_loc to any tile in target_locations
        using BFS, only traversing through tiles present in current_clear_tiles.
        Returns float('inf') if no path exists.
        """
        if start_loc in target_locations:
            return 0

        # Ensure start_loc is a valid tile in the grid
        if start_loc not in self.all_tiles:
             return float('inf') # Robot is somewhere unexpected?

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft() # Use popleft for BFS

            if current_loc in self.adj: # Check if current_loc has neighbors in the grid
                for neighbor in self.adj[current_loc]:
                    # Robot can move to neighbor if it's clear and not visited
                    # Note: The robot can move *from* a tile *to* a clear tile.
                    # The destination tile must be clear for the move action.
                    if neighbor in current_clear_tiles and neighbor not in visited:
                        if neighbor in target_locations:
                            return dist + 1 # Found shortest path to a target location

                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return float('inf') # Target locations not reachable

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract relevant information from the current state
        current_robot_locations = {} # robot -> tile
        current_robot_colors = {} # robot -> color
        current_painted_tiles = {} # tile -> color
        current_clear_tiles = set() # tile

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'robot-at' and len(parts) == 3:
                current_robot_locations[parts[1]] = parts[2]
            elif predicate == 'robot-has' and len(parts) == 3:
                current_robot_colors[parts[1]] = parts[2]
            elif predicate == 'painted' and len(parts) == 3:
                current_painted_tiles[parts[1]] = parts[2]
            elif predicate == 'clear' and len(parts) == 2:
                current_clear_tiles.add(parts[1])

        # Check for unsolvable state (goal tile painted with wrong color)
        for tile, goal_color in self.goal_tiles.items():
            if tile in current_painted_tiles and current_painted_tiles[tile] != goal_color:
                # This tile is painted with the wrong color and cannot be cleared. Unsolvable.
                return float('inf')

        total_cost = 0
        unsatisfied_goals = []

        # Identify unsatisfied goals that are potentially solvable
        for tile, goal_color in self.goal_tiles.items():
            is_satisfied = (tile in current_painted_tiles and current_painted_tiles.get(tile) == goal_color)
            if not is_satisfied:
                 # If not satisfied and not wrongly painted (checked above), it must be clear
                 # or not painted at all, which is fine.
                 unsatisfied_goals.append((tile, goal_color))

        # If all goals are satisfied, heuristic is 0.
        if not unsatisfied_goals:
            return 0

        # For each unsatisfied goal, find the minimum cost for any robot to paint it.
        for goal_tile, goal_color in unsatisfied_goals:
            min_robot_cost_for_this_goal = float('inf')

            # Tiles where a robot can stand to paint goal_tile
            target_paint_locations = self.paint_locations.get(goal_tile, set())

            # If the goal tile has no adjacent tiles in the grid, it's unreachable
            if not target_paint_locations:
                 # This shouldn't happen in a valid grid problem with paint goals
                 return float('inf')

            for robot in self.all_robots:
                if robot not in current_robot_locations:
                     # Robot not found in state? Should not happen if all robots are always 'at' some tile.
                     continue # Skip this robot

                robot_loc = current_robot_locations[robot]
                robot_color = current_robot_colors.get(robot) # Get color, might be None if robot-has fact is missing

                # Cost to get the right color
                color_cost = 0
                # Assume robot needs to change color if it has a color different from the goal color
                # and that color is not the goal color. If it has no color, it might need one.
                # The domain says robot-has ?r ?c, implying they always have one.
                if robot_color != goal_color:
                    color_cost = 1 # Assume one change_color action is enough

                # Cost to move from robot_loc to any tile in target_paint_locations
                # traversing only through clear tiles.
                move_cost = self.bfs_distance(robot_loc, target_paint_locations, current_clear_tiles)

                # If a path exists to a paint location
                if move_cost != float('inf'):
                     # Total cost for this robot for this specific goal
                     # Cost = (move to paint location) + (change color if needed) + (paint action)
                     robot_cost = move_cost + color_cost + 1
                     min_robot_cost_for_this_goal = min(min_robot_cost_for_this_goal, robot_cost)

            # If no robot can reach a paint location for this goal, it's unsolvable.
            if min_robot_cost_for_this_goal == float('inf'):
                 return float('inf')

            total_cost += min_robot_cost_for_this_goal

        return total_cost
