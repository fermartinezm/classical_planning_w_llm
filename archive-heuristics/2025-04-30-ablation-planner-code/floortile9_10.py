# Assuming Heuristic base class is available in a module named heuristics
# from heuristics.heuristic_base import Heuristic
# Note: The base class 'Heuristic' is expected to be provided by the environment.

from fnmatch import fnmatch
from collections import deque # Used for BFS

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a strict match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function for grid distance
def bfs_distance(start_node, target_nodes, adj):
    """
    Performs BFS to find the shortest distance from start_node to any node in target_nodes.
    Assumes unit cost edges.
    """
    # Ensure target_nodes is iterable and not empty
    if not target_nodes:
        return float('inf') # Cannot reach an empty set of targets

    # Check if start_node is already one of the target nodes
    if start_node in target_nodes:
        return 0

    queue = deque([(start_node, 0)]) # Use deque for efficient pop(0)
    visited = {start_node}

    while queue:
        curr_node, dist = queue.popleft() # Use popleft() for deque

        # Check neighbors
        for neighbor in adj.get(curr_node, []):
            if neighbor not in visited:
                if neighbor in target_nodes:
                    return dist + 1 # Found a target node
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # Target nodes not reachable

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total cost to paint all required tiles.
    It sums the estimated cost for each unpainted goal tile independently.
    The estimated cost for a single tile includes:
    1. The paint action itself (cost 1).
    2. The minimum movement cost for any robot to reach a tile adjacent to the target tile.
    3. An additional cost of 1 if the closest robot (in terms of movement) does not have the required color, estimating a color change action.

    # Assumptions
    - The problem instances are solvable (e.g., no tiles painted with the wrong color in the goal).
    - The grid structure is defined by `up`, `down`, `left`, `right` predicates, and movement between adjacent tiles costs 1.
    - Color change actions cost 1.
    - The heuristic calculates costs for each unpainted tile independently, potentially overestimating or underestimating due to resource sharing (robots, colors). It is not admissible but aims to be informative for greedy search.

    # Heuristic Initialization
    - Extracts the goal conditions (`painted` facts).
    - Builds an adjacency graph of tiles based on `up`, `down`, `left`, `right` static facts to compute movement distances.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts of the form `(painted tile color)`.
    2. Filter these goals to find `UnpaintedGoals`: the set of `(tile, color)` pairs where the tile needs to be painted with that color according to the goal, but is not currently painted with that color in the state. (Assumes tiles are either clear or painted correctly if they are goal tiles).
    3. If `UnpaintedGoals` is empty, the state is a goal state, and the heuristic is 0.
    4. Extract the current location of each robot (`robot-at`).
    5. Extract the current color held by each robot (`robot-has`).
    6. Initialize total heuristic cost to 0.
    7. For each `(target_tile, required_color)` in `UnpaintedGoals`:
        a. Initialize the minimum cost for this tile (`min_tile_cost`) to infinity.
        b. Find the set of tiles adjacent to `target_tile` using the pre-computed adjacency graph. These are the potential locations for a robot to paint `target_tile`.
        c. For each robot and its current location:
            i. Calculate the movement cost: the shortest distance from the robot's current location to any tile in the set of painting spots using BFS on the tile graph.
            ii. Calculate the color change cost: 1 if the robot's current color is not `required_color`, otherwise 0.
            iii. Calculate the total estimated cost for this robot to paint this tile: `movement_cost + color_change_cost + 1` (the +1 is for the paint action itself).
            iv. Update `min_tile_cost` with the minimum cost found so far across all robots for this specific `target_tile`.
        d. Add `min_tile_cost` to the total heuristic cost. If `min_tile_cost` is still infinity (e.g., tile unreachable), the state might be unsolvable, return infinity.
    8. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the tile graph.
        """
        # The base class 'Heuristic' is expected to be provided by the environment.
        # It should provide task.goals and task.static
        super().__init__(task)

        # Build adjacency graph for tiles
        self.adj = {}

        for fact in self.static:
            parts = get_parts(fact)
            # Check for adjacency predicates and correct number of arguments
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                # Predicate is direction, parts[1] is tile Y, parts[2] is tile X
                # This means Y is adjacent to X in that direction.
                # Movement is possible from X to Y and Y to X.
                t1, t2 = parts[1], parts[2]
                if t1 not in self.adj:
                    self.adj[t1] = []
                if t2 not in self.adj:
                    self.adj[t2] = []
                # Add bidirectional edges
                self.adj[t1].append(t2)
                self.adj[t2].append(t1)

        # Store goal painted facts
        self.goal_painted = {}
        # task.goals can be a single fact string or a list starting with 'and'
        if isinstance(self.goals, str): # Single fact goal
             predicate, *args = get_parts(self.goals)
             if predicate == "painted" and len(args) == 2:
                 tile, color = args
                 self.goal_painted[tile] = color
        elif isinstance(self.goals, list) and self.goals and self.goals[0] == 'and': # Conjunction goal
             # Iterate through the facts within the 'and' list
             for fact_str in self.goals[1:]:
                 predicate, *args = get_parts(fact_str)
                 if predicate == "painted" and len(args) == 2:
                     tile, color = args
                     self.goal_painted[tile] = color


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings).

        # 1. Identify Unpainted Goal Tiles
        unpainted_goals = {} # {tile: color}
        current_painted = {} # {tile: color}
        # current_clear = set() # Not strictly needed for this logic

        robot_locations = {} # {robot: tile}
        robot_colors = {} # {robot: color}

        # Efficiently extract relevant facts from the state frozenset
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "painted" and len(parts) == 3:
                current_painted[parts[1]] = parts[2]
            # elif predicate == "clear" and len(parts) == 2:
            #      current_clear.add(parts[1]) # Not used in current logic
            elif predicate == "robot-at" and len(parts) == 3:
                 robot_locations[parts[1]] = parts[2]
            elif predicate == "robot-has" and len(parts) == 3:
                 robot_colors[parts[1]] = parts[2]

        # A tile needs painting if it's a goal tile and not currently painted correctly
        for goal_tile, goal_color in self.goal_painted.items():
             if current_painted.get(goal_tile) != goal_color:
                 # Assuming solvable instances, if it's not painted correctly, it must be clear
                 unpainted_goals[goal_tile] = goal_color

        # 3. If UnpaintedGoals is empty, it's a goal state
        if not unpainted_goals:
            return 0

        # 6. Initialize total heuristic cost
        total_cost = 0

        # 7. For each unpainted goal tile
        for target_tile, required_color in unpainted_goals.items():
            min_tile_cost = float('inf')

            # 7b. Find tiles adjacent to the target tile
            # These are the tiles a robot must be AT to paint target_tile.
            painting_spots = set(self.adj.get(target_tile, []))

            if not painting_spots:
                 # This tile has no neighbors in the graph. Unreachable.
                 return float('inf')

            # 7c. For each robot
            for robot, robot_loc in robot_locations.items():
                # i. Calculate movement cost to reach a painting spot
                # Use the pre-computed adjacency graph
                move_cost = bfs_distance(robot_loc, painting_spots, self.adj)

                if move_cost == float('inf'):
                    # This robot cannot reach this tile's painting spots
                    continue # Try the next robot

                # ii. Calculate color change cost
                robot_current_color = robot_colors.get(robot)
                color_cost = 0
                if robot_current_color != required_color:
                    color_cost = 1 # Estimate one change_color action

                # iii. Total estimated cost for this robot to paint this tile
                current_tile_cost = move_cost + color_cost + 1 # +1 for the paint action

                # iv. Update minimum cost for this tile
                min_tile_cost = min(min_tile_cost, current_tile_cost)

            # 7d. Add minimum cost for this tile to total
            if min_tile_cost == float('inf'):
                 # No robot can reach this tile's painting spots
                 return float('inf') # Unsolvable state

            total_cost += min_tile_cost

        # 8. Return total heuristic cost
        return total_cost
