from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions outside the class for general utility

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_distances(start_node, graph):
    """Computes shortest path distances from start_node to all reachable nodes in the graph."""
    distances = {start_node: 0}
    queue = deque([start_node])
    while queue:
        curr = queue.popleft()
        dist = distances[curr]
        if curr in graph:
            for neighbor in graph[curr]:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append(neighbor)
    return distances


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles.
    It sums the minimum estimated cost for each unpainted goal tile independently,
    considering the closest robot, the movement needed to reach an adjacent tile,
    the paint action, and a potential color change.

    # Assumptions
    - The tile grid structure is defined by 'up', 'down', 'left', 'right' predicates.
    - Robots can move between adjacent clear tiles and change colors if available.
    - A tile must be 'clear' to be painted. Once painted, it is no longer 'clear'.
    - A tile painted with the wrong color cannot be repainted (problem is unsolvable).
    - The cost of each action (move, paint, change_color) is 1.
    - The heuristic sums costs per tile independently, ignoring potential synergies
      (e.g., one movement covering access to multiple tiles) or conflicts.

    # Heuristic Initialization
    - Extracts all tile objects from the initial state, goals, and static facts.
    - Builds a bidirectional adjacency graph of tiles based on 'up', 'down', 'left', 'right' static facts.
    - Stores the set of goal 'painted' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current state of each robot (location and color).
    2. Identify the set of goal 'painted' facts that are not true in the current state (unachieved goals).
    3. Check for unsolvable states: If any tile is currently painted with a color different from its goal color, the problem is unsolvable. Return infinity.
    4. If there are no unachieved goals, the heuristic is 0.
    5. Compute shortest path distances from each robot's current location to all other tiles using BFS on the tile adjacency graph.
    6. Initialize the total heuristic cost to 0.
    7. For each unachieved goal fact `(painted T C_goal)`:
        a. Initialize the minimum cost to paint tile `T` with color `C_goal` by any robot to infinity.
        b. For each robot `R` with location `L_R` and color `C_R`:
            i. Find the minimum distance from `L_R` to any tile adjacent to `T` using the precomputed BFS distances. If `T` has no adjacent tiles or no adjacent tile is reachable by `R`, this robot cannot paint `T`.
            ii. If an adjacent tile is reachable, calculate the estimated cost for robot `R` to paint `T`:
                - Movement cost: minimum distance from `L_R` to an adjacent tile of `T`.
                - Paint action cost: 1.
                - Color change cost: 1 if `C_R` is not `C_goal`, otherwise 0.
                Total cost for robot R = Movement cost + Paint action cost + Color change cost.
            iii. Update the minimum cost for tile `T` with the calculated cost for robot `R` if it's lower.
        c. If the minimum cost for tile `T` is still infinity (meaning no robot can paint the tile), the problem is unsolvable. Return infinity.
        d. Add the minimum cost for tile `T` to the total heuristic cost.
    8. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the tile graph."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Extract all tile objects present in the problem definition
        all_tiles = set()
        for fact in initial_state | self.goals | static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            # Predicates where arguments are tiles
            if predicate in ["robot-at", "clear", "painted"]:
                if len(parts) > 1:
                    all_tiles.add(parts[1])
            elif predicate in ["up", "down", "left", "right"]:
                if len(parts) > 2:
                    all_tiles.add(parts[1])
                    all_tiles.add(parts[2])
        self.all_tiles = list(all_tiles) # Store as list

        # 2. Build bidirectional tile adjacency graph from static facts
        self.tile_adj = {tile: set() for tile in self.all_tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate in ["up", "down", "left", "right"] and len(parts) > 2:
                tile1, tile2 = parts[1], parts[2]
                # Add bidirectional edges
                if tile1 in self.tile_adj: self.tile_adj[tile1].add(tile2)
                if tile2 in self.tile_adj: self.tile_adj[tile2].add(tile1)

        # 3. Store goal painted facts
        self.goal_painted = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) > 2:
                self.goal_painted.add((parts[1], parts[2]))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current robot states (location and color)
        robot_states = {} # Map robot name to [location, color]
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "robot-at" and len(parts) > 2:
                robot, location = parts[1], parts[2]
                if robot not in robot_states:
                    robot_states[robot] = [None, None]
                robot_states[robot][0] = location
            elif predicate == "robot-has" and len(parts) > 2:
                robot, color = parts[1], parts[2]
                if robot not in robot_states:
                     robot_states[robot] = [None, None]
                robot_states[robot][1] = color

        # 2. Identify unachieved goal 'painted' facts
        unpainted_goals = set()
        for tile, color in self.goal_painted:
            if f"(painted {tile} {color})" not in state:
                unpainted_goals.add((tile, color))

        # 3. Check for unsolvable states (tile painted wrong color)
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == "painted" and len(parts) > 2:
                tile, current_color = parts[1], parts[2]
                # Check if this tile has a goal color different from the current one
                for goal_tile, goal_color in self.goal_painted:
                    if tile == goal_tile and current_color != goal_color:
                        # Tile is painted, but needs a different color. Unsolvable.
                        return float('inf')

        # 4. If no unachieved goals, heuristic is 0
        if not unpainted_goals:
            return 0

        # 5. Compute BFS distances from each robot's current location
        robot_distances = {}
        for robot, (location, color) in robot_states.items():
            if location: # Robot must have a location to compute distances
                 robot_distances[robot] = bfs_distances(location, self.tile_adj)

        # 6. Initialize total heuristic cost
        total_cost = 0

        # 7. For each unachieved goal fact (T, C_goal)
        for tile_T, color_C_goal in unpainted_goals:
            min_cost_for_tile = float('inf')

            # Find adjacent tiles for T
            adjacent_to_T = self.tile_adj.get(tile_T, set())
            if not adjacent_to_T:
                 # Tile T has no adjacent tiles defined, cannot be painted. Unsolvable.
                 return float('inf')

            # For each robot R
            for robot_R, (location_R, color_R) in robot_states.items():
                if location_R is None: continue # Robot must have a location

                # Find minimum distance from robot_R to any tile adjacent to T
                min_dist_to_adj = float('inf')
                if robot_R in robot_distances: # Check if BFS was successful for this robot
                    distances_from_R = robot_distances[robot_R]
                    for adj_tile in adjacent_to_T:
                        if adj_tile in distances_from_R:
                            min_dist_to_adj = min(min_dist_to_adj, distances_from_R[adj_tile])

                if min_dist_to_adj == float('inf'):
                    # Robot R cannot reach any tile adjacent to T
                    continue

                # Calculate estimated cost for robot R to paint T
                cost_R = min_dist_to_adj + 1 # Move to adjacent + Paint action (cost 1)
                if color_R != color_C_goal:
                    cost_R += 1 # Color change action (cost 1)

                # Update minimum cost for tile T
                min_cost_for_tile = min(min_cost_for_tile, cost_R)

            # If after checking all robots, no robot can paint the tile, it's unsolvable
            if min_cost_for_tile == float('inf'):
                 return float('inf')

            # Add minimum cost for tile T to total heuristic
            total_cost += min_cost_for_tile

        # 8. Return total heuristic cost
        return total_cost
