from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to iterate up to the length of the shorter sequence.
    # A fact matches the pattern if all corresponding parts match.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    by summing the estimated cost for each unpainted goal tile. The cost for a
    single tile is estimated as the minimum cost for any robot to reach the tile,
    acquire the correct color, and paint it.

    # Assumptions
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates is static.
    - 'available-color' facts are static and list all colors that robots can hold.
    - Tiles that need painting are initially 'clear' or already have the correct goal color.
      (Assumes no tiles are painted the wrong color relative to the goal).
    - Robots always hold one color.
    - The cost of any move, change-color, or paint action is 1.

    # Heuristic Initialization
    - Extracts all tile objects mentioned in the problem.
    - Builds the tile adjacency graph based on 'up', 'down', 'left', 'right' static facts.
    - Computes all-pairs shortest paths between tiles using BFS.
    - Extracts the target color for each goal tile from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal conditions of the form `(painted <tile> <color>)`.
    2. Filter these goals to find the set of tiles that are *not* painted with the target color in the current state. These are the unsatisfied goal tiles.
    3. For each unsatisfied goal tile `T` with target color `C`:
        a. Find the current location and held color for each robot `R` in the state.
        b. For each robot `R`:
            i. Get the shortest distance from the robot's current location to tile `T` using the precomputed distances.
            ii. Calculate the cost to acquire the correct color `C`: 0 if the robot already has `C`, 1 otherwise (for a `change-color` action).
            iii. The estimated cost for robot `R` to paint tile `T` is `distance(R_location, T) + color_acquisition_cost + 1` (where +1 is for the `paint` action itself).
        c. The minimum cost for tile `T` is the minimum of these estimated costs over all robots.
    4. The total heuristic value is the sum of the minimum costs calculated for each unsatisfied goal tile.
    5. If all goal tiles are painted correctly, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static
        all_facts = task.facts # Includes static and non-static facts

        # 1. Extract all tile objects
        # Collect all objects that look like tiles (start with 'tile_')
        self.all_tiles = set()
        for fact in all_facts:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith('tile_'):
                     self.all_tiles.add(part)

        # 2. Build the tile adjacency graph
        self.tile_graph = {tile: set() for tile in self.all_tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                t1, t2 = parts[1], parts[2]
                # Only add edges if both tiles are in our collected list
                if t1 in self.tile_graph and t2 in self.tile_graph:
                    self.tile_graph[t1].add(t2)
                    self.tile_graph[t2].add(t1) # Graph is undirected

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_tile in self.all_tiles:
            self.distances[start_tile] = self._bfs(start_tile)

        # 4. Extract goal tiles and their target colors
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted" and len(args) == 2:
                tile, color = args
                self.goal_tiles[tile] = color

    def _bfs(self, start_tile):
        """
        Perform BFS from a start tile to find distances to all other tiles.
        Returns a dictionary {tile: distance}.
        """
        distances = {tile: float('inf') for tile in self.all_tiles}
        if start_tile not in self.all_tiles:
             # Should not happen if all_tiles is correctly populated
             return distances

        distances[start_tile] = 0
        queue = deque([start_tile])
        visited = {start_tile}

        while queue:
            current_tile = queue.popleft()
            current_dist = distances[current_tile]

            # Ensure current_tile is in the graph (might be a tile with no connections)
            if current_tile not in self.tile_graph:
                 continue

            for neighbor in self.tile_graph.get(current_tile, []): # Use .get for safety
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current robot locations and colors
        robot_info = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at" and len(parts) == 3:
                robot, location = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['location'] = location
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['color'] = color

        total_cost = 0

        # 2. Iterate through goal tiles and calculate cost for unsatisfied ones
        for goal_tile, target_color in self.goal_tiles.items():
            # Check if the goal for this tile is already satisfied
            if f"(painted {goal_tile} {target_color})" in state:
                continue # Goal already met for this tile

            # Goal is not met, calculate minimum cost to paint this tile
            min_tile_cost = float('inf')
            found_reachable_robot = False # Track if at least one robot can reach this tile

            for robot, info in robot_info.items():
                robot_loc = info.get('location')
                robot_color = info.get('color')

                if robot_loc is None or robot_color is None:
                    # Robot info is incomplete, skip this robot for now
                    continue

                # Ensure robot location and goal tile are in our distance map
                # and the goal tile is reachable from the robot's location
                if robot_loc in self.distances and goal_tile in self.distances[robot_loc]:
                    dist = self.distances[robot_loc][goal_tile]

                    # If tile is unreachable by this robot, skip it
                    if dist == float('inf'):
                        continue

                    found_reachable_robot = True

                    # Cost to change color if needed
                    color_cost = 1 if robot_color != target_color else 0

                    # Total cost for this robot to paint this tile:
                    # move to tile + change color (if needed) + paint action
                    current_robot_tile_cost = dist + color_cost + 1

                    min_tile_cost = min(min_tile_cost, current_robot_tile_cost)

            # If after checking all robots, no robot can reach this tile,
            # the state is likely unsolvable or requires actions not modeled
            # by this heuristic's assumptions (e.g., clearing a wrong color).
            # Return infinity to indicate a bad state.
            if not found_reachable_robot:
                 return float('inf')

            total_cost += min_tile_cost

        return total_cost
