from heuristics.heuristic_base import Heuristic
from task import Task
from collections import defaultdict, deque
import math

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove parentheses and split by space
    # Use shlex.split if arguments could contain spaces or quotes,
    # but simple split is sufficient for this domain's fact format.
    parts = fact_string[1:-1].split()
    return tuple(parts)

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        up the estimated costs for each goal tile that is not yet painted
        with the correct color. The estimated cost for a single such tile
        includes the paint action cost, the minimum movement cost for a robot
        to reach an adjacent tile, and a penalty if the required color is not
        currently held by any robot. It returns infinity if any goal tile is
        painted with the wrong color, as this is an unsolvable state in this domain.

    Assumptions:
        - The domain represents a grid structure defined by the up, down, left,
          and right predicates.
        - All actions have a cost of 1.
        - There is no action to unpaint or clear a painted tile.
        - Tile names are consistent with the grid structure defined by adjacency
          predicates (e.g., tile_R_C). The heuristic uses the adjacency graph
          directly, not parsing coordinates from names.
        - The heuristic is non-admissible and designed for greedy best-first search.
        - The heuristic uses static grid distances for movement cost estimation,
          ignoring dynamic obstacles (like other robots) or the 'clear' status
          of intermediate tiles during pathfinding.

    Heuristic Initialization:
        - Parses the goal facts to identify which tiles need to be painted
          and with which color, storing them in `self.goal_painted_tiles`.
        - Parses the static facts (up, down, left, right) to build an
          adjacency list representation (`self.adj`) of the grid graph.
        - Identifies all unique tile objects in the domain based on static
          and goal facts, storing them in `self.tiles`.
        - Precomputes the shortest path distance between every pair of tiles
          in the grid graph using Breadth-First Search (BFS). This distance
          map (`self.dist_map`) is stored for efficient lookup during heuristic
          computation.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the heuristic value `h` to 0.
        2. Parse the current state (`node.state`) to determine robot locations
           (`robot_locations`), robot colors (`robot_colors`), and which tiles
           are painted (`painted_tiles`).
        3. Identify the set of goal tiles that are not currently painted with
           their required color (`unsatisfied_goal_tiles`). This is done by
           checking if the specific goal fact `(painted T C)` is present in the state.
        4. Iterate through the `unsatisfied_goal_tiles`:
           - For each `tile_to_paint` in this set:
             - Check if `tile_to_paint` is present in the `painted_tiles` map.
             - If it is, it means the tile is painted with *some* color, but not
               the correct goal color (otherwise it wouldn't be in `unsatisfied_goal_tiles`).
               This indicates a dead end. Set `h` to infinity and break the loop.
        5. If any wrongly painted goal tile was found (`h` is infinity), return `math.inf`.
        6. If there are no `unsatisfied_goal_tiles`, the goal is reached. Return 0.
        7. If there are `unsatisfied_goal_tiles` and no dead ends:
           - Initialize a set `needed_colors` to store the distinct colors required
             by the `unsatisfied_goal_tiles`.
           - For each `tile_to_paint` in `unsatisfied_goal_tiles`:
             - Get the required color `goal_color` for `tile_to_paint` from
               `self.goal_painted_tiles`.
             - Add `goal_color` to `needed_colors`.
             - Add 1 to `h` (representing the estimated cost of the paint action
               for this tile).
             - Calculate the minimum movement cost for *any* robot to reach *any*
               tile adjacent to `tile_to_paint`. This is done by iterating through
               all robots, finding their current location, iterating through all
               neighbors of `tile_to_paint` (using `self.adj`), and finding the
               minimum precomputed distance (`self.dist_map`) from the robot's
               location to any of these neighbors. Add this minimum movement cost to `h`.
               If a tile has no adjacent tiles or no robot can reach any adjacent tile,
               add infinity to `h`.
           - Identify the set of colors currently held by the robots (`colors_robots_have`).
           - Calculate the set of colors required by `unsatisfied_goal_tiles` that are not
             currently held by any robot (`colors_to_acquire = needed_colors - colors_robots_have`).
           - Add the size of `colors_to_acquire` to `h`. This estimates the cost
             of changing colors (simplified, assuming one change action is sufficient
             per color needed by some tile but not held by any robot).
        8. Return the final value of `h`.
    """

    def __init__(self, task: Task):
        """
        Initializes the heuristic by precomputing grid information.

        Args:
            task: The planning task object.
        """
        super().__init__()
        self.goals = task.goals # Store goal facts
        self.static = task.static # Store static facts

        self.goal_painted_tiles = {}
        self.adj = defaultdict(list)
        self.tiles = set()

        # Parse goal facts to find required painted tiles and colors
        for goal_fact_str in self.goals:
            # Goal facts are typically (painted tile color)
            if goal_fact_str.startswith('(painted'):
                _, tile, color = parse_fact(goal_fact_str)
                self.goal_painted_tiles[tile] = color
                self.tiles.add(tile) # Add goal tiles to the set of all tiles

        # Parse static facts to build adjacency list and collect all tiles
        for static_fact_str in self.static:
            parts = parse_fact(static_fact_str)
            predicate = parts[0]
            if predicate in ('up', 'down', 'left', 'right'):
                # These predicates define adjacency. (pred t1 t2) means t1 is pred of t2
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # So, tile_1_1 is adjacent to tile_0_1.
                t1, t2 = parts[1], parts[2]
                self.adj[t1].append(t2)
                self.adj[t2].append(t1) # Adjacency is symmetric
                self.tiles.add(t1)
                self.tiles.add(t2)
            # We could also extract tiles from other static facts if needed,
            # but adjacency facts cover the grid tiles in typical problems.

        # Precompute all-pairs shortest paths using BFS
        self.dist_map = {}
        for start_node in self.tiles:
            self.dist_map[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                current_node, distance = queue.popleft()
                self.dist_map[start_node][current_node] = distance

                # Get neighbors from the adjacency list
                for neighbor in self.adj.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))

    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.

        Args:
            node: The current state node.

        Returns:
            The heuristic value (estimated cost to reach the goal).
        """
        state = node.state

        robot_locations = {}
        robot_colors = {}
        painted_tiles = {}
        # clear_tiles = set() # Not strictly needed if we check goal fact presence

        # Parse current state facts
        for fact_str in state:
            parts = parse_fact(fact_str)
            predicate = parts[0]
            if predicate == 'robot-at':
                _, robot, tile = parts
                robot_locations[robot] = tile
            elif predicate == 'robot-has':
                _, robot, color = parts
                robot_colors[robot] = color
            elif predicate == 'painted':
                _, tile, color = parts
                painted_tiles[tile] = color
            # elif predicate == 'clear':
            #     _, tile = parts
            #     clear_tiles.add(tile)
            # Ignore other predicates like available-color, free-color for heuristic

        # Identify unsatisfied goal tiles: tiles that are goals but the goal fact is not in state
        unsatisfied_goal_tiles = set()
        for goal_tile, goal_color in self.goal_painted_tiles.items():
             # Check if the goal fact (painted goal_tile goal_color) is NOT in the state
             if f'(painted {goal_tile} {goal_color})' not in state:
                 unsatisfied_goal_tiles.add(goal_tile)

        # Check for dead ends (wrongly painted goal tiles)
        # An unsatisfied goal tile is a dead end if it is painted with *any* color
        for goal_tile in unsatisfied_goal_tiles:
             if goal_tile in painted_tiles:
                 # It's painted, but not with the goal color (since it's in unsatisfied_goal_tiles)
                 # This means it's painted with the wrong color.
                 return math.inf

        # If no unsatisfied goal tiles, the goal is reached
        if not unsatisfied_goal_tiles:
            return 0

        h = 0
        needed_colors = set()

        # Calculate cost for each unsatisfied goal tile
        for tile_to_paint in unsatisfied_goal_tiles:
            goal_color = self.goal_painted_tiles[tile_to_paint]
            needed_colors.add(goal_color)

            # Cost for paint action (1 action per tile)
            h += 1

            # Cost for movement: min distance from any robot to an adjacent tile
            min_move_cost_for_tile = math.inf
            
            # If there are no robots, movement is impossible
            if not robot_locations:
                 min_move_cost_for_tile = math.inf
            else:
                for robot, robot_loc in robot_locations.items():
                    min_dist_to_adj_tile = math.inf
                    # Find adjacent tiles of the tile_to_paint
                    adjacent_tiles = self.adj.get(tile_to_paint, [])

                    # If the tile has no adjacent tiles, it cannot be painted
                    if not adjacent_tiles:
                         min_move_cost_for_tile = math.inf # Cannot paint if no adjacent tile
                         break # Stop checking robots for this tile

                    for adj_tile in adjacent_tiles:
                        # Use precomputed distance from robot's location to the adjacent tile
                        # Ensure both robot_loc and adj_tile are in the dist_map (should be if tiles are connected)
                        if robot_loc in self.dist_map and adj_tile in self.dist_map[robot_loc]:
                             dist = self.dist_map[robot_loc][adj_tile]
                             min_dist_to_adj_tile = min(min_dist_to_adj_tile, dist)
                        # else: This adjacent tile is unreachable from robot_loc in the static grid

                    min_move_cost_for_tile = min(min_move_cost_for_tile, min_dist_to_adj_tile)

            # Add the calculated min movement cost. If it's still inf, the total h becomes inf.
            h += min_move_cost_for_tile

        # Cost for colors: number of needed colors not held by any robot
        colors_robots_have = set(robot_colors.values())
        colors_to_acquire = needed_colors - colors_robots_have
        h += len(colors_to_acquire)

        return h
