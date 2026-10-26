# Required imports
from heuristics.heuristic_base import Heuristic
from task import Task # Assuming Task class is available

import collections # For BFS queue
# import logging # Optional: for debugging

# Setup logging if needed
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a predicate and its arguments.
    e.g., '(predicate arg1 arg2)' -> ('predicate', ['arg1', 'arg2'])
    """
    # Removes surrounding brackets and splits by space
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    arguments = parts[1:]
    return predicate, arguments

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    Estimates the cost to reach the goal state by summing up the estimated
    costs for each individual tile that needs to be painted according to the
    goal. The estimated cost for painting a single tile includes the paint
    action itself, the minimum movement cost for a robot to reach an adjacent
    tile, and a cost for acquiring the correct color if no robot currently
    has it.

    Assumptions:
    - The problem instance is solvable with the given domain definition.
    - Tiles needing painting are either initially clear or become clear (the
      domain only supports painting clear tiles). If a goal tile is painted
      with the wrong color, the heuristic returns infinity (dead end).
    - The tile graph defined by adjacency predicates is connected, or at least
      all tiles relevant to the problem (robot locations, goal tiles, and
      their neighbors) are within the same connected component.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes the tile adjacency graph
    from the static 'up', 'down', 'left', 'right' predicates found in the
    task's static information. It then runs a Breadth-First Search (BFS)
    starting from every tile to compute all-pairs shortest paths (minimum
    number of move actions between any two tiles). This distance information
    is stored in a dictionary `self.tile_distances` for quick lookup during
    heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state of the planning problem.
    2. Extract relevant information from the state: robot locations, robot
       colors, and the status (clear or painted with a specific color) of
       each tile.
    3. Identify the set of goal facts (tiles that need to be painted with
       specific colors) that are not satisfied in the current state.
    4. For each unsatisfied goal fact '(painted T C)':
        a. Check the current status of tile T.
        b. If tile T is currently painted with a color C' different from the
           goal color C (i.e., '(painted T C')' is in the state and C' != C),
           this state is considered a dead end as there's no unpaint action.
           Return `float('inf')`.
        c. If tile T is clear ('(clear T)' is in the state), it needs to be
           painted. Add 1 to the total heuristic value (representing the cost
           of the paint action). Note that color C is required for this tile.
           Calculate the minimum movement cost for *any* robot to reach *any*
           tile that is adjacent to T. This is done by finding the minimum
           precomputed distance from any robot's current location to any of
           T's neighbors. Add this minimum distance to the total heuristic value.
           If no robot can reach any adjacent tile of T, the state is likely
           unsolvable, and the heuristic returns `float('inf')`.
    5. After iterating through all unsatisfied goal tiles, identify the set
       of unique colors that are needed for painting at least one tile.
    6. For each needed color C, check if any robot currently possesses color C.
       If no robot has color C, add 1 to the total heuristic value. This
       represents the cost of a `change_color` action to acquire the color
       for the first time for one of the robots.
    7. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        # Build tile adjacency graph from static facts
        self.tile_neighbors = collections.defaultdict(list)
        self.all_tiles = set()
        for fact_string in self.static:
            pred, args = parse_fact(fact_string)
            if pred in {'up', 'down', 'left', 'right'}:
                t1, t2 = args
                self.tile_neighbors[t1].append(t2)
                self.tile_neighbors[t2].append(t1) # Adjacency is symmetric
                self.all_tiles.add(t1)
                self.all_tiles.add(t2)

        # Compute all-pairs shortest paths using BFS
        self.tile_distances = {}
        for start_tile in self.all_tiles:
            self.tile_distances[(start_tile, start_tile)] = 0
            queue = collections.deque([(start_tile, 0)])
            visited = {start_tile}

            while queue:
                current_tile, dist = queue.popleft()

                for neighbor in self.tile_neighbors.get(current_tile, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.tile_distances[(start_tile, neighbor)] = dist + 1
                        # Distance is symmetric, store both ways
                        self.tile_distances[(neighbor, start_tile)] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        state = node.state
        h_value = 0
        colors_needed = set()

        # Extract robot locations and colors from the current state
        robot_locs = {}
        robot_colors = {}
        tile_statuses = {} # Store clear/painted status for quick lookup

        for fact_string in state:
            pred, args = parse_fact(fact_string)
            if pred == 'robot-at':
                robot, tile = args
                robot_locs[robot] = tile
            elif pred == 'robot-has':
                robot, color = args
                robot_colors[robot] = color
            elif pred == 'clear':
                tile = args[0]
                tile_statuses[tile] = 'clear'
            elif pred == 'painted':
                tile, color = args
                tile_statuses[tile] = ('painted', color)

        # Identify unsatisfied goal facts and check for dead ends
        unsatisfied_goals = []
        for goal_fact_string in self.goals:
            # Check if the goal fact is NOT in the current state
            if goal_fact_string not in state:
                # The goal fact must be of the form (painted T C)
                _, goal_args = parse_fact(goal_fact_string)
                goal_tile, goal_color = goal_args

                current_status = tile_statuses.get(goal_tile)

                # Check if it's painted with the wrong color
                if current_status is not None and current_status != 'clear':
                     painted_color = current_status[1]
                     if painted_color != goal_color:
                         # Tile is painted with the wrong color - dead end
                         # logger.debug(f"Dead end: Tile {goal_tile} painted {painted_color}, needs {goal_color}")
                         return float('inf')
                     # If current_status is ('painted', goal_color), the goal is satisfied,
                     # which contradicts the outer 'if goal_fact_string not in state'.
                     # So, we only need to handle the wrong color case here.

                # If we reach here, the goal is unsatisfied and the tile is either clear
                # or its status is not explicitly in the state (assume clear for goal tiles).
                # If current_status is None or current_status == 'clear', it needs painting.
                if current_status is None or current_status == 'clear':
                     unsatisfied_goals.append((goal_tile, goal_color))


        # If there are no unsatisfied goals, the state is a goal state.
        # The heuristic should be 0. This is handled by the initial h_value = 0
        # and the loops not running if unsatisfied_goals is empty.

        # Calculate heuristic components for each unsatisfied goal tile
        for tile_T, color_C in unsatisfied_goals:
            # Cost for paint action
            h_value += 1
            colors_needed.add(color_C)

            # Cost for movement: min distance from any robot to any adjacent tile of T
            min_dist_robot_to_adj_T = float('inf')
            adj_tiles_T = self.tile_neighbors.get(tile_T, [])

            # If a goal tile has no neighbors in the graph, it's unreachable for painting.
            # This indicates an unsolvable problem instance.
            if not adj_tiles_T:
                 # logger.debug(f"Dead end: Goal tile {tile_T} has no neighbors.")
                 return float('inf')

            # If there are no robots, the goal is unreachable
            if not robot_locs:
                 # logger.debug("Dead end: No robots available to paint.")
                 return float('inf')

            for robot, loc_R in robot_locs.items():
                for adj_T in adj_tiles_T:
                    # Ensure both robot location and adjacent tile are in the distance map.
                    # If not, they are in different connected components, meaning unreachable.
                    if (loc_R, adj_T) in self.tile_distances:
                         min_dist_robot_to_adj_T = min(min_dist_robot_to_adj_T, self.tile_distances[(loc_R, adj_T)])
                    # else: loc_R and adj_T are in different components, distance is effectively infinity.

            # If after checking all robots and all adjacent tiles, the minimum distance
            # is still infinity, it means no robot can reach any tile adjacent to tile_T.
            # If tile_T needs painting, this state is unsolvable.
            if min_dist_robot_to_adj_T == float('inf'):
                 # logger.debug(f"Dead end: No robot can reach adjacent tile for {tile_T}.")
                 return float('inf')

            h_value += min_dist_robot_to_adj_T


        # Cost for color changes
        for color_C in colors_needed:
            has_color = False
            for robot, color in robot_colors.items():
                if color == color_C:
                    has_color = True
                    break
            if not has_color:
                h_value += 1 # Cost for one change_color action

        return h_value
