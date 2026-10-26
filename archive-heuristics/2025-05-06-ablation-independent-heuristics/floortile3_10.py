from collections import deque
from heuristics.heuristic_base import Heuristic
from task import Task # Task is used in the constructor signature

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a (predicate, [args]) tuple."""
    # Remove leading/trailing brackets and split by spaces
    parts = fact_string.strip("()").split()
    if not parts:
        return None, [] # Handle empty string case, though unlikely for facts
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing, for each
    unsatisfied goal tile, the minimum cost required for any robot to paint
    that tile with the correct color. The cost for a single tile/robot
    combination includes the cost to change color (if needed), the shortest
    distance for the robot to move to a tile adjacent to the target tile,
    and the cost of the paint action. It also adds a cost if a robot is
    currently occupying the goal tile, requiring it to move off first.

    Assumptions:
    - The tile grid structure is defined by the static 'up', 'down', 'left',
      'right' predicates and is precomputed in the constructor.
    - Movement between adjacent tiles is assumed to be bidirectional with cost 1.
    - The heuristic ignores the dynamic 'clear' predicate for calculating
      movement distances between arbitrary tiles, assuming paths eventually
      become clear.
    - It assumes that goal tiles are initially either clear, painted correctly,
      painted incorrectly, or occupied by a robot.
    - If a goal tile is painted with the wrong color, the state is considered
      a dead end (heuristic returns infinity).
    - If a goal tile has no adjacent tiles defined in the static facts, it's
      considered impossible to paint (heuristic returns infinity).
    - The heuristic is not admissible; it is designed for greedy best-first search.

    Heuristic Initialization:
    In the constructor (__init__), the heuristic performs the following steps:
    1. Parses static facts to build an adjacency list representation of the
       tile grid graph based on 'up', 'down', 'left', and 'right' predicates.
       Assumes bidirectional connections.
    2. Collects all unique tile names found in the adjacency facts.
    3. Computes all-pairs shortest paths between all tiles in the graph using
       Breadth-First Search (BFS), storing the distances. This precomputation
       allows for quick lookups of movement costs during heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic:
    In the __call__ method for a given state:
    1. Parse the current state to identify:
       - Robot locations ('robot-at' facts).
       - Robot held colors ('robot-has' facts).
       - Painted tiles and their colors ('painted' facts).
       - Clear tiles ('clear' facts). (Note: clear_tiles set is not strictly used in the final logic but parsing is done).
    2. Initialize the total heuristic value `h` to 0.
    3. Iterate through each goal fact defined in the task.
    4. For each goal fact `(painted tile_i color_i)`:
       a. Check if the goal is already satisfied in the current state. If yes,
          continue to the next goal.
       b. If the goal is not satisfied, check if `tile_i` is painted with a
          different color (`(painted tile_i color')` where `color' != color_i`).
          If yes, the state is a dead end, return `float('inf')`.
       c. Determine if any robot is currently located *on* `tile_i`.
       d. Calculate the minimum cost for *any* robot to paint `tile_i` with
          `color_i` *assuming the robot starts from an adjacent tile*. This
          minimum cost (`min_cost_from_adjacent`) is found by considering each
          robot:
          i. Cost to change color: 1 if the robot's current color is not `color_i`,
             0 otherwise.
          ii. Cost to move: The shortest distance from the robot's current
              location to *any* tile adjacent to `tile_i`. This uses the
              precomputed distances. If no adjacent tile is reachable, the
              movement cost for this robot is infinity.
          iii. Cost to paint: 1 action.
          iv. The total cost for a robot `r` to paint `tile_i` from adjacent
              is (move cost) + (color change cost) + 1.
          v. `min_cost_from_adjacent` is the minimum of this value over all robots.
       e. If `min_cost_from_adjacent` is infinity (no robot can reach an adjacent
          tile), the goal is impossible, return `float('inf')`.
       f. Add the cost for this specific goal tile to the total heuristic `h`:
          - If a robot was found to be on `tile_i` (step 4c), add 1 (for moving
            the robot off the tile) + `min_cost_from_adjacent`.
          - Otherwise (if `tile_i` is not occupied by a robot), add just
            `min_cost_from_adjacent`. (Assumes if not occupied and not painted
            wrong, it is paintable from adjacent, respecting the paint precondition
            in a relaxed way).
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task: Task):
        """
        Initializes the heuristic by building the tile graph and computing
        all-pairs shortest paths.
        """
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        self.adj = {}
        self.tiles = set()

        # Build tile graph from static adjacency facts
        for fact_str in self.static:
            pred, args = parse_fact(fact_str)
            if pred in ['up', 'down', 'left', 'right']:
                if len(args) == 2:
                    tile1, tile2 = args
                    self.adj.setdefault(tile1, []).append(tile2)
                    self.adj.setdefault(tile2, []).append(tile1) # Assuming bidirectional
                    self.tiles.add(tile1)
                    self.tiles.add(tile2)
                # else: ignore malformed adjacency fact

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_tile in self.tiles:
            self.distances[start_tile] = {}
            queue = deque([(start_tile, 0)])
            visited = {start_tile}
            self.distances[start_tile][start_tile] = 0

            while queue:
                current_tile, dist = queue.popleft()
                for neighbor in self.adj.get(current_tile, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_tile][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        """
        state = node.state
        goals = self.goals

        # Parse current state facts for quick lookup
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {} # {robot_name: color_name}
        painted_tiles = {} # {tile_name: color_name}
        # clear_tiles = set() # Not strictly needed with the current logic, but parsing is done
        state_facts_set = set(state) # Convert to set for faster lookups

        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'robot-at' and len(args) == 2:
                robot, loc = args
                robot_locations[robot] = loc
            elif pred == 'robot-has' and len(args) == 2:
                robot, color = args
                robot_colors[robot] = color
            elif pred == 'painted' and len(args) == 2:
                tile, color = args
                painted_tiles[tile] = color
            # elif pred == 'clear' and len(args) == 1:
            #     tile = args[0]
            #     clear_tiles.add(tile)
            # else: ignore other facts or malformed facts

        h = 0

        # Calculate cost for each goal
        for goal_fact_str in goals:
            # Check if goal is satisfied
            if goal_fact_str in state_facts_set:
                continue # Goal already reached

            # Goal is unsatisfied. Parse it.
            pred, args = parse_fact(goal_fact_str)
            if pred != 'painted' or len(args) != 2:
                 # Ignore malformed goal facts
                 continue

            tile_i, color_i = args

            # Check for dead ends: goal tile painted with the wrong color
            if tile_i in painted_tiles and painted_tiles[tile_i] != color_i:
                return float('inf')

            # Check if a robot is currently on the goal tile
            robot_on_tile_i = None
            for robot_name, loc in robot_locations.items():
                if loc == tile_i:
                    robot_on_tile_i = robot_name
                    break

            # Calculate minimum cost for any robot to paint this tile from an adjacent tile
            min_cost_from_adjacent = float('inf')
            adj_tiles_i = self.adj.get(tile_i, [])

            # If goal tile is not in the graph or has no adjacent tiles, it's unreachable
            if tile_i not in self.tiles or not adj_tiles_i:
                 return float('inf')

            # If there are no robots, this goal is impossible
            if not robot_locations:
                 return float('inf')

            for robot_name, r_loc in robot_locations.items():
                r_color = robot_colors.get(robot_name) # Get robot's current color

                # Cost to get the right color
                # Assumes robot always has *a* color if robot-has fact exists.
                # If robot-has fact is missing, get() returns None.
                # If None != color_i, cost is 1. This seems reasonable.
                color_change_cost = 1 if r_color != color_i else 0

                # Cost to move to an adjacent tile
                min_dist_to_adj = float('inf')
                # Ensure robot's location is in the graph before looking up distances
                if r_loc in self.distances:
                    for adj_tile in adj_tiles_i:
                        # Ensure adjacent tile is in the graph
                        if adj_tile in self.distances[r_loc]:
                            dist = self.distances[r_loc][adj_tile]
                            min_dist_to_adj = min(min_dist_to_adj, dist)

                # Total cost for this robot to paint this tile from adjacent
                if min_dist_to_adj != float('inf'):
                    cost_r_i = min_dist_to_adj + color_change_cost + 1 # +1 for paint action
                    min_cost_from_adjacent = min(min_cost_from_adjacent, cost_r_i)

            # If no robot can reach an adjacent tile, this goal is impossible
            if min_cost_from_adjacent == float('inf'):
                return float('inf')

            # Add cost for this goal tile to the total heuristic
            if robot_on_tile_i is not None:
                # Robot is on the tile, needs 1 move action to get off first
                h += 1 + min_cost_from_adjacent
            else:
                # Tile is not occupied by a robot.
                # The domain requires (clear ?y) for paint.
                # If the tile is not clear, and not occupied/painted wrong (handled above),
                # it cannot be painted. This state might be a dead end if the tile
                # can never become clear (e.g., permanently blocked).
                # However, based on typical floortile problems, tiles become clear
                # when robots move off them. The relaxation ignores complex interactions.
                # We assume if it's not occupied and not painted wrong, it's paintable
                # from adjacent (i.e., effectively clear for the purpose of this heuristic).
                h += min_cost_from_adjacent

        return h
