from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available at this path
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully, though PDDL facts are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the estimated cost for each unpainted goal tile
    independently. The cost for a single tile includes the paint action itself plus
    the minimum effort (movement and color change) for the closest robot to get
    into a position to paint that tile.

    # Assumptions
    - Solvable problems: Any goal tile that is not painted with the correct color
      is assumed to be currently `clear`. Tiles painted with the wrong color
      are not expected in solvable initial states or reachable states, as there
      is no action to unpaint.
    - Robots always hold a color (the `free-color` predicate is not used by actions
      to represent a state where a robot holds no color).
    - Available colors include all colors required by the goal state.
    - The grid connectivity defined by `up`, `down`, `left`, `right` predicates
      is static and forms an undirected graph for movement.
    - Movement cost is calculated based on the static grid distance, ignoring
      dynamic obstacles (tiles that are not `clear`). This is a relaxation.

    # Heuristic Initialization
    - Builds an undirected graph representing the grid connectivity based on
      `up`, `down`, `left`, and `right` static facts.
    - Computes all-pairs shortest paths (grid distances) on this graph using BFS.
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which colors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles and their required colors from the precomputed goals.
    2. Identify the current location and color of each robot from the current state.
    3. Initialize the total heuristic value `h` to 0.
    4. Iterate through each goal tile `T` and its required color `C`:
       a. Check if the fact `(painted T C)` is present in the current state.
       b. If `(painted T C)` is NOT in the state (meaning the tile needs painting):
          i. This tile requires at least one paint action (cost 1).
          ii. To perform the paint action, a robot must be at an adjacent tile and hold color `C`.
          iii. Calculate the minimum "preparation cost" for *any* robot to be ready to paint tile `T`. This involves finding the minimum over all robots `R`:
              - The cost for robot `R` is the sum of:
                  - The grid distance from `R`'s current location to the closest tile adjacent to `T`.
                  - 1 if `R` currently holds a color different from `C` (cost of `change_color`), 0 otherwise.
              - If no adjacent tile is reachable from the robot's current location on the grid, this robot cannot paint this tile via a path on the grid.
          iv. The minimum preparation cost for tile `T` is the minimum of the costs calculated for each robot.
          v. If a minimum preparation cost is found (i.e., at least one robot can reach an adjacent tile on the grid), add `1` (for the paint action) plus this minimum preparation cost to the total heuristic `h`.
          vi. If no robot can reach any adjacent tile on the grid (based on static distances), the problem might be unsolvable from this state, or requires complex coordination not captured by this simple heuristic. Return infinity or a large value.
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure, computing distances,
        and identifying goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        all_facts = task.facts # All possible facts in the domain

        # 1. Build the grid graph from static facts
        self.grid_graph = {}
        self.all_tiles = set()

        # Extract all tile objects first for robustness
        for fact in all_facts:
             parts = get_parts(fact)
             # Assuming tile objects start with 'tile_'
             self.all_tiles.update(p for p in parts if p.startswith('tile_'))

        # Initialize graph with all tiles
        for tile in self.all_tiles:
             self.grid_graph.setdefault(tile, set())

        # Add edges based on connectivity facts
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                # up ?x ?y means ?x is up from ?y. So ?x and ?y are adjacent.
                # The predicates define adjacency for movement.
                tile1 = parts[1]
                tile2 = parts[2]
                # Ensure tiles are valid objects before adding edges
                if tile1 in self.all_tiles and tile2 in self.all_tiles:
                    self.grid_graph[tile1].add(tile2)
                    self.grid_graph[tile2].add(tile1) # Grid is undirected for movement


        # 2. Compute all-pairs shortest paths (grid distances)
        self.distances = {}
        for start_tile in self.all_tiles:
            q = deque([(start_tile, 0)])
            visited = {start_tile}
            self.distances[(start_tile, start_tile)] = 0 # Distance to self is 0

            while q:
                current_tile, dist = q.popleft()

                for neighbor in self.grid_graph.get(current_tile, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_tile, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))


        # 3. Extract goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted" and len(args) == 2:
                tile, color = args
                self.goal_tiles[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # 1. Identify current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        robots = set() # Keep track of robot names

        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
                robots.add(robot)
            elif len(parts) == 3 and parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                robots.add(robot)

        # Handle case with no robots (unsolvable if goal requires painting)
        if not robots and self.goal_tiles:
             return float('inf') # Cannot paint without robots

        # 2. Identify painted tiles in the current state
        painted_in_state = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "painted":
                tile, color = parts[1], parts[2]
                painted_in_state[tile] = color

        # 3. Calculate heuristic based on unpainted goal tiles
        total_cost = 0

        for goal_tile, required_color in self.goal_tiles.items():
            # Check if the tile is already painted correctly
            if painted_in_state.get(goal_tile) == required_color:
                continue # This goal is met for this tile

            # This tile needs painting. Cost is 1 (paint action) + robot prep cost.
            paint_action_cost = 1

            # Find minimum preparation cost for any robot
            min_prep_cost_for_tile = float('inf')

            # Get tiles adjacent to the goal tile
            adjacent_tiles = self.grid_graph.get(goal_tile, [])

            # If the goal tile has no adjacent tiles (isolated), it cannot be painted.
            if not adjacent_tiles:
                 # This indicates a problem definition issue if the tile is in the goal
                 # but cannot be reached or painted.
                 return float('inf') # Unsolvable

            for robot in robots:
                robot_loc = robot_locations.get(robot)
                robot_color = robot_colors.get(robot)

                # If robot location or color is unknown (shouldn't happen in valid state)
                if robot_loc is None or robot_color is None:
                    continue # Skip this robot

                # Cost to change color if needed
                color_change_cost = 0 if robot_color == required_color else 1

                # Find minimum distance from robot's current location to any adjacent tile
                min_dist_from_robot_to_adj = float('inf')
                for adj_t in adjacent_tiles:
                    # Check if distance is precomputed (i.e., path exists on the grid)
                    if (robot_loc, adj_t) in self.distances:
                         dist = self.distances[(robot_loc, adj_t)]
                         min_dist_from_robot_to_adj = min(min_dist_from_robot_to_adj, dist)

                # If robot can reach at least one adjacent tile on the grid
                if min_dist_from_robot_to_adj != float('inf'):
                    prep_cost_R = min_dist_from_robot_to_adj + color_change_cost
                    min_prep_cost_for_tile = min(min_prep_cost_for_tile, prep_cost_R)

            # If no robot can reach any adjacent tile for this goal tile on the grid
            if min_prep_cost_for_tile == float('inf'):
                 # This goal tile is unreachable by any robot based on static grid.
                 return float('inf') # Unsolvable state

            # Add the cost for this unpainted tile to the total
            total_cost += paint_action_cost + min_prep_cost_for_tile

        return total_cost
