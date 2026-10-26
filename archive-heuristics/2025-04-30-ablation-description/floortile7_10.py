from fnmatch import fnmatch
from collections import deque
# Assuming the Heuristic base class is available at this path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_node, graph, all_nodes):
    """
    Performs Breadth-First Search starting from start_node on the given graph.
    Returns a dictionary mapping reachable nodes to their shortest distance from start_node.
    Distances are initialized for all nodes in all_nodes.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node in all_nodes:
        distances[start_node] = 0
        queue = deque([start_node])
    else:
        # Start node is not in the list of known nodes.
        # This might happen if a robot starts at a location not defined in the grid.
        # In this case, it can only reach itself.
        if start_node in distances: # Check if it's at least a known node name
             distances[start_node] = 0
        return distances # Return distances dictionary (with start_node at 0 if known)

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    The heuristic estimates the minimum number of actions required to paint all
    goal tiles that are not yet painted correctly. For each unpainted goal tile,
    it calculates the minimum cost to get a robot to a position from which it
    can paint the tile with the correct color, considering the robot's current
    location and color. The total heuristic is the sum of these minimum costs
    for all unpainted goal tiles.

    # Assumptions
    - Tiles are initially clear unless specified as painted.
    - Once a tile is painted with a color, it cannot be cleared or repainted
      with a different color (based on domain actions and typical problem structure).
      Therefore, if a goal tile is painted with the wrong color, the state is
      considered unsolvable from that point, and the heuristic returns a large value.
    - All colors required by the goal are available for robots to pick up.
    - Action costs are uniform (implicitly 1).
    - The grid defined by up/down/left/right predicates is traversable.

    # Heuristic Initialization
    - Extracts goal conditions (`(painted tile color)` facts).
    - Parses static facts (`up`, `down`, `left`, `right`, `available-color`) to:
        - Build a graph representing tile connectivity for robot movement (`move_adj`).
        - Build a mapping from a tile to the positions a robot must be in to paint it (`paintable_adj`).
        - Identify all available colors.
        - Collect a set of all tile names mentioned in connectivity facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the state to determine the current location and color of each robot,
       and the painted status (tile and color) of each tile.
    2. Initialize the total heuristic cost to 0.
    3. Define a large penalty value to return for states considered unsolvable
       (e.g., a goal tile is painted with the wrong color, or a goal tile cannot
       be reached or painted).
    4. For each goal fact `(painted tile_X color_Y)` in the task's goals:
        a. Check if `tile_X` is already painted with `color_Y` in the current state. If yes, this goal is satisfied for this tile; continue to the next goal tile.
        b. Check if `tile_X` is painted with any color `color_Z` where `Z != color_Y`. If yes, the tile is painted incorrectly; return the large penalty value as the state is likely unsolvable.
        c. If `tile_X` is not painted (implying it is clear and needs painting):
            i. Determine the set of tiles `required_robot_locs` from which a robot can paint `tile_X` (based on the `paintable_adj` mapping built during initialization). If this set is empty, return the large penalty (the tile cannot be painted).
            ii. Initialize `min_cost_for_tile` to infinity.
            iii. For each robot `r` at its current location `r_loc`:
                - Compute the shortest distances from `r_loc` to all other tiles using BFS on the `move_adj` graph, considering all known tiles.
                - Find the minimum distance `min_dist_from_r` from `r_loc` to any tile in `required_robot_locs`.
                - If `min_dist_from_r` is finite (meaning the robot can reach a painting position):
                    - Calculate the cost for the robot to get the correct color `color_Y`. This is 0 if the robot already has `color_Y`, and 1 otherwise (assuming `change_color` is always possible if the color is available).
                    - The estimated cost to paint this tile using robot `r` is `min_dist_from_r` (move cost) + `color_cost_for_r` (color change cost) + 1 (paint action cost).
                    - Update `min_cost_for_tile = min(min_cost_for_tile, cost_via_r)`.
            iv. If after checking all robots, `min_cost_for_tile` is still infinity, it means no robot can reach a position to paint this tile; return the large penalty.
            v. Add `min_cost_for_tile` to the `total_heuristic`.
    5. Return the `total_heuristic`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        self.move_adj = {}
        self.paintable_adj = {} # tile_to_paint -> [tile_robot_must_be_at, ...]
        self.available_colors = set()
        self.all_tiles = set()

        # Process static facts to build graphs and collect info
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # parts is [predicate, tile1, tile2] where predicate(tile1, tile2) is true
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                tile1, tile2 = parts[1], parts[2]
                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)

                # Move adjacency is bidirectional
                self.move_adj.setdefault(tile1, []).append(tile2)
                self.move_adj.setdefault(tile2, []).append(tile1)

                # Paintable adjacency: robot at tile2 paints tile1 using the corresponding action
                # e.g., robot at tile_0_1 paints tile_1_1 using paint_up
                self.paintable_adj.setdefault(tile1, []).append(tile2)

            elif parts[0] == "available-color":
                self.available_colors.add(parts[1])

        self.all_tiles = list(self.all_tiles) # Convert to list for consistent ordering if needed, though not strictly necessary for BFS.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        robot_locs = {}
        robot_colors = {}
        painted_status = {} # tile -> color
        # clear_tiles = set() # Not strictly needed, implicitly handled if not painted

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot_locs[parts[1]] = parts[2]
            elif parts[0] == "robot-has":
                robot_colors[parts[1]] = parts[2]
            elif parts[0] == "painted":
                painted_status[parts[1]] = parts[2]
            # elif parts[0] == "clear":
            #     clear_tiles.add(parts[1]) # Implicitly clear if not painted

        total_heuristic = 0
        # A large penalty for unsolvable states (e.g., painted wrong color)
        # Should be larger than any possible finite heuristic value.
        # Max possible finite cost for one tile is roughly max_dist + 1 (color) + 1 (paint)
        # Max dist is at most len(all_tiles) - 1.
        # Penalty = num_goals * (len(all_tiles) + 2)
        large_penalty = len(self.goals) * (len(self.all_tiles) + 2)
        if large_penalty == 0 and len(self.goals) > 0: # Handle case with 0 tiles but goals exist? Unlikely.
             large_penalty = 10000 # Fallback large value
        elif large_penalty == 0 and len(self.goals) == 0:
             # No goals, heuristic should be 0. Penalty doesn't matter.
             pass
        elif large_penalty < 100: # Ensure penalty is reasonably large
             large_penalty = 100

        # Precompute distances from all robot locations
        dists_from_robots = {}
        for r_name, r_loc in robot_locs.items():
             # Ensure robot location is a known tile before BFS
             if r_loc in self.all_tiles:
                 dists_from_robots[r_name] = bfs(r_loc, self.move_adj, self.all_tiles)
             else:
                 # Robot is at an unknown location? Should not happen in valid problems.
                 # Treat as unreachable for now.
                 dists_from_robots[r_name] = {}


        # Iterate through goal conditions
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "painted":
                tile_X, color_Y = parts[1], parts[2]

                # Check if the tile is already painted correctly
                if tile_X in painted_status and painted_status[tile_X] == color_Y:
                    continue # Goal met for this tile

                # Check if the tile is painted with the wrong color
                if tile_X in painted_status and painted_status[tile_X] != color_Y:
                    return large_penalty # Painted wrong color, likely unsolvable

                # Tile needs painting with color_Y (it must be clear)

                # Find the set of tiles a robot must be at to paint tile_X
                required_robot_locs = self.paintable_adj.get(tile_X, [])

                if not required_robot_locs:
                     # This goal tile cannot be painted based on static facts.
                     # This might indicate a problem definition error or an unsolvable state.
                     return large_penalty

                min_cost_for_tile = float('inf')

                # Find the minimum cost across all robots to paint this tile
                for r_name, r_loc in robot_locs.items():
                    # Check if robot location is valid and reachable in the graph
                    if r_name in dists_from_robots:
                        dist_from_r = dists_from_robots[r_name]

                        min_dist_from_r = float('inf')
                        # Find min distance from robot's current location to any required painting position
                        for req_loc in required_robot_locs:
                            if req_loc in dist_from_r and dist_from_r[req_loc] != float('inf'):
                                min_dist_from_r = min(min_dist_from_r, dist_from_r[req_loc])

                        # If the robot can reach a painting position
                        if min_dist_from_r != float('inf'):
                            # Calculate color change cost for this robot
                            # Assumes color change is possible if the robot doesn't have the color
                            color_cost_for_r = 0
                            if r_name not in robot_colors or robot_colors[r_name] != color_Y:
                                # Check if the target color is available at all.
                                # Based on domain, it should be if it's in the goal.
                                # If not available, this path is impossible?
                                # Let's assume it's available if in self.available_colors.
                                if color_Y in self.available_colors:
                                     color_cost_for_r = 1
                                else:
                                     # Required color is not available. This robot cannot paint it.
                                     # This path is not viable for this robot.
                                     continue # Skip this robot for this tile

                            # Estimated cost via this robot: move + color change + paint
                            cost_via_r = min_dist_from_r + color_cost_for_r + 1
                            min_cost_for_tile = min(min_cost_for_tile, cost_via_r)

                # If after checking all robots, no robot can paint this tile
                if min_cost_for_tile == float('inf'):
                    return large_penalty # Tile cannot be painted by any robot

                # Add the minimum cost found for this tile to the total heuristic
                total_heuristic += min_cost_for_tile

        return total_heuristic
