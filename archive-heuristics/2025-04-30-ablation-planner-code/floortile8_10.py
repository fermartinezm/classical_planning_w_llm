from collections import deque
from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists in the environment
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.strip():
        return []
    # Remove outer parentheses and split by whitespace
    try:
        return fact.strip()[1:-1].split()
    except IndexError:
        # Handle facts like "()" or single characters if they somehow appear
        return []


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


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It sums the estimated cost for each individual unpainted
    goal tile, considering the paint action, the cost for a robot to obtain the
    required color, and the minimum movement cost for a robot to reach a clear
    adjacent tile.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' facts.
    - Tiles are either 'clear' or 'painted' with one color.
    - A tile painted with the wrong color for a goal fact makes the state unsolvable
      for that goal fact.
    - Robots can only move to 'clear' adjacent tiles.
    - Robots can only paint 'clear' adjacent tiles.
    - All tiles mentioned in goal facts must be painted with the specified color.
      Other tiles are irrelevant to the goal state.

    # Heuristic Initialization
    - Extracts all tile objects from the task by looking at predicates involving tiles
      in static and goal facts.
    - Builds an adjacency graph of tiles based on 'up', 'down', 'left', 'right'
      static facts.
    - Computes all-pairs shortest paths between tiles using BFS on the adjacency graph.
    - Stores goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts of the form `(painted tile color)`.
    2. Check if any goal tile is currently painted with a *different* color than required by the goal. If so, the state is a dead end, return infinity.
    3. Identify the set of goal tiles that are *not* yet painted correctly (i.e., they are 'clear' and need painting). If this set is empty, the goal is reached, return 0.
    4. Extract the current location and color held by each robot from the state.
    5. Initialize the total heuristic value `h` to 0.
    6. For each tile `T` in the set of unpainted goal tiles, needing color `C`:
        a. Add 1 to `h` for the paint action required for tile `T`.
        b. Calculate the minimum cost for *any* robot to be able to paint tile `T`. This cost involves getting the right color and moving to a clear adjacent tile.
        c. Find all tiles adjacent to `T` based on the precomputed adjacency graph. From these, identify the subset that are currently 'clear' in the state. If there are no clear adjacent tiles, tile `T` cannot be painted, and the state is a dead end. Return infinity.
        d. For each robot `r`:
            i. Calculate the cost for robot `r` to prepare to paint `T`:
               - Cost for color change: 1 if robot `r` does not currently hold color `C`, otherwise 0.
               - Cost for movement: The minimum distance from robot `r`'s current location to any of the clear adjacent tiles found in step 6c. Use the precomputed shortest path distances. If robot `r` cannot reach any clear adjacent tile (e.g., due to a disconnected graph), this movement cost is infinity for this robot.
            ii. The total preparation cost for robot `r` for tile `T` is (color change cost) + (movement cost).
        e. Find the minimum preparation cost among all robots that *can* reach a clear adjacent tile. If no robot can reach a clear adjacent tile for this specific tile `T`, the state is a dead end (already handled in 6c, but the logic confirms it).
        f. Add this minimum robot preparation cost to `h`.
    7. Return the total heuristic value `h`.

    This heuristic sums the minimum cost to satisfy each unpainted goal fact independently, which is an overestimate but can guide greedy search effectively.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure, computing distances,
        and storing goal and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Get all tile objects by inferring from predicate arguments in static and goal facts
        self.all_tiles = set()
        facts_to_check = list(static_facts) + list(self.goals) # Check both static and goal facts

        for fact in facts_to_check:
             parts = get_parts(fact)
             if parts: # Ensure parts is not empty
                 predicate = parts[0]
                 if predicate in ['up', 'down', 'left', 'right']:
                     if len(parts) == 3:
                         self.all_tiles.add(parts[1])
                         self.all_tiles.add(parts[2])
                 elif predicate in ['clear', 'painted']:
                      if len(parts) >= 2: # painted has 3 parts, clear has 2
                         self.all_tiles.add(parts[1])
                 elif predicate == 'robot-at':
                      if len(parts) == 3:
                          # The second argument of robot-at is a tile
                          self.all_tiles.add(parts[2])

        self.all_tiles = list(self.all_tiles) # Convert to list

        # 2. Build adjacency list from static facts
        self.adj = {tile: [] for tile in self.all_tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                tile1, tile2 = parts[1], parts[2]
                # Add bidirectional edges, but only if both tiles are in our collected set
                if tile2 in self.adj:
                    self.adj[tile2].append(tile1)
                if tile1 in self.adj:
                    self.adj[tile1].append(tile2)

        # Remove duplicates from adjacency lists
        for tile in self.adj:
            self.adj[tile] = list(set(self.adj[tile]))

        # 3. Compute all-pairs shortest paths using BFS
        self.dist = {tile: {other_tile: float('inf') for other_tile in self.all_tiles} for tile in self.all_tiles}

        for start_tile in self.all_tiles:
            # Handle potential isolated tiles not in adj map
            if start_tile not in self.adj:
                 continue

            self.dist[start_tile][start_tile] = 0
            queue = deque([start_tile])
            visited = {start_tile}

            while queue:
                u = queue.popleft()
                # Ensure u is still valid and has entries in adj
                if u not in self.adj:
                    continue
                for v in self.adj[u]:
                    # Ensure v is a valid tile in our set
                    if v in self.all_tiles and v not in visited:
                        visited.add(v)
                        self.dist[start_tile][v] = self.dist[start_tile][u] + 1
                        queue.append(v)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        robot_locations = {}
        robot_colors = {}

        # Extract robot info from state
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts and parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        unpainted_goal_tiles = {} # {tile: color_needed}

        # 1. & 2. Identify unpainted and wrongly painted goal tiles
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color_needed = parts[1], parts[2]
                # Check if this goal is satisfied
                if goal not in state:
                     # Check if wrongly painted
                     is_wrongly_painted = False
                     for fact in state:
                         fact_parts = get_parts(fact)
                         # Check if fact is a painted fact for the same tile but different color
                         if fact_parts and fact_parts[0] == "painted" and len(fact_parts) == 3 and \
                            fact_parts[1] == tile and fact_parts[2] != color_needed:
                             is_wrongly_painted = True
                             break
                     if is_wrongly_painted:
                         # 2. Wrongly painted -> Dead end
                         return float('inf')
                     else:
                         # Not painted, and not wrongly painted -> must be clear (by domain definition)
                         # Add to unpainted list
                         unpainted_goal_tiles[tile] = color_needed

        # 3. If no unpainted goal tiles, goal is reached
        if not unpainted_goal_tiles:
            return 0

        h = 0

        # 6. For each tile T in the set of unpainted goal tiles
        for tile, color_needed in unpainted_goal_tiles.items():
            # a. Add 1 for the paint action
            h += 1

            # b. Calculate the minimum cost for any robot to prepare to paint T
            min_robot_prep_cost_for_tile = float('inf')

            # c. Find adjacent tiles that are clear
            # Ensure tile exists in adj map (should be true if it's a goal tile inferred from facts)
            if tile not in self.adj:
                 # This indicates a goal tile is not connected to the grid, likely unsolvable
                 return float('inf')

            adjacent_tiles = self.adj[tile]
            clear_adjacent_tiles = [adj_tile for adj_tile in adjacent_tiles if f"(clear {adj_tile})" in state]

            # If no adjacent tile is clear, this tile cannot be painted. Dead end.
            if not clear_adjacent_tiles:
                 return float('inf')

            # d. For each robot r
            for robot, robot_loc in robot_locations.items():
                cost_r = 0
                # i. Calculate cost for robot r
                # Cost for color change
                # Check if robot_colors has the robot and if the color is different
                if robot not in robot_colors or robot_colors[robot] != color_needed:
                    cost_r += 1

                # Cost for movement to a clear adjacent tile
                min_dist_r_to_clear_adj = float('inf')
                # Ensure robot_loc is in our distance map (should be true)
                if robot_loc in self.dist:
                    for clear_adj_tile in clear_adjacent_tiles:
                         # Ensure clear_adj_tile is in our distance map (should be true)
                         if clear_adj_tile in self.dist[robot_loc]:
                             min_dist_r_to_clear_adj = min(min_dist_r_to_clear_adj, self.dist[robot_loc][clear_adj_tile])

                # If robot r can reach a clear adjacent tile
                if min_dist_r_to_clear_adj != float('inf'):
                     # Total preparation cost for robot r for tile T
                     prep_cost_r = cost_r + min_dist_r_to_clear_adj
                     # e. Find the minimum preparation cost among all robots
                     min_robot_prep_cost_for_tile = min(min_robot_prep_cost_for_tile, prep_cost_r)

            # If no robot can reach a clear adjacent tile for this tile, it's a dead end.
            if min_robot_prep_cost_for_tile == float('inf'):
                 return float('inf')

            # f. Add minimum preparation cost to h
            h += min_robot_prep_cost_for_tile

        # 7. Return total heuristic value
        return h
