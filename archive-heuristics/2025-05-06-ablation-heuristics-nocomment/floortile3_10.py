from collections import deque
from fnmatch import fnmatch # Although direct string comparison is used, keep import if fnmatch was intended for broader use
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Removes parentheses and splits a PDDL fact string into parts."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def bfs_distance(start_tile, target_tiles, graph, tile_states):
    """
    Finds the shortest path distance from start_tile to any tile in target_tiles
    using only tiles marked as 'clear' in tile_states for intermediate steps.
    The start_tile itself does not need to be clear.

    Args:
        start_tile (str): The tile name where the path starts.
        target_tiles (list): A list of tile names that are potential targets.
        graph (dict): Adjacency list representation of the grid graph {tile: [neighbor_tile, ...]}
        tile_states (dict): Dictionary mapping tile names to their state ('clear' or 'painted_color').

    Returns:
        int or float('inf'): The shortest distance, or infinity if no path exists
                             to any target tile through clear tiles.
    """
    if start_tile not in graph and start_tile not in target_tiles:
         # Start tile is not part of the known grid or a target
         return float('inf')

    q = deque([(start_tile, 0)])
    visited = {start_tile}
    min_dist = float('inf')

    while q:
        curr_tile, dist = q.popleft()

        # If the current tile is one of the targets, record the distance
        if curr_tile in target_tiles:
            min_dist = min(min_dist, dist)
            # Optimization: If we found a target, we only need to explore paths
            # that are strictly shorter than the current min_dist.
            # However, for correctness of finding the minimum over *all* targets,
            # it's safer to let BFS explore fully or prune carefully.
            # The current BFS explores layer by layer, so the first time we reach
            # a target is the shortest path to *that specific target*.
            # We need the min over *all* targets. Let's add pruning.
            if dist > min_dist and min_dist != float('inf'):
                 continue # Prune this path if it's already longer than the best found target distance


        # Explore neighbors
        for neighbor in graph.get(curr_tile, []):
            # Can move to a neighbor only if it is clear and not visited
            # The tile_states dict only contains tiles explicitly mentioned as clear or painted.
            # If a tile is not in tile_states or not marked clear, assume it's not traversable.
            if neighbor not in visited and tile_states.get(neighbor) == 'clear':
                visited.add(neighbor)
                q.append((neighbor, dist + 1))

    return min_dist

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Estimates the cost to reach the goal state by summing the estimated costs
    for each unsatisfied goal tile. The cost for a single tile is the minimum
    cost for any robot to change color (if needed), move to a position adjacent
    to the tile, and paint it. Movement is estimated using BFS on the grid
    considering only clear tiles as traversable.

    This heuristic is non-admissible and designed for greedy best-first search.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information.

        Args:
            task (Task): The planning task object containing goals and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Heuristic Initialization:
        # 1. Store goal tiles and their required colors.
        # 2. Build the grid graph (adjacency list) from static facts.

        self.goal_tiles = {}  # {tile: color}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'painted':
                # Goal is (painted tile color)
                if len(parts) == 3:
                    self.goal_tiles[parts[1]] = parts[2]
                # Handle potential conjunctive goals like (and (painted t1 c1) ...)
                # The Task object flattens goals into a set of facts, so we just process facts.
                # If goal is (and ...), task.goals is the set of facts inside the and.
                # If goal is a single fact, task.goals is a set containing that fact.
                pass # Already handled by iterating task.goals

        self.grid_graph = {}  # {tile: [neighbor_tile, ...]}
        # Build adjacency list from up, down, left, right facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and len(parts) == 3:
                pred, tile1, tile2 = parts
                if pred in ['up', 'down', 'left', 'right']:
                    # These predicates define adjacency. Add bidirectional edges.
                    self.grid_graph.setdefault(tile1, []).append(tile2)
                    self.grid_graph.setdefault(tile2, []).append(tile1)

        # Remove duplicates from adjacency lists
        for tile in self.grid_graph:
            self.grid_graph[tile] = list(set(self.grid_graph[tile]))


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node (Node): The search node containing the current state.

        Returns:
            int or float('inf'): The estimated cost to reach the goal.
        """
        state = node.state

        # Step-By-Step Thinking for Computing Heuristic:
        # 1. Extract current state information: robot locations, robot colors, tile states (clear/painted).
        # 2. Identify unsatisfied goal tiles: tiles that need painting according to the goal
        #    but are not currently painted with the correct color.
        # 3. If any goal tile is painted with the wrong color, return infinity (unsolvable state).
        # 4. If no unsatisfied goal tiles, return 0 (goal state).
        # 5. For each unsatisfied goal tile, estimate the minimum cost for *any* robot
        #    to paint that tile. This cost includes:
        #    a. Cost to change robot's color (1 if wrong color, 0 if correct).
        #    b. Cost to move the robot from its current location to a clear tile
        #       adjacent to the goal tile (a "painting location"). This is estimated
        #       using BFS on the grid graph, only traversing through clear tiles.
        #    c. Cost of the paint action (1).
        # 6. Sum the minimum costs calculated for each unsatisfied goal tile.

        # 1. Extract current state information
        robot_locations = {}  # {robot: tile}
        robot_colors = {}     # {robot: color}
        tile_states = {}      # {tile: 'clear' or 'painted_color'}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'robot-at' and len(parts) == 3:
                robot_locations[parts[1]] = parts[2]
            elif predicate == 'robot-has' and len(parts) == 3:
                robot_colors[parts[1]] = parts[2]
            elif predicate == 'clear' and len(parts) == 2:
                tile_states[parts[1]] = 'clear'
            elif predicate == 'painted' and len(parts) == 3:
                tile_states[parts[1]] = 'painted_' + parts[2]

        # 2. Identify unsatisfied goal tiles and check for wrongly painted tiles
        unsatisfied_goals = [] # [(tile, color), ...]

        for tile, goal_color in self.goal_tiles.items():
            current_state = tile_states.get(tile)

            if current_state == 'painted_' + goal_color:
                # Goal satisfied for this tile
                continue
            elif current_state is not None and current_state.startswith('painted_'):
                # Tile is painted, but with the wrong color
                # Assumptions: In solvable instances, tiles are not painted with the wrong color.
                # If this happens, the state is likely a dead end.
                return float('inf')
            elif current_state == 'clear':
                 # Tile needs painting and is currently clear
                 unsatisfied_goals.append((tile, goal_color))
            # If current_state is None, the tile is not explicitly clear or painted.
            # Assuming all relevant tiles are described in the state.
            # If a goal tile is not described, it's not painted correctly and not clear,
            # which might indicate an issue or an unsolvable state depending on domain interpretation.
            # Given the domain, tiles are either clear or painted. If not painted correctly,
            # it must be clear or painted wrongly. Wrongly painted is handled.
            # So, if not painted correctly, and not painted wrongly, it must be clear.
            # The check `current_state == 'clear'` handles this explicitly.

        # 4. If no unsatisfied goal tiles, return 0
        if not unsatisfied_goals:
            return 0

        # 5. Calculate total heuristic by summing minimum costs for each unsatisfied tile
        h = 0

        for tile_T, goal_color_C in unsatisfied_goals:
            # Find painting locations for tile_T: tiles X such that robot at X can paint T.
            # This means T is a neighbor of X according to the grid graph.
            painting_locations_T = [X for X, neighbors in self.grid_graph.items() if tile_T in neighbors]

            # If a goal tile has no adjacent tiles in the graph, it cannot be painted.
            if not painting_locations_T:
                 return float('inf') # Should not happen in valid grid problems

            # Find clear painting locations for tile_T. Robot must move *to* a clear tile.
            # However, the paint action does *not* require the robot's location to be clear.
            # The move action *does* require the destination to be clear.
            # So, the robot needs to move from L_R to some X in painting_locations_T.
            # The path must go through clear tiles, but the start (L_R) and end (X)
            # tiles of the path do not need to be clear for the BFS calculation itself.
            # The BFS finds the shortest path through clear intermediate tiles.
            # The target tiles for BFS are simply the painting_locations_T.

            min_cost_for_tile = float('inf')

            for robot, current_location in robot_locations.items():
                current_color = robot_colors.get(robot) # Get robot's current color

                # a. Cost to change color
                color_cost = 1 if current_color != goal_color_C else 0

                # b. Cost to move to a painting location
                # BFS finds distance from current_location to any tile in painting_locations_T
                # using only clear tiles for traversal.
                move_cost = bfs_distance(current_location, painting_locations_T, self.grid_graph, tile_states)

                # If no path exists to any painting location through clear tiles
                if move_cost == float('inf'):
                    continue # This robot cannot paint this tile in this state

                # c. Cost of paint action
                paint_cost = 1

                cost_for_robot = color_cost + move_cost + paint_cost
                min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)

            # If no robot can paint this tile from the current state
            if min_cost_for_tile == float('inf'):
                 return float('inf') # Problem likely unsolvable from this state

            h += min_cost_for_tile

        # 6. Return the total sum
        return h

