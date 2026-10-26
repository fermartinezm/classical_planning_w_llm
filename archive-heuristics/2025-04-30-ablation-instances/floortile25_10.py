import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles
    that are not currently painted with their goal color. It simulates a greedy
    strategy: prioritize painting tiles that need the robot's current color,
    moving to the closest such tile's adjacent location, painting it, and
    repeating. If no tiles need the current color, the robot changes color
    to the one needed by the most remaining unpainted tiles.

    # Assumptions:
    - Tiles needing painting must be clear to be painted. If a tile needs painting
      but is not clear, the state is considered unsolvable (heuristic returns infinity).
    - The grid defined by adjacency predicates forms a connected graph.
    - The heuristic calculates movement distance using BFS on the full grid graph,
      ignoring the 'clear' predicate for intermediate tiles on the path. The 'clear'
      predicate is only checked initially for the tile being painted.
    - Each action (move, paint, change_color) costs 1.

    # Heuristic Initialization
    - Extracts the goal painted tiles and their colors.
    - Builds the grid graph (adjacency map) from static facts to calculate movement distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted with a specific color according to the goal,
       but are not currently painted with that color. A tile needs painting only if
       `(painted T C)` is in the goal and not in the state. Additionally, the tile must
       be `clear` in the current state to be paintable. If a tile needs painting but is
       not clear, the state is considered a dead end and the heuristic returns infinity.
       Store these as a set of (tile, color) pairs.
    2. If there are no tiles needing painting, the heuristic is 0.
    3. Get the robot's current location and the color it is holding from the state.
    4. Initialize the heuristic cost to 0.
    5. Simulate a greedy process until all necessary tiles are painted:
       a. Check if there are any remaining unpainted tiles that require the robot's current color.
       b. If yes:
          i. Among these tiles, find the one whose adjacent tile is closest to the robot's current location.
             The distance is calculated using BFS on the full grid graph (ignoring the 'clear' status of intermediate tiles).
             The target nodes for BFS are all tiles adjacent to the potential tiles to paint.
          ii. Add the minimum distance found to the heuristic cost.
          iii. Update the robot's current location to the adjacent tile that is closest and reachable.
          iv. Add 1 to the heuristic cost (for the paint action).
          v. Mark the corresponding tile as painted (remove the task from the set of remaining tasks).
       c. If no (no remaining tiles need the current color):
          i. Find the color that is needed by the largest number of remaining unpainted tiles.
          ii. Add 1 to the heuristic cost (for the change_color action).
          iii. Update the robot's current color to this new color.
    6. Return the total heuristic cost accumulated during the simulation.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal painted tiles and their colors.
        - Grid adjacency map from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal painted tiles and their colors
        self.goal_painted = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_painted[tile] = color

        # Build the grid adjacency map and collect all tiles
        self.adj_map = collections.defaultdict(set)
        self.all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                t1, t2 = parts[1], parts[2]
                self.adj_map[t1].add(t2)
                self.adj_map[t2].add(t1) # Grid is undirected
                self.all_tiles.add(t1)
                self.all_tiles.add(t2)

    def bfs_distance(self, start_node, target_nodes, graph_adj):
        """
        Performs BFS on the graph_adj to find the shortest distance from start_node
        to any node in target_nodes.
        Returns (min_distance, closest_target_node) or (float('inf'), None) if no target is reachable.
        """
        if not target_nodes:
             return float('inf'), None # No targets to reach

        # Handle case where start_node is not in the graph (shouldn't happen if start_node is a tile)
        if start_node not in graph_adj and start_node not in target_nodes:
             return float('inf'), None

        queue = collections.deque([(start_node, 0)])
        visited = {start_node}
        min_dist = float('inf')
        closest_target = None

        while queue:
            current_node, dist = queue.popleft()

            # Optimization: If current distance already exceeds min found, stop exploring this path
            if dist > min_dist:
                 continue

            if current_node in target_nodes:
                # Found a target node. Since BFS explores layer by layer,
                # this is the shortest path to *this specific* target node.
                # We need the minimum distance to *any* target node.
                # We must continue BFS until the queue is empty to guarantee
                # finding the minimum distance among *all* reachable target nodes.
                if dist < min_dist:
                    min_dist = dist
                    closest_target = current_node
                # No early return here, continue exploring for potentially closer targets

            # Explore neighbors
            for neighbor in graph_adj.get(current_node, []):
                 if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # After exploring all reachable nodes, return the minimum distance found
        # If no target node was reached, min_dist remains infinity
        return min_dist, closest_target

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify robot's current location and color
        current_robot_loc = None
        current_robot_color = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                # Assuming only one robot
                current_robot_loc = parts[2]
            elif parts[0] == "robot-has":
                # Assuming only one robot
                current_robot_color = parts[2]

        if current_robot_loc is None or current_robot_color is None:
             # Should not happen in valid states
             return float('inf')

        # 2. Identify tiles needing painting
        # A tile needs painting if the goal requires (painted T C) but the state does not have it.
        # Additionally, the tile must be clear to be paintable.
        remaining_tasks = set() # Store as (tile, color)
        for tile, goal_color in self.goal_painted.items():
            # Check if the tile is already painted with the goal color
            is_painted_correctly = f"(painted {tile} {goal_color})" in state
            if not is_painted_correctly:
                 # Check if it's clear - paint precondition
                 is_clear = f"(clear {tile})" in state
                 if not is_clear:
                     # Tile needs painting but is not clear (e.g., painted wrongly, or robot is on it).
                     # In this domain, painted tiles are not clear and cannot be repainted.
                     # If a tile needs painting (goal requires C, state doesn't have C) AND it's not clear,
                     # it's likely painted with the wrong color or blocked, making it unsolvable.
                     # Return infinity as this state is a dead end for this goal tile.
                     return float('inf')

                 # If it needs painting and is clear, add it to tasks
                 remaining_tasks.add((tile, goal_color))

        # 3. If no tiles need painting, goal reached
        if not remaining_tasks:
            return 0

        # 4. Simulate greedy painting process
        h = 0
        current_loc = current_robot_loc
        current_color = current_robot_color

        # Use a copy of the set as we remove elements.
        tasks_to_process = set(remaining_tasks)

        while tasks_to_process:
            # Find tasks matching current color
            tasks_with_current_color = {(T, C) for (T, C) in tasks_to_process if C == current_color}

            if tasks_with_current_color:
                # Find the closest tile T among tasks_with_current_color
                # We need to find the minimum distance from current_loc to *any* tile
                # that is adjacent to *any* tile in tasks_with_current_color.

                potential_target_adj_locs = set()
                # Map target adjacent location back to the tile it serves, for later removal
                adj_to_tile_map = {} # {adj_loc: tile_to_paint}

                for (T, C) in tasks_with_current_color:
                    adj_locs = self.adj_map.get(T, set())
                    for adj in adj_locs:
                         potential_target_adj_locs.add(adj)
                         # Store one tile this adj loc is for. If multiple tiles share an adj loc,
                         # we just need one mapping to remove a task later.
                         if adj not in adj_to_tile_map:
                             adj_to_tile_map[adj] = T


                if not potential_target_adj_locs:
                     # Should not happen if grid is connected and tiles exist
                     # If robot is on an isolated tile and needs to paint something...
                     # Or if a tile needs painting but has no adjacent tiles defined.
                     # This implies an unsolvable state.
                     return float('inf')

                # Calculate distance from current_loc to any of the potential_target_adj_locs
                # BFS on the full grid graph (ignoring 'clear' for movement)
                dist, closest_reachable_adj = self.bfs_distance(current_loc, potential_target_adj_locs, self.adj_map)

                if dist == float('inf'):
                    # Cannot reach any adjacent tile of any remaining task with current color
                    # This path is likely unsolvable
                    return float('inf')

                # The closest_reachable_adj is adjacent to some tile T in tasks_with_current_color.
                # We need to identify which tile T this corresponds to so we can remove the task.
                # We can use the map created earlier.
                best_tile_to_paint = adj_to_tile_map.get(closest_reachable_adj)

                if best_tile_to_paint is None:
                     # This should not happen if closest_reachable_adj came from potential_target_adj_locs
                     return float('inf') # Defensive check

                # We already checked that tiles in remaining_tasks are clear at the start.
                # So best_tile_to_paint should be clear in the initial state.
                # The simulation doesn't update 'clear' status, which is a simplification.

                h += dist # Move cost
                current_loc = closest_reachable_adj # Update robot location
                h += 1 # Paint cost
                # Remove the painted task
                painted_color = current_color # Store color before potentially changing
                # Need to find the exact task (tile, color) to remove.
                # We know the tile and the color (current_color).
                task_to_remove = (best_tile_to_paint, painted_color)
                if task_to_process in tasks_to_process: # Check if the specific task is still pending
                    tasks_to_process.remove(task_to_remove)
                else:
                    # This could happen if the same tile needed painting with the same color
                    # multiple times (not possible in this domain) or if there's a logic error.
                    # Given the domain, a tile only needs to be painted once with its goal color.
                    # If the task is not found, it implies it was already painted in a previous step
                    # of the simulation, which shouldn't happen if we remove correctly.
                    # Let's return inf as it indicates an unexpected state in the simulation.
                     return float('inf')


            else: # No tasks with current color remaining
                # Need to change color. Find the most "useful" color to change to.
                # Most useful = color needed by most remaining tiles.
                color_counts = {}
                for (T, C) in tasks_to_process:
                    color_counts[C] = color_counts.get(C, 0) + 1

                if not color_counts:
                    # Should not happen if tasks_to_process is not empty
                    break # Exit loop if no tasks left

                # Find the color with the maximum count
                next_color = max(color_counts, key=color_counts.get)
                h += 1 # Cost for change_color
                current_color = next_color

        return h
