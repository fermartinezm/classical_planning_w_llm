import math
from collections import deque, defaultdict
# The problem description implies the existence of a base class Heuristic.
# We assume it's available at this path. If the execution environment differs,
# this import might need adjustment.
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extracts the components of a PDDL fact string by removing parentheses
    and splitting by space. Handles potential extra whitespace.
    Example: "(at robot1 tile_0_0)" -> ["at", "robot1", "tile_0_0"]
    """
    return fact.strip()[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their specified colors. It calculates the cost for each unsatisfied
    painting goal and sums them up. The cost for a single goal considers the
    minimum cost for *any* robot to achieve it, including moving to an adjacent
    tile from which painting is possible, potentially changing color, and
    performing the paint action itself.

    # Assumptions
    - All actions (move_up/down/left/right, paint_up/down, change_color) have a cost of 1.
    - Robots can move freely between adjacent tiles defined by `up`, `down`,
      `left`, `right` static facts. Potential blocks by other robots or
      the `clear` precondition for movement are ignored for simplicity and efficiency.
    - Painting a tile `y` can only be done from an adjacent tile `x` if
      `(up y x)` or `(down y x)` is a static fact (i.e., painting happens
      vertically). The `clear` precondition for the tile being painted (`?y`) is ignored.
    - Changing color (`change_color`) costs 1 action if the robot does not currently hold the
      required color. It's assumed the target color is always available
      (i.e., `(available-color target_color)` holds).
    - The heuristic is designed for informativeness in greedy best-first search,
      not admissibility (it might overestimate or underestimate the true cost).

    # Heuristic Initialization
    - Parses static facts (`task.static`) provided during initialization to build:
        1. `tile_graph`: An adjacency list (dictionary mapping tile -> list of adjacent tiles)
           representing tile connectivity for robot movement, based on `up`, `down`, `left`, `right` facts.
        2. `paint_source_tiles`: A dictionary mapping each paintable tile `y`
           to a set of source tiles `x` from which `y` can be painted. This is derived
           from `(up y x)` facts (for `paint_up`) and `(down y x)` facts (for `paint_down`).
    - Stores the target color for each goal tile specified in `task.goals`
      (specifically, `(painted tile color)` goals) in the `goal_painted` dictionary.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract current robot locations (`robot-at`),
        robot held colors (`robot-has`), and currently painted tiles (`painted`)
        from the input `node.state` (a frozenset of facts).
    2.  **Check Goal Completion:** If all `(painted tile color)` goals defined in
        `self.goal_painted` are present in the `current_painted` dictionary derived
        from the state, the heuristic value is 0 (goal reached).
    3.  **Handle Dead Ends (No Robots):** If there are unsatisfied painting goals but
        no robots are detected in the current state, return `math.inf` (unsolvable state).
    4.  **Iterate Unsatisfied Goals:** For each goal `(painted target_tile target_color)`
        that is not satisfied in the current state:
        a.  **Base Paint Cost:** Add 1 to the total heuristic value (`h_value`). This accounts
            for the `paint_up` or `paint_down` action itself.
        b.  **Find Paint Sources:** Retrieve the precomputed set `PaintSources` for `target_tile`
            from `self.paint_source_tiles`. If `target_tile` has no associated paint sources
            (meaning it cannot be painted according to the domain's static facts),
            return `math.inf` (unsolvable state).
        c.  **Calculate Minimum Robot Cost:** Find the minimum cost for *any* available robot
            to satisfy the preconditions for painting `target_tile` with `target_color`.
            This involves moving to a tile in `PaintSources` and potentially changing color.
            i.  Initialize `min_robot_cost = math.inf`.
            ii. Set a flag `possible_for_any_robot = False`.
            iii. For each robot `r` present in the state:
                - Get its location `robot_loc` and held color `robot_color`.
                - If `robot_loc` is unknown (e.g., robot only has `robot-has` fact),
                  this robot cannot perform the move/paint, so skip it.
                - Calculate `color_cost = 0` if `robot_color == target_color`, else `1` (for `change_color`).
                - Calculate `move_cost` using Breadth-First Search (`_bfs_shortest_path`)
                  from `robot_loc` to the nearest tile in `PaintSources` using the `tile_graph`.
                - If `move_cost` is not `math.inf` (meaning a path exists):
                    - Set `possible_for_any_robot = True`.
                    - Calculate `cost_for_this_robot = color_cost + move_cost`.
                    - Update `min_robot_cost = min(min_robot_cost, cost_for_this_robot)`.
        d.  **Check Reachability:** After checking all robots, if `possible_for_any_robot` is still `False`,
            it means no robot can reach a position to paint this specific `target_tile`.
            Return `math.inf` (dead end state).
        e.  **Add Minimum Robot Cost:** Add the calculated `min_robot_cost` (the cost for the
            best-suited robot for this specific goal) to the total heuristic value `h_value`.
    5.  **Return Total Value:** After iterating through all unsatisfied goals, return the
        accumulated `h_value`. If any step returned `math.inf`, that value will be returned.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing graph structures from static
        facts and storing goal conditions.
        """
        # Initialize the base class, if necessary (depends on Heuristic implementation)
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        self.tile_graph = defaultdict(list)
        self.paint_source_tiles = defaultdict(set)
        all_tiles = set() # Keep track of all tiles mentioned
        adj_relations = {"up", "down", "left", "right"}

        # Step 1: Build adjacency graph for movement based on static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts
            predicate = parts[0]

            if predicate in adj_relations and len(parts) == 3:
                # Example: (up t1 t2) means t1 is up from t2. They are adjacent.
                t1, t2 = parts[1], parts[2]
                all_tiles.add(t1)
                all_tiles.add(t2)
                # Add edges for movement graph (undirected)
                self.tile_graph[t1].append(t2)
                self.tile_graph[t2].append(t1)

        # Step 2: Build map of which tiles can paint which other tiles
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            # Action paint_up(?r, ?y, ?x, ?c) requires (up ?y ?x)
            # => Robot at source tile ?x paints target tile ?y if ?y is UP from ?x.
            if predicate == "up" and len(parts) == 3:
                target_tile, source_tile = parts[1], parts[2]
                all_tiles.add(target_tile)
                all_tiles.add(source_tile)
                self.paint_source_tiles[target_tile].add(source_tile)
            # Action paint_down(?r, ?y, ?x, ?c) requires (down ?y ?x)
            # => Robot at source tile ?x paints target tile ?y if ?y is DOWN from ?x.
            elif predicate == "down" and len(parts) == 3:
                target_tile, source_tile = parts[1], parts[2]
                all_tiles.add(target_tile)
                all_tiles.add(source_tile)
                self.paint_source_tiles[target_tile].add(source_tile)

        # Ensure all tiles mentioned anywhere are keys in the tile_graph,
        # even if they have no connections explicitly listed.
        for tile in all_tiles:
            if tile not in self.tile_graph:
                 self.tile_graph[tile] = []

        # Step 3: Store goal painted facts: mapping tile -> target color
        self.goal_painted = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Check if the goal is a 'painted' predicate with 3 parts (predicate, tile, color)
            if parts and parts[0] == "painted" and len(parts) == 3:
                self.goal_painted[parts[1]] = parts[2]

    def _bfs_shortest_path(self, start_node, end_nodes):
        """
        Calculates the shortest path distance from start_node to any node
        in the end_nodes set using BFS on the precomputed self.tile_graph.
        Returns math.inf if no path exists, or if start/end nodes are invalid.
        """
        # Check if start node is a valid tile in our graph connectivity
        if start_node not in self.tile_graph:
             # Robot might be on a tile not mentioned in connectivity facts
             return math.inf

        # Filter the target end_nodes to only include those that are valid tiles in our graph
        # This handles cases where a paint source might be an isolated/unknown tile.
        valid_end_nodes = {node for node in end_nodes if node in self.tile_graph}

        # If the start node is already one of the valid destinations
        if start_node in valid_end_nodes:
            return 0

        # If there are no valid, reachable destinations from the provided set
        if not valid_end_nodes:
            return math.inf

        # Initialize BFS queue and visited set
        queue = deque([(start_node, 0)]) # Store (node, distance)
        visited = {start_node}

        while queue:
            current_node, distance = queue.popleft()

            # Explore neighbors from the precomputed graph
            # tile_graph[current_node] is guaranteed to exist due to initialization logic
            for neighbor in self.tile_graph[current_node]:
                if neighbor not in visited:
                    # Check if this neighbor is one of the target destinations
                    if neighbor in valid_end_nodes:
                        return distance + 1 # Found the shortest path

                    # Otherwise, mark as visited and add to the queue
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        # If the queue becomes empty and no destination was reached
        return math.inf

    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        Estimates the remaining actions to achieve all painting goals.
        Returns 0 for goal states, math.inf for dead ends, and a positive integer otherwise.
        """
        state = node.state
        h_value = 0

        # --- Step 1: Parse Current State ---
        # Use defaultdict for easier handling of potentially missing info
        robot_info = defaultdict(lambda: {'loc': None, 'color': None})
        current_painted = {} # tile -> color
        robots_present = set() # Keep track of robots actually existing in the state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts
            predicate = parts[0]

            if predicate == "robot-at" and len(parts) == 3:
                robot, loc = parts[1], parts[2]
                robot_info[robot]['loc'] = loc
                robots_present.add(robot)
            elif predicate == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_info[robot]['color'] = color
                # Assume a robot with color also exists, add to set
                robots_present.add(robot)
            elif predicate == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                current_painted[tile] = color

        # --- Step 2: Check Goal Completion ---
        # If there are no painting goals defined for the task
        if not self.goal_painted:
            return 0

        # Count how many goals are not yet satisfied
        num_unsatisfied = 0
        for target_tile, target_color in self.goal_painted.items():
            if current_painted.get(target_tile) != target_color:
                num_unsatisfied += 1

        # If all goals are satisfied
        if num_unsatisfied == 0:
            return 0 # Goal state reached

        # --- Step 3: Handle Dead Ends (No Robots) ---
        # If goals remain but no robots are present in the state
        if not robots_present:
            return math.inf # Unsolvable state

        # --- Step 4: Iterate Unsatisfied Goals and Calculate Cost ---
        for target_tile, target_color in self.goal_painted.items():
            # Only consider goals that are not currently satisfied
            if current_painted.get(target_tile) != target_color:

                # --- Step 4a: Base Paint Cost ---
                # Add cost for the 'paint' action itself
                h_value += 1

                # --- Step 4b: Find Paint Sources ---
                # Retrieve the precomputed set of tiles from which target_tile can be painted
                paint_sources = self.paint_source_tiles.get(target_tile)

                # Check if the target tile is paintable at all according to static facts
                if not paint_sources:
                    # This goal tile cannot be painted based on domain structure
                    return math.inf # Dead end state

                # --- Step 4c: Calculate Minimum Robot Cost (Move + Color Change) ---
                min_robot_cost = math.inf
                possible_for_any_robot = False # Flag to track if any robot can do this task

                for robot in robots_present:
                    info = robot_info[robot]
                    robot_loc = info.get('loc')
                    robot_color = info.get('color') # This might be None

                    # A robot needs a location to be able to move and paint
                    if robot_loc is None:
                        continue # Skip robot if its location is unknown in this state

                    # Calculate cost components for this specific robot
                    # Cost for changing color (if needed)
                    color_cost = 0 if robot_color == target_color else 1
                    # Cost for moving to the nearest valid painting spot
                    move_cost = self._bfs_shortest_path(robot_loc, paint_sources)

                    # Check if this robot can reach a suitable painting spot
                    if move_cost != math.inf:
                        possible_for_any_robot = True # Mark that at least one robot can do it
                        cost_for_this_robot = color_cost + move_cost
                        # Update the minimum cost found so far for this goal
                        min_robot_cost = min(min_robot_cost, cost_for_this_robot)
                    # else: This robot cannot reach any source tile for this target tile

                # --- Step 4d: Check Reachability ---
                # If after checking all robots, none could reach a paint source for this goal
                if not possible_for_any_robot:
                    return math.inf # Dead end state, this goal is unreachable

                # --- Step 4e: Add Minimum Robot Cost ---
                # Add the minimum cost found (move + color change for the best robot)
                h_value += min_robot_cost

        # --- Step 5: Return Total Value ---
        # Return the accumulated heuristic value
        return h_value
