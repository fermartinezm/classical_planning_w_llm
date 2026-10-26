import sys
from collections import deque
from fnmatch import fnmatch

# Try to import the base class. If running standalone or in an environment
# where the base class is not directly available, define a dummy class.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Heuristic base class not found. Defining a dummy base class.", file=sys.stderr)
    class Heuristic:
        """Dummy base class for Heuristic."""
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError("Heuristic calculation not implemented.")

def get_parts(fact: str) -> list[str]:
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes surrounding parentheses and splits by space.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    Handles potential empty strings if fact is malformed, though expects valid PDDL format.
    """
    if not fact or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Handle malformed facts gracefully, though they shouldn't occur
        print(f"Warning: Malformed fact string encountered: {fact}", file=sys.stderr)
        return []
    return fact[1:-1].split()

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain, designed for use
    with Greedy Best-First Search.

    # Summary
    This heuristic estimates the cost to reach the goal state in a Sokoban problem.
    The primary estimation is based on the sum of two components for each box not
    yet at its goal location:
    1. The shortest path distance for the box to reach its target location.
    2. The shortest path distance for the robot to reach the box's current location.
    This sum approximates the combined effort of robot movement and box pushing.
    Distances are calculated based on the empty grid graph derived from static
    'adjacent' predicates, ignoring dynamic obstacles like other boxes for efficiency.

    # Assumptions
    - The cost of both 'move' (robot moves) and 'push' (robot pushes box) actions is 1.
    - The heuristic uses shortest path distances on the static grid graph. It does not
      account for the fact that paths might be blocked by other boxes in the current state.
      This makes the computation fast but potentially less accurate than a state-aware distance.
    - Each box specified in the goal conditions has a unique target location.
    - The grid connectivity is fully defined by the static 'adjacent' predicates in the PDDL file.
    - Box names consistently start with 'box' (case-sensitive) to distinguish them.
    - The PDDL file provides symmetric 'adjacent' predicates for traversability (e.g., if
      `(adjacent l1 l2 right)` exists, `(adjacent l2 l1 left)` also exists).

    # Heuristic Initialization
    - Stores the `task` object to access goal conditions and the `goal_reached` method.
    - Parses static `adjacent` facts to identify all unique `location` objects.
    - Builds an adjacency list (`self.adj`) representing the grid connectivity graph.
    - Pre-computes all-pairs shortest path distances between all locations using
      Breadth-First Search (BFS). Distances are stored in `self.distances`. Unreachable
      location pairs are assigned a large constant cost (`_UNREACHABLE_COST`).
    - Extracts and stores the target location for each box (`self.goal_locations`) and
      the target location for the robot (`self.robot_goal_loc`, if specified) from the
      task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check Goal State:** First, check if the current state `node.state` satisfies all goal conditions using `self.task.goal_reached()`. If true, the cost is 0, return `0.0`.
    2.  **Parse Current State:** Iterate through the facts in `node.state` to find the robot's current location (`robot_loc`) and the current location of each box (`box_locations`). If `robot_loc` cannot be found, return `float('inf')` as the state is invalid or incomplete.
    3.  **Initialize Cost:** Set `total_heuristic_cost = 0.0`.
    4.  **Iterate Through Boxes:** For each box `b` identified in the current state:
        a.  Get its current location `current_loc`.
        b.  Retrieve its target location `target_loc` from the pre-stored `self.goal_locations`.
        c.  If the box does not have a target location defined in the goal, skip it (it's either an obstacle or not relevant to the goal).
        d.  If `current_loc == target_loc`, this box is already correctly placed; continue to the next box.
        e.  **Calculate Distances:** Compute the shortest path distance `box_dist` from `current_loc` to `target_loc` using `_grid_distance()`. Compute the shortest path distance `robot_dist` from `robot_loc` to `current_loc`.
        f.  **Check Reachability:** If either `box_dist` or `robot_dist` equals `_UNREACHABLE_COST`, it implies that based on the static grid layout, the goal is impossible to reach from this state (either the box cannot reach its goal, or the robot cannot reach the box). Return `float('inf')` to signal this and prune the search branch.
        g.  **Accumulate Cost:** Add the sum (`box_dist + robot_dist`) to `total_heuristic_cost`.
    5.  **Handle Zero Cost for Non-Goals:** If `total_heuristic_cost` remains `0.0` after checking all boxes (meaning all goal-relevant boxes are in their target locations), but the state is *not* a goal state (failed the check in step 1):
        a. Check if there is a specific goal location `self.robot_goal_loc` for the robot and if the robot is not currently at that location.
        b. If so, calculate the distance `robot_move_cost = _grid_distance(robot_loc, self.robot_goal_loc)`. Return this cost if the robot's goal is reachable (`< _UNREACHABLE_COST`), otherwise return `float('inf')`.
        c. If there's no specific robot goal, or the robot is already at its goal location, but the overall state is still not the goal (this might happen if there are other types of goal predicates not handled by this heuristic), return `1.0`. This small positive value ensures the search prefers this state over states with higher costs but correctly signals it's not the final goal.
    6.  **Return Cost:** If `total_heuristic_cost > 0.0`, return this value as the heuristic estimate.
    """
    # A large constant representing infinity for internal distance calculations.
    # Used to detect unreachable locations on the static grid.
    _UNREACHABLE_COST = 1_000_000

    def __init__(self, task):
        """
        Initializes the heuristic. Processes static facts from the task to build
        the grid graph, precompute distances, and store goal locations.
        """
        super().__init__(task) # Initialize base class if needed
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency graph and identify all locations from static facts
        self.locations = set()
        self.adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Example: (adjacent loc_1_1 loc_1_2 right)
            if parts and parts[0] == 'adjacent' and len(parts) >= 3:
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                # Add directed edge based on PDDL definition
                self.adj.setdefault(l1, []).append(l2)
                # Assumes symmetric definitions exist in PDDL for traversability

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            # Calculate distances from each location to all others
            self.distances[start_node] = self._bfs(start_node)

        # Store goal locations for boxes and potentially the robot
        self.goal_locations = {} # Map: box_name -> target_location
        self.robot_goal_loc = None # Stores robot's target location if specified
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            predicate = parts[0]
            # Goal for a box: (at box1 loc_X_Y)
            if predicate == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, loc = parts[1], parts[2]
                self.goal_locations[box] = loc
            # Goal for the robot: (at-robot loc_X_Y)
            elif predicate == 'at-robot' and len(parts) == 2:
                 self.robot_goal_loc = parts[1]

    def _bfs(self, start_node: str) -> dict[str, int]:
        """
        Performs Breadth-First Search from start_node on the static grid graph.
        Returns a dictionary mapping reachable location names to their shortest
        distance from start_node. Unreachable locations retain _UNREACHABLE_COST.
        """
        # Initialize distances for all known locations on the map
        distances = {loc: self._UNREACHABLE_COST for loc in self.locations}

        # Check if the start node is valid before proceeding
        if start_node not in self.locations:
            print(f"Warning: BFS start node '{start_node}' not found in known locations.", file=sys.stderr)
            return distances # Return all unreachable

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            # Iterate through neighbors using the precomputed adjacency list
            for neighbor in self.adj.get(current_node, []):
                # Check if neighbor is a valid location and was previously unreachable
                if neighbor in distances and distances[neighbor] == self._UNREACHABLE_COST:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def _grid_distance(self, loc1: str, loc2: str) -> int:
        """
        Retrieves the precomputed shortest grid distance between loc1 and loc2.
        Returns _UNREACHABLE_COST if either location is invalid or if loc2 is
        unreachable from loc1 based on the static grid graph.
        """
        # Check if both locations are known map locations
        if loc1 not in self.distances or loc2 not in self.locations:
             return self._UNREACHABLE_COST

        # Retrieve the distance from the precomputed map starting from loc1.
        # Default to unreachable if loc2 is not a key in the distances from loc1.
        dist = self.distances.get(loc1, {}).get(loc2, self._UNREACHABLE_COST)
        return dist

    def __call__(self, node) -> float:
        """
        Calculates the heuristic value for the given search node (state).
        Returns a float estimate of the cost to reach the goal. Returns 0.0 for
        goal states and float('inf') for states deemed unsolvable.
        """
        state = node.state

        # Check if the current state already satisfies all goal conditions
        if self.task.goal_reached(state):
            return 0.0

        # Find the robot's current location from the state facts
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at-robot':
                robot_loc = parts[1]
                break

        # If robot location is missing, the state is invalid/incomplete
        if robot_loc is None:
            print("Error: Robot location ('at-robot') not found in state.", file=sys.stderr)
            return float('inf')

        # Find the current locations of all boxes in the state
        box_locations = {} # Map: box_name -> current_location
        for fact in state:
            parts = get_parts(fact)
            # Check for facts like (at boxN locM)
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box_locations[parts[1]] = parts[2]

        total_heuristic_cost = 0.0
        # Calculate cost for each box that needs to reach a goal location
        for box, current_loc in box_locations.items():
            # Find the target location for this box, if one exists
            target_loc = self.goal_locations.get(box)

            # Only consider boxes that are part of the goal definition
            if target_loc is not None:
                # If the box is not already at its target
                if current_loc != target_loc:
                    # Calculate distance: box's current location to its target
                    box_dist = self._grid_distance(current_loc, target_loc)
                    # Calculate distance: robot's current location to the box's location
                    robot_dist = self._grid_distance(robot_loc, current_loc)

                    # Check for unreachability based on static grid
                    if box_dist >= self._UNREACHABLE_COST or robot_dist >= self._UNREACHABLE_COST:
                        # If either distance is 'infinite', goal is likely unreachable
                        return float('inf')

                    # Add the sum of distances to the total heuristic cost
                    total_heuristic_cost += float(box_dist + robot_dist)

        # If cost is zero, all boxes are at their goals. Check if state is truly goal.
        if total_heuristic_cost == 0.0:
             # If task.goal_reached was false, check robot position if relevant
             if self.robot_goal_loc and robot_loc != self.robot_goal_loc:
                 # Calculate cost for robot to reach its goal
                 cost = self._grid_distance(robot_loc, self.robot_goal_loc)
                 # Return robot move cost or infinity if unreachable
                 return float(cost) if cost < self._UNREACHABLE_COST else float('inf')
             else:
                 # Boxes are ok, robot is ok (or no goal), but still not goal state.
                 # Return 1.0 to indicate it's not a goal but has low remaining cost.
                 return 1.0

        # Return the total calculated heuristic cost
        return total_heuristic_cost
