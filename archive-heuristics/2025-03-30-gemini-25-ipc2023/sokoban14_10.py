import heapq
from collections import deque
# Ensure the Heuristic base class is correctly imported or defined.
# If the environment provides 'heuristics.heuristic_base', use that.
# Otherwise, define a placeholder for standalone execution/testing.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails
    class Heuristic:
        """Placeholder for the Heuristic base class."""
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found.")

def get_parts(fact: str):
    """
    Helper function to parse PDDL fact strings like '(predicate obj1 obj2 ...)'
    into a list of strings ['predicate', 'obj1', 'obj2', ...].
    Removes the surrounding parentheses and splits the content by space.

    Args:
        fact: A string representing a PDDL fact.

    Returns:
        A list of strings representing the components of the fact.
    """
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in Sokoban,
    designed for use with greedy best-first search (admissibility is not required).
    It primarily estimates the number of 'push' actions required by summing the
    shortest path distances for each box from its current location to its target
    goal location. Additionally, it adds the shortest path distance from the
    robot's current location to the nearest misplaced box, estimating the initial
    'move' actions needed to start pushing. Distances are computed on the static
    grid graph using Breadth-First Search (BFS), ignoring dynamic obstacles like
    other boxes for simplicity and efficiency. The heuristic incorporates a basic
    dead-end detection mechanism: if a box is located in a square from which no
    goal location is reachable (based on the static grid), the state is assigned
    an infinite heuristic value, effectively pruning such branches.

    # Assumptions
    - The main difficulty and cost in Sokoban involve pushing boxes. Each 'push'
      action has a cost of 1. 'move' actions are primarily for positioning.
    - The shortest path distance for a box on the empty grid to its goal is a
      reasonable estimate of the minimum number of pushes required for that box.
    - Blocking effects between boxes are ignored in the distance calculation to
      keep the heuristic computation fast.
    - The cost for the robot to start interacting with the boxes is estimated by
      its distance to the nearest box that is not yet in its goal location.
    - Locations are considered potential dead ends if they cannot reach any goal
      location through any path on the static grid. Pushing a box into such a
      non-goal square makes the state unsolvable in this heuristic's view.

    # Heuristic Initialization (`__init__`)
    - The constructor takes the planning `task` object as input.
    - It parses the `task.goals` to identify the target location for each box,
      storing this in `self.goal_locations` (e.g., `{'box1': 'loc_A', 'box2': 'loc_B'}`).
    - It processes the `task.static` facts, specifically the `(adjacent loc1 loc2 dir)`
      predicates, to build an undirected adjacency list representation (`self.adj`)
      of the level's grid structure. It also collects all unique location names
      in `self.all_locations`.
    - It precomputes all-pairs shortest path distances between every pair of locations
      on the grid using BFS. The results are stored in `self.distances`, where
      `self.distances[loc1][loc2]` gives the shortest distance. Unreachable pairs
      are implicitly handled (key `loc2` won't exist in `self.distances[loc1]`).
    - It identifies potential "dead squares" (`self.dead_squares`). This is done by
      performing a multi-source BFS starting *backwards* from all goal locations.
      Any location on the grid that cannot be reached by this backwards search is
      considered a dead square, as a box there cannot possibly reach a goal.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Retrieve the current state (`frozenset` of fact strings) from the input `node`.
    2. Check if the current state satisfies the goal conditions using `self.task.goal_reached(state)`.
       If true, the goal is reached, and the heuristic value is 0.
    3. Parse the `state` to find the robot's current location (`robot_loc`) and the
       current location of each box specified in `self.goal_locations` (`box_locations`).
       If the robot's location or any required box's location is missing or invalid
       (not on the known grid), return `float('inf')` as the state is ill-defined.
    4. Initialize `total_box_distance = 0`. This will accumulate the estimated push costs.
    5. Initialize `min_robot_to_box_distance = float('inf')`. This will store the distance
       to the nearest misplaced box.
    6. Initialize `misplaced_boxes_exist = False`.
    7. Iterate through each box `b` and its corresponding target location `goal_loc`
       from `self.goal_locations`:
       a. Get the box's current location `current_loc` from `box_locations`. Check if valid.
       b. **Dead End Check:** If `current_loc` is found in the `self.dead_squares` set AND
          `current_loc` is not the `goal_loc` for this specific box `b`, then this box
          is in a dead end. Return `float('inf')` immediately.
       c. If `current_loc` is different from `goal_loc`:
          i. Mark `misplaced_boxes_exist = True`.
          ii. Look up the precomputed shortest distance from `current_loc` to `goal_loc`:
             `box_dist = self.distances[current_loc].get(goal_loc)`.
          iii. If `box_dist` is `None`, the goal is unreachable for this box. Return `float('inf')`.
          iv. Add `box_dist` to `total_box_distance`.
          v. Look up the precomputed shortest distance from `robot_loc` to `current_loc`:
             `robot_dist = self.distances[robot_loc].get(current_loc)`.
          vi. If `robot_dist` is not `None` (i.e., the robot can reach the box location),
              update `min_robot_to_box_distance = min(min_robot_to_box_distance, robot_dist)`.
    8. After iterating through all goal boxes: If `misplaced_boxes_exist` is still `False`,
       it implies the goal state was reached (this case should be caught in step 2, but
       serves as a safeguard). Return 0.
    9. If `min_robot_to_box_distance` is still `float('inf')`, it means the robot cannot
       reach any of the boxes that need to be moved. The state is likely unsolvable
       from the robot's current position. Return `float('inf')`.
    10. The final heuristic value is the sum of the estimated push costs for all boxes
        plus the estimated cost for the robot to move to the nearest misplaced box:
        `total_box_distance + min_robot_to_box_distance`.
    """

    def __init__(self, task):
        # Initialize the base class if necessary
        super().__init__(task)
        self.task = task
        self.goal_locations = {}

        # 1. Parse goals to find target location for each box
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                # Assumes format (at <box_name> <location_name>)
                self.goal_locations[parts[1]] = parts[2]

        # 2. Build adjacency graph from static 'adjacent' facts
        self.adj = {}
        self.all_locations = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2 = parts[1], parts[2]
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)
                # Assuming symmetry based on PDDL examples (e.g., (adj a b right), (adj b a left))
                # Build an undirected graph representation
                self.adj.setdefault(loc1, set()).add(loc2)
                self.adj.setdefault(loc2, set()).add(loc1)

        # 3. Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for loc in self.all_locations:
            # distances[start_loc] maps destination_loc -> distance
            self.distances[loc] = self._bfs(loc)

        # 4. Precompute dead squares using backwards reachability from goals
        goal_locs_set = set(self.goal_locations.values())
        # Filter for goal locations that actually exist on the map
        valid_goal_locs = goal_locs_set.intersection(self.all_locations)

        reachable_from_goal = set()
        if valid_goal_locs:
            # Start BFS from all valid goal locations simultaneously
            queue = deque(valid_goal_locs)
            visited = set(valid_goal_locs)
            while queue:
                loc = queue.popleft()
                reachable_from_goal.add(loc)
                # Explore neighbors using the adjacency list
                for neighbor in self.adj.get(loc, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        # Dead squares are all locations minus those reachable from any goal
        self.dead_squares = self.all_locations - reachable_from_goal

    def _bfs(self, start_node):
        """
        Performs Breadth-First Search on the grid graph starting from start_node.
        Uses the precomputed adjacency list `self.adj`.

        Args:
            start_node: The location name to start BFS from.

        Returns:
            A dictionary mapping reachable location names to their shortest distance
            from start_node. Returns an empty dictionary if start_node is invalid.
        """
        if start_node not in self.all_locations:
            # Handle cases where the start node might be isolated or invalid
            return {}

        distances = {start_node: 0}
        queue = deque([start_node])
        visited = {start_node} # Track visited nodes

        while queue:
            current_node = queue.popleft()
            # Explore neighbors based on the adjacency list
            for neighbor in self.adj.get(current_node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Calculates the heuristic value for the given search node.

        Args:
            node: A search node containing the current state (`node.state`).
                  `node.state` is expected to be a frozenset of PDDL fact strings.

        Returns:
            A float representing the estimated cost to reach the goal. Returns 0
            for goal states and float('inf') for states detected as dead ends or invalid.
        """
        state = node.state

        # 1. Check if the goal is already reached
        if self.task.goal_reached(state):
            return 0

        robot_loc = None
        box_locations = {}
        # 2. Parse state to find current locations of robot and goal boxes
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                box_name = parts[1]
                # Store location only if this box is relevant to the goal
                if box_name in self.goal_locations:
                    box_locations[box_name] = parts[2]

        # 3. Validate robot location
        if robot_loc is None:
            # Cannot calculate heuristic without robot position
            return float('inf')
        if robot_loc not in self.distances:
            # Robot is on an unknown or unreachable location
            return float('inf')

        total_box_distance = 0
        min_robot_to_box_distance = float('inf')
        misplaced_boxes_exist = False

        # 4. Iterate through each box defined in the goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)

            # Validate box location
            if current_loc is None:
                # A required box is missing from the state description
                return float('inf')
            if current_loc not in self.distances:
                # Box is on an unknown or unreachable location
                return float('inf')

            # 5. Dead End Check: If box is in a dead square and not at its goal
            if current_loc in self.dead_squares and current_loc != goal_loc:
                # This state leads to a situation deemed unsolvable
                return float('inf')

            # 6. Calculate costs if the box is not in its goal location
            if current_loc != goal_loc:
                misplaced_boxes_exist = True

                # 6a. Add estimated push cost (box distance to goal)
                box_dist = self.distances[current_loc].get(goal_loc)
                if box_dist is None:
                    # The goal location is unreachable for this box
                    return float('inf')
                total_box_distance += box_dist

                # 6b. Find robot distance to this box (for positioning cost)
                robot_dist = self.distances[robot_loc].get(current_loc)
                if robot_dist is not None:
                    # Update the minimum distance to any misplaced box
                    min_robot_to_box_distance = min(min_robot_to_box_distance, robot_dist)
                # If robot_dist is None, robot can't reach this box. Continue checking others.

        # 7. Final evaluation after checking all boxes
        if not misplaced_boxes_exist:
            # If all boxes are in their goal locations (should be caught by goal check)
            return 0

        if min_robot_to_box_distance == float('inf'):
            # Robot cannot reach any of the boxes that still need to be moved
            return float('inf')

        # 8. Return the combined heuristic value
        # Sum of estimated pushes + estimated moves to the nearest misplaced box
        return total_box_distance + min_robot_to_box_distance
