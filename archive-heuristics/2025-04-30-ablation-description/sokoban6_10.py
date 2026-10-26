# Required imports
from fnmatch import fnmatch # Included for consistency with examples, though not strictly used in get_parts
from heuristics.heuristic_base import Heuristic # Assuming this is the correct path
import collections # For deque in BFS

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Define the heuristic class
class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box to its goal location and adding the shortest
    path distance from the robot to the nearest box that is not yet at its goal.
    Distances are calculated based on the adjacency graph defined in the PDDL domain.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates, and movement is possible
      in both directions between adjacent locations for distance calculation purposes.
    - Shortest path distances on this grid represent a lower bound on movement costs.
    - Each box has a single specific goal location defined in the task goals.
    - The cost of moving the robot and pushing boxes is related to grid distances.
    - The heuristic is 0 if and only if the state is a goal state.
    - The heuristic is finite for solvable states and infinite for unsolvable states
      (specifically, if a box cannot reach its goal or the robot cannot reach a box
       that needs moving due to graph disconnection).

    # Heuristic Initialization
    - Extract goal locations for each box from the task goals.
    - Build an undirected graph of locations based on `adjacent` static facts.
    - Pre-compute all-pairs shortest path distances using BFS on this graph.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state satisfies all goal conditions (`self.goals <= state`). If yes, the heuristic is 0.
    2. Identify the current location of the robot by finding the fact `(at-robot ?l)` in the state. If not found or location is invalid, return infinity.
    3. Identify the current location of each box that has a goal by finding the fact `(at ?b ?l)` for each relevant box `?b` in the state. If a box from the goals is not found or its location is invalid, return infinity.
    4. Initialize `total_box_distance = 0` and `min_robot_to_box_distance = float('inf')`.
    5. Initialize a counter `needs_moving_boxes_count = 0`.
    6. Iterate through each box and its goal location stored during initialization (`self.box_goals`):
       a. Get the box's current location.
       b. If the box's current location is different from its goal location:
          i. Increment `needs_moving_boxes_count`.
          ii. Calculate the shortest path distance from the box's current location to its goal location using the pre-computed distances (`self.distances`). Add this distance to `total_box_distance`. If the distance is infinite (goal unreachable), the total heuristic is infinite.
          iii. Calculate the shortest path distance from the robot's current location to the box's current location using the pre-computed distances (`self.distances`). Update `min_robot_to_box_distance` with the minimum distance found so far among all boxes needing movement. If the distance is infinite (box unreachable by robot), the total heuristic is infinite.
    7. If `needs_moving_boxes_count` is 0, it means all relevant boxes are at their goals. The heuristic is 0 (this case is covered by step 1).
    8. If `needs_moving_boxes_count` is greater than 0, the heuristic value is `total_box_distance + min_robot_to_box_distance`. If any distance calculation resulted in infinity during step 6, the total heuristic will be infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the location graph and pre-computing distances."""
        # Assuming task object has 'goals' (frozenset of goal facts) and 'static' (frozenset of static facts)
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each box from the goal facts
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at box location)
            if parts and parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

        # Build the graph from adjacent facts
        self.graph = collections.defaultdict(list)
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            # Adjacent facts are (adjacent loc1 loc2 direction)
            if parts and parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2 = parts[1], parts[2]
                self.graph[loc1].append(loc2)
                # Assume adjacency is symmetric for distance calculation in the grid
                self.graph[loc2].append(loc1)
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Pre-compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """Perform BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            u = queue.popleft()
            # Ensure u is a valid key in the graph, although locations set should guarantee this
            # Use .get() for safety, although direct access should be fine if locations set is built correctly
            for v in self.graph.get(u, []):
                if distances[v] == float('inf'):
                    distances[v] = distances[u] + 1
                    queue.append(v)
        return distances

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state # state is a frozenset of facts (strings)

        # 1. Check if goal is reached
        # The Task object has a goal_reached method, but the heuristic receives the task in __init__
        # and the state in __call__. The goal conditions are stored in self.goals.
        if self.goals <= state:
             return 0

        # 2. Find robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
                break
        # If robot location is not found or is not a known location in the graph
        if robot_loc is None or robot_loc not in self.locations:
             return float('inf') # Indicates an invalid or unsolvable state

        # 3. Find current box locations for boxes relevant to the goal
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Find facts like (at box1 loc_X_Y)
            if parts and parts[0] == "at" and len(parts) == 3 and parts[1] in self.box_goals:
                 current_box_locations[parts[1]] = parts[2]

        # 4. & 5. Initialize counters and accumulators
        total_box_distance = 0
        min_robot_to_box_distance = float('inf')
        needs_moving_boxes_count = 0

        # 6. Iterate through each box and its goal location
        for box, goal_loc in self.box_goals.items():
            current_loc = current_box_locations.get(box)

            # If a box from the goal is not found in the current state, or its location is not in the graph
            if current_loc is None or current_loc not in self.locations:
                 return float('inf') # Indicates an invalid or unsolvable state

            # b. If the box is not at its goal location
            if current_loc != goal_loc:
                needs_moving_boxes_count += 1

                # ii. Calculate box-to-goal distance
                # Use .get() to safely retrieve distance, returning inf if path unknown
                box_to_goal_dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))
                if box_to_goal_dist == float('inf'):
                    # Box cannot reach its goal location
                    return float('inf')
                total_box_distance += box_to_goal_dist

                # iii. Calculate robot-to-box distance
                # Use .get() to safely retrieve distance, returning inf if path unknown
                robot_to_box_dist = self.distances.get(robot_loc, {}).get(current_loc, float('inf'))
                if robot_to_box_dist == float('inf'):
                     # Robot cannot reach this box
                     return float('inf')
                min_robot_to_box_distance = min(min_robot_to_box_distance, robot_to_box_dist)

        # 7. & 8. Compute final heuristic value
        if needs_moving_boxes_count > 0:
             # If there are boxes needing movement, add the minimum robot distance
             # If min_robot_to_box_distance is still inf here, it means no box needing movement was reachable by robot,
             # which should have been caught inside the loop, but this adds robustness.
             if min_robot_to_box_distance == float('inf'):
                 return float('inf')
             return total_box_distance + min_robot_to_box_distance
        else:
             # If needs_moving_boxes_count is 0, all relevant boxes are at their goals.
             # This case should have been handled by the initial goal check, but return 0 just in case.
             return 0
