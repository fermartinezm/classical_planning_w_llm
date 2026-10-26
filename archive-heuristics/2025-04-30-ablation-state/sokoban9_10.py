# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe return empty list or raise error
        # For this context, assuming valid fact strings
        return []
    return fact[1:-1].split()

# Need to import deque for BFS
from collections import deque
# Need to import Heuristic base class
from heuristics.heuristic_base import Heuristic

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each box not at its goal,
    the shortest path distance from the box's current location to its goal location,
    plus the shortest path distance from the robot's current location to the box's current location.
    Distances are computed on the static graph of locations defined by 'adjacent' predicates.

    # Assumptions
    - The locations form a graph defined by 'adjacent' predicates.
    - The cost of moving the robot between adjacent locations is 1.
    - The cost of a 'push' action is implicitly handled; the heuristic focuses on movement distances.
    - The heuristic ignores potential deadlocks where boxes cannot be moved further due to obstacles or walls.
    - The heuristic assumes the robot can always reach a box and push it if the target location is clear (which is a relaxation).
    - The goal state is defined by the locations of the boxes and potentially the robot or clear locations. The heuristic explicitly checks if the full goal is met to return 0.

    # Heuristic Initialization
    - Builds an undirected graph of locations based on 'adjacent' predicates.
    - Computes all-pairs shortest paths between locations using BFS on this graph.
    - Extracts the goal location for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. During initialization (`__init__`):
        a. Iterate through the `task.static` facts to find all `(adjacent loc1 loc2 dir)` predicates.
        b. Build an undirected graph where nodes are locations and edges connect `loc1` and `loc2` if they are adjacent in any direction. Store all unique locations encountered.
        c. For every location in the graph, run a Breadth-First Search (BFS) to compute the shortest path distance to all other reachable locations. Store these distances in a dictionary, e.g., `self.distances[start_loc][end_loc]`.
        d. Iterate through the `task.goals` facts to find all `(at box loc)` predicates. Store these as the goal locations for each box in a dictionary `self.goal_locations[box_name] = goal_location`. Note that goals might include predicates other than `(at box loc)`.
    2. In the heuristic computation method (`__call__`) for a given `node` (representing a state):
        a. Check if the current state (`node.state`) satisfies all goal conditions (`self.goals <= node.state`). If yes, return 0.
        b. Find the robot's current location by searching for the `(at-robot ?l)` fact in `node.state`.
        c. Find the current location for each object that has a goal location (which we assume are boxes) by searching for `(at obj ?l)` facts in `node.state`. Store these in a temporary dictionary `current_object_locations`.
        d. Initialize the total heuristic cost `total_cost = 0`.
        e. Iterate through each object (box) and its goal location stored in `self.goal_locations`.
        f. For the current object `obj` and its `goal_location`:
            i. Get the object's `current_location` from `current_object_locations`. If the object is not found (should not happen in valid states), skip or handle as an error.
            ii. If `current_location` is the same as `goal_location`, this object is already in place; add 0 to `total_cost` for this object.
            iii. If the object is not at its goal:
                - Look up the shortest distance from `current_location` to `goal_location` in the pre-computed `self.distances`. This distance estimates the minimum number of pushes required for this box, ignoring obstacles. Let this be `box_dist`.
                - Look up the shortest distance from the robot's current location (`robot_loc`) to the box's `current_location` in `self.distances`. This distance estimates the robot's effort to reach the box's vicinity to prepare for a push. Let this be `robot_dist`.
                - If either `box_dist` or `robot_dist` is `float('inf')` (meaning the location is unreachable in the static graph), the state is likely unsolvable or a deadlock; return `float('inf')` immediately.
                - Add `box_dist + robot_dist` to `total_cost`.
        g. Return the final `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, computing distances,
        and extracting box goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Build the location graph and get all locations
        self.locations = set()
        # Using a dict to store adjacency: location -> {neighbor_location} for undirected graph
        self.graph = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                # Add undirected edge
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1)

        # 2. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # 3. Extract box goal locations
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming 'at' predicates in goals only apply to objects that are boxes
            if parts and parts[0] == "at" and len(parts) == 3: # Ensure it's (at obj loc)
                obj, location = parts[1], parts[2]
                self.goal_locations[obj] = location
            # Note: Other goal conditions like (at-robot ...) or (clear ...) are handled
            # by the explicit goal check `self.goals <= state` in __call__.

    def _bfs(self, start_loc):
        """
        Perform BFS from a start location to find distances to all reachable locations
        in the undirected graph.
        Returns a dictionary mapping location to distance.
        """
        distances = {loc: float('inf') for loc in self.locations}
        # Handle case where start_loc might not be in self.locations (e.g., malformed state)
        if start_loc not in self.locations:
             # Cannot compute distances from an unknown location
             return distances # All distances remain inf

        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()

            # Iterate through neighbors in the undirected graph
            if current_loc in self.graph: # Check if current_loc is in the graph keys
                for neighbor_loc in self.graph[current_loc]:
                    if distances[neighbor_loc] == float('inf'):
                        distances[neighbor_loc] = distances[current_loc] + 1
                        queue.append(neighbor_loc)

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 2a. Check if goal is reached
        if self.goals <= state:
             return 0

        # 2b. Find robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
                break
        # If robot_loc is None, state is malformed, return inf
        if robot_loc is None:
             return float('inf')

        # 2c. Find object locations (specifically those that are boxes with goals)
        current_object_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check for 'at' predicates involving objects that are boxes (i.e., have a goal location)
            if parts and parts[0] == "at" and len(parts) == 3 and parts[1] in self.goal_locations:
                 current_object_locations[parts[1]] = parts[2]

        # 2d. Initialize total heuristic cost
        total_cost = 0

        # 2e, 2f. Calculate heuristic for each box not at its goal
        for obj, goal_location in self.goal_locations.items():
            current_location = current_object_locations.get(obj)

            # If object is not found in state, state is malformed, return inf
            if current_location is None:
                 return float('inf')

            # If object is already at goal, cost is 0 for this object
            if current_location == goal_location:
                continue

            # Calculate object distance to goal
            # Check if locations are in the distance map (should be if in self.locations)
            if current_location not in self.distances or goal_location not in self.distances[current_location]:
                 # This means the goal location is unreachable from the object's current location in the static graph.
                 # This state is likely unsolvable.
                 return float('inf')

            box_dist = self.distances[current_location][goal_location]

            # Calculate robot distance to object
            # Check if locations are in the distance map
            if robot_loc not in self.distances or current_location not in self.distances[current_location]:
                 # This means the object location is unreachable from the robot's current location in the static graph.
                 # This state is likely unsolvable.
                 return float('inf')

            robot_dist = self.distances[robot_loc][current_location]

            # If any distance is inf, it means unreachability in the static graph
            if box_dist == float('inf') or robot_dist == float('inf'):
                 # This check is redundant due to the checks above, but harmless.
                 return float('inf')

            # Add costs for this object (box)
            total_cost += box_dist + robot_dist

        # 2g. Return the total heuristic cost
        return total_cost
