from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this is the correct import path for the base class

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing the shortest path distances for each misplaced box to its goal
    location and adding the shortest path distance from the robot to the
    closest misplaced box.

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - The cost of any action (move or push) is 1.
    - The heuristic does not attempt to detect complex dead-end states (e.g., boxes
      pushed into corners they cannot leave unless they are explicitly unreachable
      in the graph).
    - The robot object is not explicitly named but its location is given by the
      `(at-robot ?l)` predicate.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task goals.
    - Build a graph representation of the locations based on `adjacent` facts
      from the static information. The graph is undirected.
    - Precompute the shortest path distances between all pairs of locations
      within their connected components using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot using the `(at-robot ?l)` fact.
    2. Identify the current location of each box using the `(at ?b ?l)` facts.
    3. Identify the goal location for each box from the pre-extracted goal conditions.
    4. Initialize the total heuristic cost to 0.
    5. Create a list of misplaced boxes (boxes not at their goal location).
    6. For each box, check if its current location matches its goal location.
       If not, it's a misplaced box.
       Add the precomputed shortest distance between the current box location
       and its goal location to the total cost. If the goal is unreachable
       from the current location in the graph, add a large penalty.
    7. If there are any misplaced boxes:
       - Find the robot's current location.
       - Calculate the shortest distance from the robot's current location to the
         location of each misplaced box.
       - Find the minimum of these distances. If the robot cannot reach any
         misplaced box, add a large penalty.
       - Add this minimum distance (or penalty) to the total cost. This encourages the robot
         to move towards a box that needs pushing.
    8. Return the total heuristic cost. If there are no misplaced boxes, the
       total cost is 0 (goal state).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        location graph, and precomputing shortest path distances.
        """
        super().__init__(task)

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at box location)
                box, location = args
                self.goal_locations[box] = location
            # Assuming other goal predicates are not relevant for box goals

        # Build the location graph from adjacent facts.
        # Graph is represented as an adjacency list: {location: [adjacent_location, ...]}
        self.graph = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Add bidirectional edges
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Get all unique locations from the graph
        self.locations = list(self.graph.keys())

        # Precompute all-pairs shortest paths using BFS from each location.
        # Store distances in a dictionary: {(loc1, loc2): distance}
        self.distances = {}
        # Define a large penalty for unreachable locations
        # A value larger than any possible path length in a connected component
        # Max path length is |V|-1. Using |V| * 2 + 10 as a safe large number.
        self.unreachable_penalty = len(self.locations) * 2 + 10 if self.locations else 1000

        for start_loc in self.locations:
            self._bfs(start_loc)

    def _bfs(self, start_loc):
        """
        Perform BFS starting from start_loc to compute shortest distances
        to all other reachable locations within its connected component.
        """
        queue = deque([(start_loc, 0)])
        visited = {start_loc: 0}
        self.distances[(start_loc, start_loc)] = 0

        while queue:
            current_loc, dist = queue.popleft()

            # Ensure current_loc is in the graph (it should be if it's in self.locations)
            if current_loc in self.graph:
                for neighbor in self.graph[current_loc]:
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        self.distances[(start_loc, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

    def get_robot_location(self, state):
        """Find the current location of the robot in the state."""
        for fact in state:
            if match(fact, "at-robot", "*"):
                return get_parts(fact)[1]
        return None # Should not happen in a valid state

    def get_box_location(self, state, box_name):
        """Find the current location of a given box in the state."""
        for fact in state:
            if match(fact, "at", box_name, "*"):
                return get_parts(fact)[2]
        return None # Should not happen in a valid state for a box

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        total_cost = 0
        misplaced_boxes = []
        box_locations = {}

        # Calculate sum of box-to-goal distances and identify misplaced boxes
        for box, goal_loc in self.goal_locations.items():
            current_loc = self.get_box_location(state, box)
            box_locations[box] = current_loc # Store for robot distance calculation

            if current_loc != goal_loc:
                misplaced_boxes.append(box)
                # Add distance from current box location to its goal
                # Use precomputed distance, apply penalty if unreachable
                dist = self.distances.get((current_loc, goal_loc), self.unreachable_penalty)
                total_cost += dist

        # Add robot distance to the closest misplaced box
        if misplaced_boxes:
            robot_loc = self.get_robot_location(state)
            min_robot_dist = self.unreachable_penalty # Initialize with penalty

            for box in misplaced_boxes:
                 box_loc = box_locations[box]
                 # Get distance from robot to box, apply penalty if unreachable
                 dist = self.distances.get((robot_loc, box_loc), self.unreachable_penalty)
                 min_robot_dist = min(min_robot_dist, dist)

            # Add the minimum robot distance (or penalty) to the total cost
            total_cost += min_robot_dist

        return total_cost
