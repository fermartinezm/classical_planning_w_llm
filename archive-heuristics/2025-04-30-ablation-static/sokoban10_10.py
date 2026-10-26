from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to find shortest path distance
def bfs_distance(start_loc, end_loc, graph):
    """
    Computes the shortest path distance between start_loc and end_loc
    in the given graph using BFS.
    Graph is an adjacency dictionary: location -> list of adjacent locations.
    Returns distance or float('inf') if unreachable.
    """
    if start_loc == end_loc:
        return 0

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end_loc:
            return dist

        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # Not reachable


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    shortest path distances for each box from its current location to its
    assigned goal location, plus the shortest path distance from the robot
    to the closest box that is not yet at its goal. The distances are
    calculated on the location graph defined by the 'adjacent' predicates.

    # Assumptions
    - Each box has a unique goal location specified in the task goals.
    - The location graph defined by 'adjacent' predicates is static.
    - The heuristic does not explicitly check for deadlocks (e.g., pushing a box into a corner).
    - The robot's effort to get into a specific pushing position is approximated
      by its distance to the box's location.

    # Heuristic Initialization
    - Build the location graph (adjacency list) from the 'adjacent' static facts.
    - Build the reverse location graph.
    - Extract the goal location for each box from the task's goal conditions.
    - Precompute shortest path distances from all locations to each goal location
      using BFS on the reverse graph.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box from the task's goal conditions (done in __init__).
    2. Build the location graph and precompute distances to goals (done in __init__).
    3. For the current state:
       a. Find the current location of the robot by examining the 'at-robot' fact.
       b. Find the current location of each box by examining the 'at' facts.
       c. Initialize the total heuristic cost to 0.
       d. Calculate the sum of box-goal distances:
          i. For each box:
             - Get its current location and its goal location.
             - If the box is not at its goal location:
                 - Get the precomputed shortest path distance between the box's current
                   location and its goal location.
                 - Add this distance to the total heuristic cost.
       e. Calculate the robot-to-closest-box distance:
          i. Find the minimum shortest path distance from the robot's current location
             to the location of any box that is not yet at its goal.
          ii. If such a box exists and the distance is finite, add this minimum distance
              to the total heuristic cost.
    4. Return the total heuristic cost. If any required distance was infinite (unreachable),
       the total cost will be infinite, indicating a likely unsolvable state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, reverse graph,
        extracting goal locations for boxes, and precomputing distances to goals.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph and collect all mentioned locations
        self.location_graph = {}
        self.all_locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                self.location_graph[loc1].append(loc2)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Build the reverse location graph
        self.reverse_location_graph = {}
        for loc1, neighbors in self.location_graph.items():
            for loc2 in neighbors:
                if loc2 not in self.reverse_location_graph:
                    self.reverse_location_graph[loc2] = []
                self.reverse_location_graph[loc2].append(loc1)

        # Extract goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            # Goal facts are typically (at box loc)
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                # Assuming objects starting with 'box' are boxes
                if len(parts) == 3 and parts[1].startswith('box'):
                     _, box, loc = parts
                     self.box_goals[box] = loc

        # Precompute distances from all locations *to* each goal location using BFS on the reverse graph
        self._precompute_distances()


    def _precompute_distances(self):
        """
        Precomputes shortest path distances from all locations to all goal locations.
        Stores results in self.distances: {goal_loc: {start_loc: distance}}
        Uses BFS on the reverse graph.
        """
        self.distances = {}

        for goal_loc in set(self.box_goals.values()):
            self.distances[goal_loc] = {}
            queue = deque([(goal_loc, 0)])
            visited = {goal_loc}
            self.distances[goal_loc][goal_loc] = 0

            while queue:
                current_loc, dist = queue.popleft()

                # Neighbors in the reverse graph are locations that can reach current_loc
                if current_loc in self.reverse_location_graph:
                    for neighbor in self.reverse_location_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[goal_loc][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

            # Ensure all known locations have a distance entry (inf if unreachable)
            for loc in self.all_locations:
                 if loc not in self.distances[goal_loc]:
                     self.distances[goal_loc][loc] = float('inf')


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        This is the sum of shortest path distances for each box to its goal,
        plus the distance from the robot to the closest box needing movement.
        """
        state = node.state  # Current world state.

        # Find current location of the robot
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_loc = get_parts(fact)
                break

        # Find current location of each box and identify boxes needing movement
        current_box_locations = {}
        boxes_to_move = []
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                # Assuming objects starting with 'box' are boxes
                if len(parts) == 3 and parts[1].startswith('box'):
                     box, loc = parts[1], parts[2]
                     current_box_locations[box] = loc
                     if box in self.box_goals and loc != self.box_goals[box]:
                         boxes_to_move.append(box)


        total_cost = 0  # Initialize action cost counter.
        any_box_unreachable = False

        # Sum distances for boxes not at their goal
        for box in boxes_to_move:
            current_location = current_box_locations.get(box)
            goal_location = self.box_goals[box]

            # Get precomputed distance from current_location TO goal_location
            # Use .get with default float('inf') for safety if location/goal not in precomputed dict
            distance = self.distances.get(goal_location, {}).get(current_location, float('inf'))

            if distance == float('inf'):
                 any_box_unreachable = True
                 break # If one box goal is unreachable, the state is likely unsolvable

            total_cost += distance

        # If any box goal is unreachable, return infinity
        if any_box_unreachable:
             return float('inf')

        # Add robot distance to the closest box that needs moving
        min_robot_to_box_dist = float('inf')
        if robot_loc and boxes_to_move:
             for box in boxes_to_move:
                 box_loc = current_box_locations.get(box)
                 if box_loc: # Ensure box location is known
                     # Calculate distance from robot_loc TO box_loc using BFS on the forward graph
                     dist = bfs_distance(robot_loc, box_loc, self.location_graph)
                     min_robot_to_box_dist = min(min_robot_to_box_dist, dist)

        if min_robot_to_box_dist != float('inf'):
             total_cost += min_robot_to_box_dist

        return total_cost
