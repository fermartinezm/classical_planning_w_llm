# Assuming Heuristic base class is available in a module named heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a placeholder Heuristic class if the actual one is not provided
# In a real scenario, this would be imported.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
    def __call__(self, node):
        raise NotImplementedError

from collections import deque
from fnmatch import fnmatch


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
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box from its current location to its goal location.
    The distance is calculated on the grid graph defined by the 'adjacent'
    predicates, ignoring the robot and other boxes as obstacles.

    # Assumptions
    - The goal state specifies the target location for each box using the `(at ?b ?l)` predicate.
    - The grid defined by `adjacent` predicates is connected, allowing paths between any two locations.
    - The heuristic ignores the robot's position and the positions of other boxes,
      treating all locations as traversable for distance calculation purposes.
    - The heuristic ignores the requirement for the robot to be in a specific
      position relative to the box to push it.

    # Heuristic Initialization
    - Extract the goal location for each box from the task's goal conditions.
    - Build an undirected graph representation of the grid from the `adjacent`
      static facts to enable shortest path calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify the current location of each box from the current state.
    3. For each box:
       - Determine its current location and its goal location (extracted during initialization).
       - If the box is not already at its goal location:
         - Calculate the shortest path distance between the box's current location
           and its goal location using Breadth-First Search (BFS) on the grid graph.
         - Add this distance to the total heuristic cost.
    4. The total heuristic cost is the sum of these distances for all misplaced boxes.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the grid graph.
        """
        # The base class constructor is expected to initialize self.goals and self.static
        # super().__init__(task) # Uncomment if inheriting from a proper base class

        self.goals = task.goals
        self.static = task.static


        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are expected to be (at box location)
            if parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

        # Build the adjacency list graph from static adjacent facts.
        # Treat the graph as undirected for distance calculation.
        self.adj_list = {}
        for fact in self.static:
            parts = get_parts(fact)
            # Adjacent facts are expected to be (adjacent loc1 loc2 dir)
            if parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in self.adj_list:
                    self.adj_list[loc1] = []
                if loc2 not in self.adj_list:
                    self.adj_list[loc2] = []
                # Add connections in both directions for undirected distance
                # Avoid adding duplicates
                if loc2 not in self.adj_list[loc1]:
                    self.adj_list[loc1].append(loc2)
                if loc1 not in self.adj_list[loc2]:
                    self.adj_list[loc2].append(loc1)

    def get_distance(self, start_loc, end_loc):
        """
        Calculates the shortest path distance between two locations using BFS.
        Returns float('inf') if the end location is unreachable.
        """
        if start_loc == end_loc:
            return 0

        # Check if start or end location exists in the graph
        # If a location has no adjacencies defined, it won't be in adj_list keys.
        # BFS cannot start or end at such a location.
        if start_loc not in self.adj_list or end_loc not in self.adj_list:
             return float('inf')


        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                return dist

            # Get neighbors from the adjacency list
            neighbors = self.adj_list.get(current_loc, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # If BFS completes without finding the end_loc
        return float('inf') # Indicates unreachable

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Find current locations of all boxes relevant to the goal.
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is (at ?box ?location) and the object is one of the goal boxes
            if parts[0] == "at" and len(parts) == 3 and parts[1] in self.box_goals:
                 box, location = parts[1], parts[2]
                 current_box_locations[box] = location

        total_heuristic = 0

        # Sum distances for all boxes that are not at their goal location.
        for box, goal_location in self.box_goals.items():
            current_location = current_box_locations.get(box)

            # If a box from the goal is not found in the current state,
            # it implies an invalid state or the box is somehow removed.
            # For a standard Sokoban problem, all boxes should always exist.
            # If not found, treat as infinite cost.
            if current_location is None:
                 return float('inf')

            if current_location != goal_location:
                distance = self.get_distance(current_location, goal_location)
                # If distance is inf, the total heuristic is inf
                if distance == float('inf'):
                    return float('inf')
                total_heuristic += distance

        return total_heuristic
