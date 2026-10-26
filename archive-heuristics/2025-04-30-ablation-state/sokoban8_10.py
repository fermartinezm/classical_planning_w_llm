# from heuristics.heuristic_base import Heuristic # Assuming this is provided by the environment
import collections
from fnmatch import fnmatch

# Define a dummy Heuristic base class if not provided by the environment
# This is just for local testing or if the base class is not automatically available
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError


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


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing the shortest path distances for each box from its current location
    to its designated goal location. The shortest path is calculated on the
    grid graph defined by the 'adjacent' predicates, ignoring the robot and
    other boxes as obstacles.

    # Assumptions
    - The grid structure is defined by 'adjacent' predicates.
    - Each box has a unique designated goal location specified in the task goals.
    - The cost of moving a box is primarily determined by the distance it needs to travel.
    - The robot's movement cost and the need for clear paths/push positions are ignored
      for simplicity and computational efficiency (making it non-admissible).
    - Adjacency is symmetric (if A is adjacent to B, B is adjacent to A).
    - Box objects in the state facts start with the string "box".

    # Heuristic Initialization
    - Parses 'adjacent' facts from the static information to build a graph
      representation of the grid connectivity.
    - Precomputes shortest path distances between all pairs of locations
      reachable on the grid graph using Breadth-First Search (BFS).
    - Extracts the designated goal location for each box from the task's
      goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. For a given state (represented by `node.state`), identify the current
       location of each box by examining the `(at ?box ?location)` facts
       where `?box` is a box object.
    2. Initialize the total heuristic value to 0.
    3. Iterate through each box that has a designated goal location (identified
       during initialization from the task goals).
    4. For the current box, retrieve its current location from the state and
       its designated goal location (from initialization).
    5. If the box's current location is the same as its goal location, it is
       already satisfied, and no cost is added for this box.
    6. If the box is not at its goal location:
       a. Look up the precomputed shortest path distance between the box's
          current location and its goal location in the `self.distances` dictionary.
       b. If the box's current location is not a key in the precomputed distances
          or the goal location is unreachable from the current location on the
          static grid graph (distance is infinity), the state is likely
          a dead end or part of an unsolvable problem. Return `float('inf')`
          to indicate a very high cost.
       c. Otherwise (a finite path exists), add this distance to the
          `total_heuristic`. This distance represents a lower bound on the
          number of pushes required for this box, ignoring robot movement
          and dynamic obstacles.
    7. After iterating through all boxes with goals, the `total_heuristic`
       represents the sum of minimum pushes needed for all misplaced boxes,
       based on static grid connectivity. Return this value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph, precomputing
        distances, and extracting box-goal mappings.
        """
        super().__init__(task)

        # Build the grid graph from adjacent facts
        self.grid_graph = collections.defaultdict(list)
        # We only need one direction for distance calculation, so store undirected edges
        # Assuming adjacency is symmetric based on typical grid domains
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.grid_graph[loc1].append(loc2)
                self.grid_graph[loc2].append(loc1)

        # Get all unique locations present in the graph
        all_locations = list(self.grid_graph.keys())

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in all_locations:
            self.distances[start_loc] = self._bfs(start_loc, all_locations)

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Check if the goal is (at ?box ?location) where ?box is a box object
            # We assume box objects start with "box" based on the example instance.
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                self.goal_locations[box] = location

    def _bfs(self, start_node, all_nodes):
        """
        Performs BFS from a start node to find distances to all reachable nodes.
        Returns a dictionary mapping node to distance.
        """
        distances = {node: float('inf') for node in all_nodes}
        distances[start_node] = 0
        queue = collections.deque([start_node])

        # Handle cases where start_node might not be in the graph (e.g., isolated location)
        if start_node not in self.grid_graph:
             # If the start node is not in the graph, it cannot reach any other node
             # except itself (distance 0). The initial distances dictionary already
             # reflects this (all infinity except start_node=0).
             return distances

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in self.grid_graph: # This check is redundant if start_node was in graph
                for neighbor in self.grid_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Find current location of each box
        current_box_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            # Check if the fact is (at ?box ?location) where ?box is a box object
            # We assume box objects start with "box" based on the example instance.
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                current_box_locations[box] = location

        total_heuristic = 0

        # Sum distances for boxes not at their goal
        # Iterate through the boxes we know have goals
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If a box expected to have a goal isn't currently located anywhere,
            # something is wrong with the state representation or problem definition.
            # In a valid Sokoban state, every box should be 'at' a location.
            # We'll assume valid states and proceed. If current_location is None,
            # it implies the box is not 'at' any location in the state facts.
            # This heuristic will effectively ignore such boxes, which might be
            # misleading if the state is malformed. Assuming valid states:
            if current_location is None:
                 # This case indicates an issue with the state representation
                 # or problem file if a box with a goal isn't 'at' a location.
                 # For robustness, we could return infinity or log a warning.
                 # Returning infinity is safer for a heuristic.
                 return float('inf')


            if current_location != goal_location:
                # Get the distance from the precomputed table
                # Check if the current location is a valid start node in our distance table
                if current_location not in self.distances:
                     # This location might be isolated or not part of the main grid graph
                     # from which goals are reachable. Treat as unreachable.
                     return float('inf')

                distance = self.distances[current_location].get(goal_location, float('inf'))

                if distance == float('inf'):
                    # If a box's goal is unreachable from its current location
                    # on the static grid, the problem is likely unsolvable
                    # from this state.
                    return float('inf')

                total_heuristic += distance

        return total_heuristic
