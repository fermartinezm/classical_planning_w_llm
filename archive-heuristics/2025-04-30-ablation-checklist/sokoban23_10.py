import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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

def bfs(graph, start_node):
    """
    Performs a Breadth-First Search to find shortest distances from a start node
    to all reachable nodes in a graph.

    Args:
        graph: A dictionary representing the graph, where keys are nodes
               and values are lists of adjacent nodes.
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its shortest distance
        from the start_node. Returns float('inf') for unreachable nodes.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Handle cases where the start_node might not be in the graph (e.g., isolated location)
         # Though in Sokoban, all locations are usually connected in the grid.
         return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        u = queue.popleft()
        for v in graph.get(u, []):
            if distances[v] == float('inf'):
                distances[v] = distances[u] + 1
                queue.append(v)

    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the
    shortest path distances for each box from its current location to its
    goal location. It ignores the robot's position and other boxes as
    obstacles during the distance calculation, providing a lower bound
    on the number of *push* actions required for each box individually.
    It does not consider the robot's movement cost to get into position
    to push a box, nor does it consider potential deadlocks.

    # Assumptions
    - The goal is defined by the final locations of the boxes.
    - The grid structure and connectivity are defined by `adjacent` predicates.
    - Each box has a unique goal location.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task goals.
    - Builds a graph representing the grid connectivity from `adjacent` facts.
    - Pre-computes all-pairs shortest paths on this grid graph using BFS.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify the goal location for each box from the task definition.
    2. Build a graph representing the connections between locations based on the `adjacent` predicates.
    3. Pre-compute the shortest path distance between every pair of locations in the graph using BFS. This is done once during initialization.
    4. For a given state:
       - Find the current location of each box.
       - For each box that is not yet at its goal location:
         - Look up the pre-computed shortest path distance from the box's current location to its goal location.
         - Add this distance to a running total.
    5. The total sum of these distances is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph.
        """
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the graph from adjacent facts.
        # The graph maps a location to a list of locations adjacent to it.
        self.graph = collections.defaultdict(list)
        all_locations = set()
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "adjacent":
                loc1, loc2, direction = args
                self.graph[loc1].append(loc2)
                # Assuming adjacency is symmetric, add the reverse direction too
                # (though PDDL often lists both explicitly, being safe)
                # Check if the reverse fact exists, if not, add it implicitly
                reverse_dir = None
                if direction == 'up': reverse_dir = 'down'
                elif direction == 'down': reverse_dir = 'up'
                elif direction == 'left': reverse_dir = 'right'
                elif direction == 'right': reverse_dir = 'left'

                if reverse_dir:
                     # Check if the reverse fact is explicitly in static_facts
                     reverse_fact_str = f"(adjacent {loc2} {loc1} {reverse_dir})"
                     if reverse_fact_str not in static_facts:
                          self.graph[loc2].append(loc1)

                all_locations.add(loc1)
                all_locations.add(loc2)

        # Ensure all locations from the problem are in the graph keys, even if isolated
        # (though unlikely in Sokoban grids)
        for fact in task.initial_state:
             predicate, *args = get_parts(fact)
             if predicate in ["at-robot", "at"]:
                  all_locations.add(args[-1]) # The last argument is the location

        for loc in all_locations:
             if loc not in self.graph:
                  self.graph[loc] = [] # Add isolated locations to graph keys

        # Pre-compute all-pairs shortest paths using BFS.
        self.all_pairs_distances = {}
        for start_loc in self.graph:
            self.all_pairs_distances[start_loc] = bfs(self.graph, start_loc)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The current state node.

        Returns:
            An estimated cost (integer) to reach the goal.
        """
        state = node.state

        # Find the current location of each box.
        current_box_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2 and args[0] in self.goal_locations:
                 box, location = args
                 current_box_locations[box] = location

        total_distance = 0

        # Sum the shortest path distances for each misplaced box.
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box) # Use .get() in case a box isn't found (shouldn't happen in valid states)

            if current_location and current_location != goal_location:
                # Look up the pre-computed distance
                distance = self.all_pairs_distances.get(current_location, {}).get(goal_location, float('inf'))

                # If a box cannot reach its goal, return infinity (or a very large number)
                # as this state is likely unsolvable or very bad.
                if distance == float('inf'):
                    return float('inf') # Or a large constant like 1000000

                total_distance += distance

        return total_distance

