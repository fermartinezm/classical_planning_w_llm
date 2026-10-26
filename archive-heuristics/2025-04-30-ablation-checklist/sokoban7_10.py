import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    This heuristic estimates the number of actions needed to reach a goal state
    by summing the shortest path distances for each box from its current location
    to its goal location. The distance is calculated on the static grid graph,
    ignoring the robot's position and other boxes as obstacles.

    # Assumptions
    - The grid structure (adjacencies) is static.
    - Each box has a unique goal location.
    - The primary cost is moving boxes towards their goals.
    - Robot movement cost and the need to clear paths are ignored for simplicity
      and efficiency, making the heuristic non-admissible but potentially
      informative for greedy search.

    # Heuristic Initialization
    - Parses the static facts to build the grid graph (adjacency list).
    - Parses the goal conditions to map each box to its specific goal location.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1.  Build the grid graph: From the static `adjacent` facts, create an
        adjacency list representation where nodes are locations and edges
        represent direct connections.
    2.  Map boxes to goals: From the goal conditions, create a dictionary
        mapping each box object to its target location.
    3.  For a given state:
        a.  Find the current location of each box by inspecting the state facts.
        b.  Initialize the total heuristic cost to 0.
        c.  For each box that is not yet at its goal location:
            i.  Calculate the shortest path distance between the box's current
                location and its goal location using BFS on the pre-built grid graph.
                This distance represents the minimum number of 'push' steps
                required for that box if there were no dynamic obstacles.
            ii. Add this distance to the total heuristic cost.
        d.  Return the total heuristic cost.

    This heuristic is non-admissible because it ignores the cost of moving the
    robot to a position where it can push the box, and it ignores the fact
    that other boxes or the robot itself might block the shortest path for a box.
    However, it provides a reasonable estimate of the minimum "box movement"
    effort required.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Static facts (`adjacent` relationships).
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the grid graph from adjacent facts
        self.graph = collections.defaultdict(set)
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Assuming adjacency is symmetric

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                self.goal_locations[box] = location

        # Pre-calculate all-pairs shortest paths if the grid is small enough,
        # or just use BFS on demand. Given the problem scale might vary,
        # on-demand BFS is safer and often efficient enough for heuristics.
        # We will use on-demand BFS.

    def _bfs_distance(self, start_loc, end_loc):
        """
        Calculates the shortest path distance between two locations using BFS
        on the pre-built grid graph. Returns float('inf') if no path exists.
        """
        if start_loc == end_loc:
            return 0

        queue = collections.deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, distance = queue.popleft()

            if current_loc == end_loc:
                return distance

            for neighbor in self.graph.get(current_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf') # No path found

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        Sum of shortest path distances for each box to its goal.
        """
        state = node.state  # Current world state.

        # Find the current location of each box
        current_box_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                current_box_locations[box] = location

        total_distance = 0  # Initialize heuristic cost

        # Sum distances for all boxes not at their goal
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box) # Use .get() in case a box isn't in the state (shouldn't happen in valid states, but good practice)

            if current_location is None:
                 # This box is not in the state, or not 'at' a location.
                 # This shouldn't happen in a valid Sokoban state representation
                 # where all boxes are always at some location.
                 # Treat as infinite cost or a large penalty if it could happen.
                 # For standard Sokoban, we assume boxes are always 'at' a location.
                 continue # Skip this box or handle error

            if current_location != goal_location:
                distance = self._bfs_distance(current_location, goal_location)
                # If a box is unreachable from its goal, the state is likely a dead end.
                # Summing infinity correctly reflects this high cost.
                total_distance += distance

        return total_distance

