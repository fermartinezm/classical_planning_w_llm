from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import networkx as nx

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(road l1 l2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move all packages to their goal locations.
    It calculates the shortest path (number of roads) for each package from its current location to its goal location and sums up the estimated costs (pick-up, drive, drop) for all packages.

    # Assumptions
    - There is always a vehicle available at the package's current location when needed.
    - The heuristic assumes that for each package, we need to perform a pick-up action, a drop action, and drive actions corresponding to the shortest path.
    - Vehicle capacity and capacity-predecessor predicates are implicitly considered by assuming pick-up and drop actions are always possible when needed for path calculation.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Builds a road network graph from the static facts using 'road' predicates to calculate shortest paths.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Initialize the total heuristic cost to 0.
    2. Extract the current location of each package from the current state.
    3. For each package:
        a. Get the current location of the package.
        b. Get the goal location of the package.
        c. If the current location is the same as the goal location, the cost for this package is 0.
        d. Otherwise, calculate the shortest path length (number of roads) from the current location to the goal location in the road network graph. Use Breadth-First Search (BFS) for efficiency.
        e. If a path exists, estimate the cost for this package as (shortest path length + 2) actions (1 for pick-up, shortest path length for drive actions, and 1 for drop).
        f. Add the estimated cost for the package to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the transportHeuristic by extracting goal conditions and static facts.
        Builds the road network graph and stores goal locations for each package.
        """
        self.goals = task.goals
        static_facts = task.static

        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == 'at' and args[0] in task.type_to_objects['package']:
                package, location = args
                self.package_goals[package] = location

        self.road_graph = nx.Graph()
        locations = set()
        for fact in static_facts:
            if match(fact, 'road', '*', '*'):
                _, l1, l2 = get_parts(fact)
                self.road_graph.add_edge(l1, l2)
                locations.add(l1)
                locations.add(l2)
        self.road_graph.add_nodes_from(locations)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        Estimates the number of actions to reach the goal state.
        """
        state = node.state
        package_locations = {}
        for fact in state:
            if match(fact, 'at', '*', '*'):
                parts = get_parts(fact)
                if parts[1] in node.task.type_to_objects['package']:
                    package_locations[parts[1]] = parts[2]

        heuristic_cost = 0
        for package, goal_location in self.package_goals.items():
            current_location = package_locations.get(package)
            if current_location is None: # Package might be in vehicle, assume worst case location for heuristic
                continue # Or handle differently if needed, e.g., find vehicle location and use that. For now, skip if not 'at' location.

            if current_location == goal_location:
                continue # Package already at goal

            try:
                path_length = nx.shortest_path_length(self.road_graph, source=current_location, target=goal_location)
                heuristic_cost += path_length + 2 # pick-up, path_length drives, drop
            except nx.NetworkXNoPath:
                return float('inf') # No path, unsolvable from this state, or very high cost.

        return heuristic_cost
