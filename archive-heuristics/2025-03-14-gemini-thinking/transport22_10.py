from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import itertools

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
    This heuristic estimates the number of actions required to transport all packages to their goal locations.
    It calculates the shortest path between the current location of each package and its goal location,
    and sums up the estimated costs for each package, including pick-up and drop actions.

    # Assumptions:
    - For each package that is not at its goal location, it assumes that it needs to be picked up, transported, and dropped.
    - The cost is based on the shortest path in terms of road connections between locations.
    - It does not consider vehicle capacity or the need to move vehicles to pick up packages explicitly in the heuristic calculation,
      but rather focuses on the package movements themselves.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Precomputes the shortest path distances between all pairs of locations based on the static `road` predicates using the Floyd-Warshall algorithm.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract goal locations for each package from the task's goal conditions.
    2. Extract all locations from the static facts to build the graph for shortest path calculation.
    3. Initialize a distance matrix with infinity for all pairs of locations and 0 for distance to itself.
    4. Populate the distance matrix with 1 for locations connected by a `road` predicate (in either direction).
    5. Apply the Floyd-Warshall algorithm to compute all-pairs shortest paths.
    6. For a given state, iterate through each package.
    7. Determine the current location of the package from the state.
    8. Determine the goal location of the package from the pre-extracted goal locations.
    9. If the current location is not the goal location:
        a. Retrieve the shortest path distance between the current location and the goal location from the precomputed distance matrix.
        b. Add this distance to the heuristic value.
        c. Add 2 to the heuristic value to account for one pick-up and one drop action.
    10. If the current location is the goal location, add 0 to the heuristic value for this package.
    11. The total heuristic value is the sum of the estimated costs for all packages.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing shortest paths and extracting goal information."""
        self.goals = task.goals
        static_facts = task.static

        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                obj = parts[1]
                location = parts[2]
                is_package = False
                for obj_type_def in task.facts: # Check object types to identify packages
                    if match(obj_type_def, "package", obj):
                        is_package = True
                        break
                if is_package:
                    self.package_goals[obj] = location

        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                locations.add(parts[1])
                locations.add(parts[2])
            elif match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                locations.add(parts[2])

        location_list = list(locations)
        location_indices = {loc: i for i, loc in enumerate(location_list)}
        num_locations = len(location_list)

        dist_matrix = [[float('inf')] * num_locations for _ in range(num_locations)]
        for i in range(num_locations):
            dist_matrix[i][i] = 0

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                l1 = parts[1]
                l2 = parts[2]
                if l1 in location_indices and l2 in location_indices: # Ensure locations are in our list
                    u = location_indices[l1]
                    v = location_indices[l2]
                    dist_matrix[u][v] = 1
                    dist_matrix[v][u] = 1 # Roads are bidirectional in examples

        for k in range(num_locations):
            for i in range(num_locations):
                for j in range(num_locations):
                    if dist_matrix[i][k] != float('inf') and dist_matrix[k][j] != float('inf'):
                        dist_matrix[i][j] = min(dist_matrix[i][j], dist_matrix[i][k] + dist_matrix[k][j])
        self.dist_matrix = dist_matrix
        self.location_list = location_list
        self.location_indices = location_indices


    def __call__(self, node):
        """Calculate the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0

        package_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                location = parts[2]
                is_package = False
                for obj_type_def in node.task.facts: # Check object types to identify packages
                    if match(obj_type_def, "package", obj):
                        is_package = True
                        break
                if is_package:
                    package_locations[obj] = location

        for package, goal_location in self.package_goals.items():
            current_location = package_locations.get(package)
            if current_location and current_location != goal_location:
                if current_location in self.location_indices and goal_location in self.location_indices:
                    start_index = self.location_indices[current_location]
                    goal_index = self.location_indices[goal_location]
                    distance = self.dist_matrix[start_index][goal_index]
                    if distance != float('inf'):
                        heuristic_value += distance + 2 # drive actions + pick-up + drop
                    else:
                        heuristic_value += 1000 # Assign a large cost if no path exists, to discourage this path

        return heuristic_value
