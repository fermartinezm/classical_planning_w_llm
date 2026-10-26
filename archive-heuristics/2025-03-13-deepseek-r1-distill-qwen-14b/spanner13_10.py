from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

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

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed for the man to tighten all loose nuts. Each nut requires the man to be in its location while carrying a usable spanner. If the man doesn't have a spanner, he must retrieve one and then proceed to the nut's location.

    # Assumptions:
    - The man can carry only one spanner at a time.
    - Each nut requires a separate spanner.
    - The static links between locations form a connected graph.
    - The minimal path between any two locations is precomputed.

    # Heuristic Initialization
    - Precompute the shortest paths between all pairs of locations using the static link information.
    - Extract the initial locations of usable spanners and loose nuts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each loose nut, determine if the man is already at its location and carrying a spanner.
    2. If not, calculate the minimal path for the man to retrieve a spanner and then move to the nut's location.
    3. Sum the actions required for all nuts, considering the need to retrieve a new spanner for each.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information and precomputing shortest paths."""
        super().__init__(task)

        # Build the location graph from static links
        self.locations = set()
        self.graph = {}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                self.graph[loc1].append(loc2)
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                self.graph[loc2].append(loc1)

        # Precompute shortest paths between all pairs of locations
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = {}
            queue = deque()
            queue.append((loc, 0))
            visited = {loc: True}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = True
                        self.distances[loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Extract man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_location = get_parts(fact)[2]
                break

        if not man_location:
            return 0  # Shouldn't happen; man is always present

        # Extract locations of usable spanners
        usable_spanners = []
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                # Find the location of the spanner
                for fact_loc in state:
                    if match(fact_loc, "at", spanner, "*"):
                        loc = get_parts(fact_loc)[2]
                        usable_spanners.append(loc)
                        break

        # Extract locations of loose nuts
        loose_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                # Find the location of the nut
                for fact_loc in state:
                    if match(fact_loc, "at", nut, "*"):
                        loc = get_parts(fact_loc)[2]
                        loose_nuts.append(loc)
                        break

        total_actions = 0

        # For each loose nut, calculate the required actions
        for nut_loc in loose_nuts:
            # Check if the man is at the nut's location and carrying a spanner
            at_nut_location = (man_location == nut_loc)
            carrying_spanner = False
            for fact in state:
                if match(fact, "carrying", "bob", "*"):
                    carrying_spanner = True
                    break

            if at_nut_location and carrying_spanner:
                total_actions += 1
                continue

            # Need to retrieve a spanner and go to nut's location
            # Find the closest usable spanner location to the man's current location
            min_distance = float('inf')
            closest_spanner = None
            for spanner_loc in usable_spanners:
                if spanner_loc in self.distances[man_location]:
                    distance = self.distances[man_location][spanner_loc]
                    if distance < min_distance:
                        min_distance = distance
                        closest_spanner = spanner_loc

            if closest_spanner is None:
                # No usable spanner available; this shouldn't happen in solvable problems
                continue

            # Calculate the distance from the closest spanner to the nut's location
            if nut_loc in self.distances[closest_spanner]:
                d2 = self.distances[closest_spanner][nut_loc]
            else:
                d2 = 0  # Same location

            # Add the actions: walk to spanner (d1), pickup (1), walk to nut (d2), tighten (1)
            total_actions += (min_distance + 1 + d2 + 1)

        return total_actions
