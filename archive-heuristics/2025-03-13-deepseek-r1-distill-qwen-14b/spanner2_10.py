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
    This heuristic estimates the number of actions needed to tighten all loose nuts by collecting spanners and moving them to the nuts' locations.

    # Assumptions:
    - The man starts at his current location and needs to collect spanners to tighten each nut.
    - Each nut requires a separate spanner, which must be collected before tightening it.
    - The minimal path is assumed by selecting the closest available spanner for each nut.

    # Heuristic Initialization
    - Extract static facts to build a graph of locations and precompute shortest paths between all pairs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the man's current location from the state.
    2. For each loose nut:
       a. Identify all usable spanners in the current state.
       b. For each usable spanner, calculate the minimal distance from the man's current location to the spanner's location.
       c. Select the spanner with the smallest distance.
       d. Add the distance to the total cost.
       e. Add 1 action for picking up the spanner.
       f. Calculate the minimal distance from the spanner's location to the nut's location.
       g. Add this distance to the total cost.
       h. Add 1 action for tightening the nut.
       i. Update the man's current location to the nut's location for the next iteration.
    3. Sum all the actions for each nut to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and building the location graph."""
        super().__init__(task)
        self.task = task

        # Extract all locations from static facts
        self.locations = set()
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Build the location graph
        self.graph = {loc: [] for loc in self.locations}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Precompute all pairs shortest paths
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = {}
            queue = deque()
            queue.append((loc, 0))
            visited = {loc}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_cost = 0
        man_location = self._get_man_location(state)
        if not man_location:
            return 0  # No man in state, which is impossible

        loose_nuts = [fact for fact in state if match(fact, "loose", "*")]
        if not loose_nuts:
            return 0  # All nuts are already tightened

        for nut_fact in loose_nuts:
            nut = get_parts(nut_fact)[1]
            nut_location = self._get_nut_location(nut, state)

            # Find all usable spanners
            usable_spanners = [fact for fact in state if match(fact, "usable", "*")]
            if not usable_spanners:
                # No spanners available, cannot proceed
                return float('inf')

            min_distance = float('inf')
            selected_spanner_location = None
            for spanner_fact in usable_spanners:
                spanner = get_parts(spanner_fact)[1]
                spanner_location = self._get_spanner_location(spanner, state)
                # Calculate distance from current man location to spanner location
                if man_location in self.distances and spanner_location in self.distances[man_location]:
                    distance = self.distances[man_location][spanner_location]
                    if distance < min_distance:
                        min_distance = distance
                        selected_spanner_location = spanner_location

            if selected_spanner_location is None:
                # No path found, which is impossible given the static facts
                return float('inf')

            # Add the distance to get to the spanner
            current_cost += min_distance
            # Add pickup action
            current_cost += 1
            # Calculate distance from spanner to nut
            if selected_spanner_location in self.distances and nut_location in self.distances[selected_spanner_location]:
                spanner_to_nut_distance = self.distances[selected_spanner_location][nut_location]
            else:
                # No path found, which is impossible
                return float('inf')
            current_cost += spanner_to_nut_distance
            # Add tighten action
            current_cost += 1

            # Update man's location to the nut's location for the next iteration
            man_location = nut_location

        return current_cost

    def _get_man_location(self, state):
        """Extract the man's current location from the state."""
        for fact in state:
            if match(fact, "at", "*", "man"):
                return get_parts(fact)[2]
        return None  # Man not found, which is impossible

    def _get_nut_location(self, nut, state):
        """Extract the current location of a specific nut."""
        for fact in state:
            if match(fact, "at", nut, "*"):
                return get_parts(fact)[2]
        return None  # Nut not found, which is impossible

    def _get_spanner_location(self, spanner, state):
        """Extract the current location of a specific spanner."""
        for fact in state:
            if match(fact, "at", spanner, "*"):
                return get_parts(fact)[2]
        return None  # Spanner not found, which is impossible
