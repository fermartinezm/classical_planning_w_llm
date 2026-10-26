from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.

    # Assumptions:
    - The man can carry multiple spanners.
    - Each nut requires one action to tighten.
    - The man must be at the nut's location and carry a spanner to tighten it.

    # Heuristic Initialization
    - Extracts the graph of locations from static facts to compute distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify the locations of all loose nuts.
    3. For each loose nut, compute the shortest path distance from the man's current location.
    4. Sum these distances to get the total movement cost.
    5. Count the number of loose nuts to add the tightening actions.
    6. If the man isn't carrying a spanner, find the nearest spanner and add the distance to it plus one action.
    7. The total heuristic is the sum of movement costs, tightening actions, and spanner pickup actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting location information from static facts.
        """
        self.location_graph = {}
        static_facts = task.static

        # Build the location graph from static facts
        for fact in static_facts:
            if not match(fact, "link", "*", "*"):
                continue
            loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
            if loc1 not in self.location_graph:
                self.location_graph[loc1] = []
            self.location_graph[loc1].append(loc2)
            if loc2 not in self.location_graph:
                self.location_graph[loc2] = []
            self.location_graph[loc2].append(loc1)

    def get_distance(self, from_loc, to_loc):
        """
        Compute the shortest path distance between two locations using BFS.
        """
        if from_loc == to_loc:
            return 0
        visited = set()
        queue = deque([(from_loc, 0)])
        while queue:
            current, dist = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.location_graph.get(current, []):
                if neighbor == to_loc:
                    return dist + 1
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float('inf')  # No path found

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Find the man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
            return float('inf')  # Man's location not found

        # Find all loose nuts and their locations
        loose_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                # Find the location of the nut
                nut_location = None
                for loc_fact in state:
                    if match(loc_fact, "at", nut, "*"):
                        nut_location = get_parts(loc_fact)[2]
                        break
                if nut_location is not None:
                    loose_nuts.append(nut_location)

        # If no loose nuts, return 0
        if not loose_nuts:
            return 0

        # Calculate total distance to all nuts
        total_distance = 0
        for nut_loc in loose_nuts:
            distance = self.get_distance(man_location, nut_loc)
            if distance == float('inf'):
                return float('inf')  # No path to a nut, heuristic is infinity
            total_distance += distance

        # Number of nuts to tighten
        num_nuts = len(loose_nuts)

        # Check if the man is carrying any spanner
        has_spanner = any(match(fact, "carrying", "bob", "*") for fact in state)

        # If not carrying a spanner, find the nearest spanner
        if not has_spanner:
            nearest_spanner_distance = float('inf')
            for fact in state:
                if match(fact, "at", "*", "*"):
                    obj, loc = get_parts(fact)
                    if match(obj, "spanner*"):
                        distance = self.get_distance(man_location, loc)
                        if distance < nearest_spanner_distance:
                            nearest_spanner_distance = distance
            if nearest_spanner_distance == float('inf'):
                return float('inf')  # No spanner available
            # Add the distance to the nearest spanner and one action to pick it up
            total_distance += nearest_spanner_distance + 1

        # Total heuristic is the sum of distances, number of nuts, and spanner pickup
        heuristic_value = total_distance + num_nuts

        return heuristic_value
