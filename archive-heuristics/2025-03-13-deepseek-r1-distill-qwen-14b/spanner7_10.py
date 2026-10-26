from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ?m - man ?l - location)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts by calculating the minimal path for the man to reach each nut, considering the need to pick up a spanner if not already carrying one.

    # Assumptions:
    - The man can carry multiple spanners.
    - The man starts at the shed.
    - Each nut must be at the same location as a usable spanner to be tightened.
    - The graph of locations is unweighted, and the shortest path is used.

    # Heuristic Initialization
    - Extract static facts to build a map of locations and their connections.
    - Precompute the shortest path distances between all pairs of locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the man, spanners, and nuts.
    2. For each loose nut, determine the shortest path from the man's current position to the nut's location.
    3. If the man doesn't have a spanner, find the nearest spanner and calculate the path to pick it up.
    4. For each nut, sum the actions needed to move to its location, pick up a spanner if necessary, and tighten it.
    5. Sum the total actions for all nuts to get the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and precomputing distances.
        """
        self.task = task
        static_facts = task.static

        # Build location graph
        self.locations = set()
        self.graph = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Precompute distances between all pairs of locations using BFS
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = {}
            queue = deque()
            queue.append((loc, 0))
            visited = {loc}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        man_pos = None
        spanner_positions = {}
        nut_positions = {}
        loose_nuts = 0

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj == "?m - man":
                    man_pos = loc
                elif obj.startswith("spanner"):
                    spanner_positions[obj] = loc
                elif obj.startswith("nut"):
                    nut_positions[obj] = loc
            elif parts[0] == "loose":
                loose_nuts += 1

        # If no loose nuts, return 0
        if loose_nuts == 0:
            return 0

        total_actions = 0

        # Check if man has any spanner
        carrying_spanner = any(fact.startswith("carrying ?m - man ?s - spanner") for fact in state)

        # For each loose nut, calculate the required actions
        for nut in nut_positions:
            nut_loc = nut_positions[nut]
            if not carrying_spanner:
                # Find the nearest spanner
                nearest_spanner = None
                min_dist = float('inf')
                for spanner, loc in spanner_positions.items():
                    if loc != nut_loc:  # Don't pick up spanner at nut's location if not needed
                        dist = self.distances[man_pos].get(loc, float('inf'))
                        if dist < min_dist:
                            min_dist = dist
                            nearest_spanner = loc
                if nearest_spanner is None:
                    # No spanner available, which should not happen in solvable problems
                    return float('inf')
                else:
                    # Move to spanner
                    total_actions += self.distances[man_pos][nearest_spanner]
                    # Pick up spanner
                    total_actions += 1
                    carrying_spanner = True  # After pickup, man carries a spanner

            # Move to nut's location
            dist = self.distances[man_pos].get(nut_loc, float('inf'))
            total_actions += dist

            # Tighten the nut
            total_actions += 1

            # Update man's position
            man_pos = nut_loc

        return total_actions
