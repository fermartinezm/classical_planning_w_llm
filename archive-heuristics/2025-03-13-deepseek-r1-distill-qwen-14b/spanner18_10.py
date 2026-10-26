from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all nuts by considering the man's current position, the locations of usable spanners, and the positions of loose nuts.

    # Assumptions:
    - The man can carry at most one spanner at a time.
    - Each nut requires exactly one spanner to be tightened.
    - The man must be at the same location as a nut and carry a usable spanner to tighten it.

    # Heuristic Initialization
    - Extract static facts to build a graph of locations and identify initial spanner positions.
    - Precompute shortest paths between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the man and all nuts.
    2. For each loose nut, calculate the shortest path from the man's current location to the nut's location.
    3. If the man isn't carrying a spanner, find the nearest usable spanner and calculate the path to it.
    4. Each nut requires moving to its location, picking up a spanner (if needed), and tightening it, which costs a total of 3 actions per nut.
    5. Sum the actions required for all nuts to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information and building the location graph."""
        super().__init__(task)
        self.task = task

        # Build the location graph from static facts
        self.locations = set()
        self.links = {}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.links:
                    self.links[loc1] = []
                if loc2 not in self.links:
                    self.links[loc2] = []
                self.links[loc1].append(loc2)
                self.links[loc2].append(loc1)

        # Precompute shortest paths between all pairs of locations using BFS
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = {}
            queue = deque()
            queue.append((loc, 0))
            visited = {loc}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.links.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

        # Extract initial spanner positions
        self.spanner_positions = {}
        for fact in task.static:
            if match(fact, "at", "spanner*", "*"):
                spanner, loc = get_parts(fact)[1], get_parts(fact)[2]
                self.spanner_positions[spanner] = loc

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_state = set(state)

        # Extract current positions
        man_pos = None
        for fact in current_state:
            if match(fact, "at", "*", "location*"):
                obj, loc = get_parts(fact)
                if obj == "bob":
                    man_pos = loc
                    break

        # Extract positions of spanners
        spanner_info = {}
        for fact in current_state:
            if match(fact, "at", "spanner*", "*"):
                spanner, loc = get_parts(fact)
                spanner_info[spanner] = loc
            if match(fact, "carrying", "bob", "spanner*"):
                spanner = get_parts(fact)[2]
                spanner_info[spanner] = "carried"

        # Extract positions of nuts
        nuts = []
        for fact in current_state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                for fact2 in current_state:
                    if match(fact2, "at", nut, "*"):
                        loc = get_parts(fact2)[2]
                        nuts.append((nut, loc))
                        break

        # If all nuts are already tightened, return 0
        if not nuts:
            return 0

        total_actions = 0

        # For each nut, calculate the required actions
        for nut, nut_loc in nuts:
            # Check if the nut is already tightened
            if f"(tightened {nut})" in current_state:
                continue

            # Calculate distance from man's current position to nut's location
            if man_pos is None:
                # Man hasn't moved yet, assume he's at the initial position
                # This should be handled by the planner, but in case, find his position
                for fact in current_state:
                    if match(fact, "at", "bob", "*"):
                        man_pos = get_parts(fact)[2]
                        break

            # Get the distance between man's location and nut's location
            distance = self.distances.get(man_pos, {}).get(nut_loc, float('inf'))

            # Check if the man is carrying a spanner
            carrying_spanner = any(match(f, "carrying", "bob", "spanner*") for f in current_state)
            if not carrying_spanner:
                # Find the nearest usable spanner
                nearest_spanner = None
                min_spanner_dist = float('inf')
                for spanner, pos in self.spanner_positions.items():
                    # Check if the spanner is usable
                    if f"(usable {spanner})" in current_state:
                        # Calculate distance to this spanner
                        dist = self.distances.get(man_pos, {}).get(pos, float('inf'))
                        if dist < min_spanner_dist:
                            min_spanner_dist = dist
                            nearest_spanner = pos
                if nearest_spanner is None:
                    # No usable spanner found, which should not happen in a solvable state
                    return float('inf')
                else:
                    # Add the distance to the nearest spanner
                    distance += min_spanner_dist

            # Each nut requires moving to it (distance steps), picking it up (1), and tightening (1)
            # But since the man is already at the nut's location after moving, only 1 action (tighten)
            # However, if moving is required, it's distance steps, plus 1 for tightening
            # But in our case, the man needs to move to the nut's location, which is 'distance' steps
            # Then, tighten the nut, which is 1 action
            total_actions += distance + 1

        return total_actions

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
