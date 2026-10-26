from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    This heuristic estimates the number of actions needed to tighten all loose nuts by the man, considering the need to collect spanners and move between locations.

    # Assumptions:
    - The man starts at a specific location and can move between connected locations.
    - Each nut requires a spanner to be tightened.
    - The man can carry multiple spanners but needs to pick them up individually.
    - The locations form a linear chain, allowing distance calculation based on their order.

    # Heuristic Initialization
    - Extracts goal locations for each nut and the static links between locations.
    - Builds a graph of locations and assigns each a position in a linear sequence.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the man.
    2. Determine the locations of all loose nuts.
    3. Count the number of loose nuts (L) and the number of spanners the man is carrying (S).
    4. Calculate the number of spanners needed: max(0, L - S).
    5. For each required spanner, find its location and calculate the distance from the man's current location.
    6. Sort the required spanners in the order of their positions along the chain.
    7. Calculate the total distance to collect all required spanners.
    8. For each loose nut, calculate the distance from the last spanner's location (or the man's current location if no spanners are needed).
    9. Sort the nuts in the order of their positions along the chain.
    10. Calculate the total distance to visit all nuts.
    11. Sum the total distances and add the number of pickup and tighten actions to get the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each nut.
        - Static facts (links between locations).
        - Build a graph of locations and assign each a position in a linear sequence.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build location graph and assign positions
        self.location_graph = {}
        self.location_positions = {}
        # Extract links
        links = []
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                links.append((l1, l2))
                if l1 not in self.location_graph:
                    self.location_graph[l1] = []
                self.location_graph[l1].append(l2)
                if l2 not in self.location_graph:
                    self.location_graph[l2] = []
                self.location_graph[l2].append(l1)
        
        # Assign positions using BFS from the first location
        # Find all locations
        locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                locations.add(l1)
                locations.add(l2)
        # Sort locations to have a consistent order
        locations = sorted(locations)
        # Find the start location (assumed to be the first in the instance's initial state)
        # For the spanner domain, the man starts at 'shed' typically
        # But to generalize, find the location that is a starting point in the links
        # Alternatively, assume the first location in the sorted list is the start
        start_location = locations[0] if locations else None
        if not start_location:
            self.location_positions = {}
            return
        # BFS to assign positions
        from collections import deque
        visited = {start_location}
        queue = deque([(start_location, 0)])
        while queue:
            loc, pos = queue.popleft()
            self.location_positions[loc] = pos
            for neighbor in self.location_graph.get(loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, pos + 1))
        # If some locations are not reachable, assign them a high position
        for loc in locations:
            if loc not in self.location_positions:
                self.location_positions[loc] = float('inf')

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Extract man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                _, loc = get_parts(fact)
                man_location = loc
                break
        if not man_location:
            return 0  # Should not happen in valid state

        # Extract carried spanners
        carried_spanners = []
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                _, s = get_parts(fact)
                carried_spanners.append(s)
        
        # Extract loose nuts and their locations
        loose_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                # Find the location of the nut
                nut_location = None
                for fact2 in state:
                    if match(fact2, "at", nut, "*"):
                        nut_location = get_parts(fact2)[2]
                        break
                if nut_location:
                    loose_nuts.append(nut_location)
        
        L = len(loose_nuts)
        S = len(carried_spanners)
        needed_spanners = max(0, L - S)

        if needed_spanners == 0:
            spanner_locations = []
        else:
            # Collect all available spanner locations not carried
            all_spanner_locations = []
            for fact in state:
                if match(fact, "at", "*", "location*") and match(fact, "spanner", "*"):
                    s, loc = get_parts(fact)
                    if s not in carried_spanners:
                        all_spanner_locations.append(loc)
            # Select the needed_spanners closest to man_location
            all_spanner_locations.sort(key=lambda x: abs(self.location_positions.get(x, float('inf')) - self.location_positions.get(man_location, 0)))
            spanner_locations = all_spanner_locations[:needed_spanners]

        # Calculate distance for collecting spanners
        total_distance = 0
        current_loc = man_location
        if spanner_locations:
            # Sort spanner_locations in the order of their positions
            spanner_locations.sort(key=lambda x: self.location_positions[x])
            for loc in spanner_locations:
                total_distance += abs(self.location_positions[loc] - self.location_positions[current_loc])
                current_loc = loc

        # Calculate distance for tightening nuts
        if loose_nuts:
            # Sort nuts by their distance from current_loc (last spanner location or man_location)
            loose_nuts.sort(key=lambda x: abs(self.location_positions.get(x, float('inf')) - self.location_positions.get(current_loc, 0)))
            for nut_loc in loose_nuts:
                total_distance += abs(self.location_positions[nut_loc] - self.location_positions[current_loc])
                current_loc = nut_loc

        # Number of actions: needed_spanners (pickup) + L (tighten)
        total_actions = needed_spanners + L
        heuristic_value = total_distance + total_actions

        return heuristic_value
