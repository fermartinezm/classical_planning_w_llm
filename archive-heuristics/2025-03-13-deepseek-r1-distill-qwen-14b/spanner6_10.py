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
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    The man must move to each nut's location, carry a spanner, and perform the tightening action.

    # Assumptions:
    - The man starts without carrying any spanner.
    - Each nut must be at its designated location to be tightened.
    - The man can carry only one spanner at a time.
    - Moving between adjacent locations takes one action per link.

    # Heuristic Initialization
    - Extract the goal locations for each nut and the static facts (links between locations).

    # Step-by-Step Thinking for Computing Heuristic Value
    1. For each loose nut:
        a. If the man doesn't have a spanner, calculate the actions needed to pick one up.
        b. Calculate the shortest path from the man's current location to the nut's location.
        c. Add the actions needed to move to the nut's location.
        d. Add the action to tighten the nut.
    2. Sum the actions for all loose nuts to get the total estimated actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each nut.
        - Static facts (links between locations).
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts (links between locations)

        # Extract location links and build a graph
        self.location_links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                if loc1 not in self.location_links:
                    self.location_links[loc1] = []
                self.location_links[loc1].append(loc2)
                if loc2 not in self.location_links:
                    self.location_links[loc2] = []
                self.location_links[loc2].append(loc1)

        # Store goal locations for each nut
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut = args[0]
                self.goal_locations[nut] = "goal"  # Simplified, assuming all goals are same type

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state

        # Track relevant information about the man and nuts
        man_location = None
        carrying_spanner = False
        loose_nuts = []

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[2] == "man":
                man_location = parts[1]
            elif parts[0] == "carrying" and parts[2] == "spanner":
                carrying_spanner = True
            elif parts[0] == "loose":
                loose_nuts.append(parts[1])

        # If all nuts are already tightened, return 0
        if not loose_nuts:
            return 0

        total_actions = 0

        # For each loose nut, calculate the required actions
        for nut in loose_nuts:
            # Determine the nut's current location
            nut_location = None
            for fact in state:
                if fact.startswith("(at nut") and nut in fact:
                    nut_location = get_parts(fact)[2]
                    break

            # If the man doesn't have a spanner, find one
            if not carrying_spanner:
                # Find the nearest spanner
                nearest_spanner = None
                min_distance = float('inf')
                for fact in state:
                    if fact.startswith("(at spanner") and get_parts(fact)[1] != man_location:
                        spanner_loc = get_parts(fact)[1]
                        # Calculate distance from man's location to spanner's location
                        visited = {man_location}
                        queue = deque([(man_location, 0)])
                        while queue:
                            current, dist = queue.popleft()
                            if current == spanner_loc:
                                if dist < min_distance:
                                    min_distance = dist
                                    nearest_spanner = spanner_loc
                                break
                            for neighbor in self.location_links.get(current, []):
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    queue.append((neighbor, dist + 1))
                # Add actions to pick up the spanner
                if nearest_spanner:
                    total_actions += min_distance  # Move to spanner
                    total_actions += 1  # Pick up spanner
                else:
                    # No spanner available (shouldn't happen in solvable instances)
                    return float('inf')

            # Calculate distance from man's current location to nut's location
            if man_location != nut_location:
                visited = {man_location}
                queue = deque([(man_location, 0)])
                found = False
                while queue:
                    current, dist = queue.popleft()
                    if current == nut_location:
                        total_actions += dist  # Move to nut's location
                        found = True
                        break
                    for neighbor in self.location_links.get(current, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))
                if not found:
                    # Nut is unreachable (shouldn't happen in solvable instances)
                    return float('inf')

            # Add action to tighten the nut
            total_actions += 1

        return total_actions
