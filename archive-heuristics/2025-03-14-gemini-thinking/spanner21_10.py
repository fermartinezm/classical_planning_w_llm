from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

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

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the minimum number of actions required to tighten all loose nuts.
    It considers the actions needed to pick up a usable spanner, navigate to the nut's location,
    and tighten the nut for each nut that is currently loose.

    # Assumptions:
    - For each loose nut, we need to perform a 'tighten_nut' action.
    - To perform 'tighten_nut', we might need to 'pickup_spanner' and 'walk' actions.
    - We assume there is always a usable spanner available and locations are reachable if linked.
    - We estimate the cost based on the number of actions, not considering action costs if they were different.

    # Heuristic Initialization
    - Extracts the goal predicates (tightened nuts) and static facts (links between locations) from the task.
    - Preprocesses the static link information to facilitate shortest path calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all nuts that are required to be tightened based on the goal description.
    2. For each nut that is currently 'loose' in the given state:
        a. Increment the heuristic cost by 1, assuming a 'tighten_nut' action is needed.
        b. Check if the man is at the same location as the nut. If not, calculate the shortest path (number of 'walk' actions) from the man's current location to the nut's location using the 'link' predicates. Add the path length to the heuristic cost.
        c. Check if the man is carrying a usable spanner. If not, increment the heuristic cost by 1, assuming a 'pickup_spanner' action is needed.
    3. The total accumulated cost is the heuristic estimate for the given state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

        self.links = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.links[l1].append(l2)
                self.links[l2].append(l1) # links are bidirectional in provided examples

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_cost = 0

        loose_nuts_in_state = set()
        for fact in state:
            if match(fact, "loose", "*"):
                nut_name = get_parts(fact)[1]
                if nut_name in self.goal_nuts: # Only consider nuts that are in the goal
                    loose_nuts_in_state.add(nut_name)

        if not loose_nuts_in_state:
            return 0 # Goal state reached or no relevant loose nuts

        man_location = None
        carried_spanner = None
        has_usable_spanner = False

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] == 'bob': # Assuming 'bob' is the man's name
                    man_location = parts[2]
            elif match(fact, "carrying", "bob", "*"):
                carried_spanner = get_parts(fact)[2]
            elif match(fact, "usable", "*"):
                if get_parts(fact)[1] == carried_spanner:
                    has_usable_spanner = True

        for nut_name in loose_nuts_in_state:
            nut_location = None
            for fact in state:
                if match(fact, "at", "*", "*"):
                    parts = get_parts(fact)
                    if parts[1] == nut_name:
                        nut_location = parts[2]
                        break

            if nut_location is None:
                continue # Nut location not found, should not happen in valid problems

            heuristic_cost += 1 # For tighten_nut action

            if man_location != nut_location:
                path_length = self._get_shortest_path_length(man_location, nut_location)
                if path_length > 0:
                    heuristic_cost += path_length

            if not has_usable_spanner:
                heuristic_cost += 1 # For pickup_spanner action

        return heuristic_cost

    def _get_shortest_path_length(self, start_location, end_location):
        """Calculates the shortest path length between two locations using BFS."""
        if start_location == end_location:
            return 0

        queue = collections.deque([(start_location, 0)]) # (location, distance)
        visited = {start_location}

        while queue:
            current_location, distance = queue.popleft()

            for neighbor in self.links[current_location]:
                if neighbor not in visited:
                    if neighbor == end_location:
                        return distance + 1
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return float('inf') # No path found, should not happen in valid problems for this domain.

