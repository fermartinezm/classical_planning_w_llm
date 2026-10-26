from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define utility functions outside the class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if parts and args have different lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assume Heuristic base class exists as described in examples
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         pass

# Define the heuristic class
class spannerHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It sums the number of tighten actions, the number of spanner pickup actions
    required, and an estimate of the necessary travel distance for the first
    nut-tightening sequence.

    # Assumptions:
    - The goal is to tighten a specific set of nuts.
    - Each usable spanner can tighten exactly one nut.
    - The man can carry multiple spanners.
    - Travel between linked locations costs 1 action. Pickup and Tighten actions cost 1.
    - The heuristic assumes the man will acquire needed spanners and visit nut locations.
    - There is exactly one man object in the domain.

    # Heuristic Initialization
    - Parses the domain objects to identify locations, men, spanners, and nuts.
    - Builds a graph of locations based on `link` predicates.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the set of nuts that need to be tightened based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Identify the man's current location.
    2.  Identify all loose nuts that are part of the goal and their current locations.
    3.  If there are no loose goal nuts, the heuristic is 0 (goal state reached).
    4.  Count the number of loose goal nuts (`N_loose`). This contributes `N_loose` to the heuristic (for the tighten actions).
    5.  Count the number of usable spanners the man is currently carrying (`N_carried_usable`).
    6.  Count the number of usable spanners available at locations (`N_usable_at_loc`).
    7.  Check if the total number of usable spanners (`N_carried_usable + N_usable_at_loc`) is less than the number of loose nuts (`N_loose`). If so, the problem is unsolvable from this state, return infinity.
    8.  Calculate the number of additional spanners the man needs to pick up from locations (`spanners_to_pickup = max(0, N_loose - N_carried_usable)`). This contributes `spanners_to_pickup` to the heuristic (for the pickup actions).
    9.  Estimate the travel cost:
        *   If `spanners_to_pickup > 0` (the man needs to pick up at least one spanner from a location): The man must travel from his current location (`L_M`) to a location with a usable spanner (`L_S`), pick it up, and then travel from `L_S` to a location with a loose nut (`L_N`) to tighten it. The estimated travel cost is the minimum value of `dist(L_M, L_S) + dist(L_S, L_N)` over all pairs of usable spanners at locations (`L_S`) and loose nuts (`L_N`).
        *   If `spanners_to_pickup == 0` (the man is already carrying enough usable spanners for the remaining nuts): The man can travel directly from his current location (`L_M`) to a location with a loose nut (`L_N`). The estimated travel cost is the minimum value of `dist(L_M, L_N)` over all locations with loose nuts (`L_N`).
    10. Sum the costs: `Heuristic = N_loose + spanners_to_pickup + EstimatedTravelCost`.
    11. Handle cases where required locations are unreachable (distance is infinity) by returning infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        # Store goal conditions to identify goal nuts
        self.goals = task.goals

        # Parse objects to identify types and instances
        self.locations = []
        self.men = []
        self.spanners = []
        self.nuts = []

        # task.objects is a list of strings like 'obj1 obj2 - type'
        obj_strings = task.objects
        current_type = None
        current_objects = []
        for item in obj_strings:
            if item == '-':
                pass # Separator
            elif item in ['location', 'man', 'spanner', 'nut', 'locatable', 'object']:
                # Process previous group
                if current_type == 'location':
                    self.locations.extend(current_objects)
                elif current_type == 'man':
                    self.men.extend(current_objects)
                elif current_type == 'spanner':
                    self.spanners.extend(current_objects)
                elif current_type == 'nut':
                    self.nuts.extend(current_objects)
                # Start new group
                current_type = item
                current_objects = []
            else:
                current_objects.append(item)

        # Process the last group
        if current_type == 'location':
            self.locations.extend(current_objects)
        elif current_type == 'man':
            self.men.extend(current_objects)
        elif current_type == 'spanner':
            self.spanners.extend(current_objects)
        elif current_type == 'nut':
            self.nuts.extend(current_objects)

        # Build adjacency list for the location graph from 'link' facts
        adj = {loc: [] for loc in self.locations}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                if l1 in adj and l2 in adj: # Ensure locations are known
                    adj[l1].append(l2)
                    adj[l2].append(l1) # Assuming links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distance = {}
        for start_node in self.locations:
            self.distance[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            while q:
                curr_node, d = q.popleft()
                self.distance[start_node][curr_node] = d
                for neighbor in adj.get(curr_node, []): # Use .get for safety
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, d + 1))

        # Identify goal nuts from goal conditions
        self.goal_nuts = set()
        # task.goals is a list of facts like '(tightened nut1)'
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'tightened':
                self.goal_nuts.add(parts[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Find man's current location
        man_location = None
        # Assuming there is exactly one man and his name is in self.men[0]
        if not self.men: # Should not happen in valid problems, but safety check
             return float('inf')
        man_name = self.men[0]
        for fact in state:
            if match(fact, "at", man_name, "*"):
                man_location = get_parts(fact)[2]
                break

        # If man's location is unknown, something is wrong, return inf
        if man_location is None:
             return float('inf')

        # Find loose nuts that are goal nuts and their locations
        loose_nuts = {} # {nut_name: location}
        for nut in self.goal_nuts:
            # Check if the nut is loose in the current state
            if f"(loose {nut})" in state:
                # Find location of this loose nut
                for fact in state:
                    if match(fact, "at", nut, "*"):
                        loose_nuts[nut] = get_parts(fact)[2]
                        break
                # If a loose goal nut doesn't have an 'at' fact, something is wrong
                if nut not in loose_nuts:
                     return float('inf')


        N_loose = len(loose_nuts)

        # If all goal nuts are tightened, heuristic is 0
        if N_loose == 0:
            return 0

        # Find usable spanners carried by the man
        carried_usable_spanners = set()
        for spanner in self.spanners:
            if f"(carrying {man_name} {spanner})" in state and f"(usable {spanner})" in state:
                carried_usable_spanners.add(spanner)
        N_carried_usable = len(carried_usable_spanners)

        # Find usable spanners at locations
        usable_spanners_at_loc = {} # {spanner_name: location}
        for spanner in self.spanners:
            # Only consider spanners not currently carried by the man
            if spanner not in carried_usable_spanners:
                 if f"(usable {spanner})" in state:
                    # Find location of this usable spanner
                    for fact in state:
                        if match(fact, "at", spanner, "*"):
                            usable_spanners_at_loc[spanner] = get_parts(fact)[2]
                            break
                    # If a usable spanner at loc doesn't have an 'at' fact, something is wrong
                    if spanner not in usable_spanners_at_loc:
                         return float('inf')


        N_usable_at_loc = len(usable_spanners_at_loc)

        # Check if the problem is unsolvable from this state (not enough usable spanners total)
        if N_loose > N_carried_usable + N_usable_at_loc:
             return float('inf')

        # Calculate heuristic components
        h = N_loose # Cost for tighten actions (one per loose goal nut)

        # Number of spanners the man needs to pick up from locations
        # This is the number of *additional* spanners required beyond those already carried and usable
        spanners_to_pickup = max(0, N_loose - N_carried_usable)

        if spanners_to_pickup > 0:
            h += spanners_to_pickup # Cost for pickup actions

            # Travel cost: Estimate travel for the first spanner pickup and subsequent nut tightening
            # Find min_dist(man_loc -> spanner_loc -> nut_loc) over available spanners at locations and loose nuts
            min_travel = float('inf')
            found_path = False
            # Iterate through usable spanners that are *at locations*
            for s_loc in usable_spanners_at_loc.values():
                # Iterate through loose nuts
                for n_loc in loose_nuts.values():
                     # Check if paths exist in the precomputed distances
                     # A location might not be in self.distance if it's disconnected from the start node used in BFS
                     if man_location in self.distance and s_loc in self.distance[man_location] and n_loc in self.distance[s_loc]:
                         travel = self.distance[man_location][s_loc] + self.distance[s_loc][n_loc]
                         min_travel = min(min_travel, travel)
                         found_path = True

            # If no path exists from man to any usable spanner at loc and then to any loose nut
            if not found_path:
                 return float('inf') # Unsolvable

            h += min_travel

        else: # spanners_to_pickup == 0, man has enough usable spanners carried for the remaining nuts
            # Travel cost: Estimate travel to the first nut location
            # Find min_dist(man_loc -> nut_loc) over loose nuts
            min_travel = float('inf')
            found_path = False
            for n_loc in loose_nuts.values():
                if man_location in self.distance and n_loc in self.distance[man_location]:
                    travel = self.distance[man_location][n_loc]
                    min_travel = min(min_travel, travel)
                    found_path = True

            # If no path exists from man to any loose nut
            if not found_path:
                 return float('inf') # Unsolvable

            h += min_travel

        return h
