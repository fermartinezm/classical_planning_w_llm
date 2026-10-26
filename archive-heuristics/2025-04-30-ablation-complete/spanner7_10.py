from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts specified in the goal. It considers the number of nuts
    remaining, the cost to acquire a usable spanner if one is not currently
    carried, and the cost to reach the nearest location with a loose nut.

    # Assumptions:
    - Each nut requires a separate usable spanner to be tightened (spanners become unusable).
    - The man can carry one or more spanners, but only usable ones are helpful for tightening.
    - The graph of locations connected by 'link' predicates is connected for all relevant locations (man start, spanner locations, nut locations).

    # Heuristic Initialization
    - Identify the man object.
    - Identify the nuts that need to be tightened (those tightened in the goal).
    - Build the location graph from 'link' facts.
    - Precompute shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Determine if the man is currently carrying at least one usable spanner.
    3. Identify all nuts that are currently loose and are part of the goal.
    4. If no such nuts exist, the heuristic is 0 (goal reached for nuts).
    5. Calculate the base cost: This is 2 actions per remaining loose goal nut (1 pickup + 1 tighten), representing the minimum actions per nut assuming a new spanner is needed and picked up for each.
    6. Calculate the cost to acquire a usable spanner *if the man is not currently carrying any usable spanner*:
       - If the man is carrying at least one usable spanner, this cost is 0.
       - If not, find the nearest usable spanner (not currently carried) and calculate the distance to its location plus 1 for the pickup action. If no usable spanners exist anywhere, the problem is likely unsolvable from this state, return a large value (infinity).
    7. Calculate the cost to reach the nearest loose goal nut: Find the minimum distance from the man's current location to any location containing a loose goal nut.
    8. The total heuristic is the sum of the base cost (2 * num_loose_goal_nuts), the spanner acquisition cost (if needed for the first nut), and the cost to reach the nearest loose goal nut.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals
        self.static_facts = task.static

        # Identify the man object (assuming there's only one man)
        self.man = None
        # Try finding the object involved in a 'carrying' predicate (only man can carry)
        for fact in task.initial_state | task.goals: # Check both initial and goal states for 'carrying'
             if match(fact, "carrying", "*", "*"):
                 self.man = get_parts(fact)[1]
                 break

        if self.man is None: # If not found via carrying, try the first object in an 'at' fact in initial state
             for fact in task.initial_state:
                 if match(fact, "at", "*", "*"):
                     # Assuming the first object in an 'at' fact is the man. This is a common heuristic trick
                     # when object types are not explicitly available from the fact strings alone.
                     self.man = get_parts(fact)[1]
                     break

        if self.man is None:
             # Should not happen in valid problems, but handle defensively
             print("Warning: Could not identify the man object.")
             # The heuristic will likely fail or return inf later if man is None.

        # Identify nuts that are goals (must be tightened)
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Build location graph and compute distances
        self.locations = set()
        self.location_graph = {} # Adjacency list {loc: {neighbor1, neighbor2}}

        # Extract locations and links from static facts
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Add locations from initial state and goals if not already included
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 self.locations.add(get_parts(fact)[2])
        for goal in self.goals:
             if match(goal, "at", "*", "*"): # Goals might include object locations
                 self.locations.add(get_parts(goal)[2])

        # Ensure all locations found are in the graph keys (even if they have no links)
        for loc in self.locations:
             self.location_graph.setdefault(loc, set())

        self.distances = {} # distances[loc1][loc2] = shortest_distance

        # Compute all-pairs shortest paths using BFS
        for start_loc in self.locations:
            self.distances[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, dist = queue.popleft()
                self.distances[start_loc][current_loc] = dist

                for neighbor in self.location_graph.get(current_loc, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returns infinity if no path."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # Man location not found in state, indicates an invalid state representation
             return float('inf')

        # 2. Determine if man is carrying at least one usable spanner
        carrying_usable_spanner = False
        carried_spanners = set()
        for fact in state:
            if match(fact, "carrying", self.man, "*"):
                carried_s = get_parts(fact)[2]
                carried_spanners.add(carried_s)
                if f"(usable {carried_s})" in state:
                    carrying_usable_spanner = True
                    # No break here, continue finding all carried spanners if needed later (though not in this heuristic)

        # 3. Identify loose goal nuts in the current state
        loose_goal_nuts_in_state = {
            nut for nut in self.goal_nuts
            if f"(loose {nut})" in state
        }

        # 4. If no loose goal nuts, goal is reached for nuts
        if not loose_goal_nuts_in_state:
            return 0

        # 5. Calculate base cost (tighten + pickup per nut)
        # Assumes each loose goal nut needs a tighten action (cost 1)
        # and requires picking up a usable spanner (cost 1), total 2 actions per nut.
        h_base = len(loose_goal_nuts_in_state) * 2

        # 6. Calculate cost to acquire a usable spanner if needed for the *first* nut
        h_acquire_spanner = 0
        if not carrying_usable_spanner:
            usable_spanners_in_state = {
                get_parts(fact)[1] for fact in state
                if match(fact, "usable", "*")
            }

            # Find locations of usable spanners that are NOT currently carried by the man
            usable_spanners_not_carried_locs = {}
            for spanner in usable_spanners_in_state:
                 if spanner not in carried_spanners:
                     for fact in state:
                         if match(fact, "at", spanner, "*"):
                             usable_spanners_not_carried_locs[spanner] = get_parts(fact)[2]
                             break # Found location for this spanner

            if not usable_spanners_not_carried_locs:
                 # No usable spanners available in the state (either carried usable or at a location)
                 # and man needs one. Problem likely unsolvable.
                 return float('inf')

            min_dist_to_spanner = float('inf')
            for loc in usable_spanners_not_carried_locs.values():
                 min_dist_to_spanner = min(min_dist_to_spanner, self.get_distance(man_loc, loc))

            if min_dist_to_spanner == float('inf'):
                 # Cannot reach any usable spanner
                 return float('inf')

            h_acquire_spanner = min_dist_to_spanner + 1 # +1 for pickup action

        # 7. Calculate cost to reach the nearest loose goal nut
        loose_nut_locs = {}
        for nut in loose_goal_nuts_in_state:
             for fact in state:
                 if match(fact, "at", nut, "*"):
                     loose_nut_locs[nut] = get_parts(fact)[2]
                     break # Found location for this nut

        if not loose_nut_locs:
             # Should not happen if loose_goal_nuts_in_state is not empty, but defensive
             return float('inf') # Indicates a problem state where loose nuts have no location

        min_dist_to_nut = float('inf')
        for loc in loose_nut_locs.values():
             min_dist_to_nut = min(min_dist_to_nut, self.get_distance(man_loc, loc))

        if min_dist_to_nut == float('inf'):
             # Cannot reach any loose nut location
             return float('inf')

        h_reach_nut = min_dist_to_nut

        # 8. Total heuristic
        total_cost = h_base + h_acquire_spanner + h_reach_nut

        return total_cost
