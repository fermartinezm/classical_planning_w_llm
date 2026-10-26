# Add the necessary import for the base class if it's in a specific path
# from heuristics.heuristic_base import Heuristic

# If the base class is not provided in a separate file, define a placeholder
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the actual one isn't available
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass

import collections # For BFS queue
from fnmatch import fnmatch # For pattern matching


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-string input gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all required nuts by summing the estimated
    cost for each individual nut, assuming a greedy approach of picking up
    the closest available spanner and then walking to the nut.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and
        precomputing distances between locations.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # 1. Identify objects and locations by inferring roles from predicates
        objects = set()
        self.nuts = set()
        self.spanners = set()
        self.man = None
        self.locations = set()
        links = set() # Store as frozenset({l1, l2})

        all_facts = self.initial_state | self.goals | self.static_facts

        # First pass: Collect all objects and identify nuts/spanners/locations based on common predicates
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            args = parts[1:]
            objects.update(args) # Add all arguments as potential objects

            if predicate == 'loose' or predicate == 'tightened':
                if len(args) > 0: self.nuts.add(args[0])
            elif predicate == 'usable':
                 if len(args) > 0: self.spanners.add(args[0])
            elif predicate == 'carrying':
                if len(args) > 1:
                    # Assume first arg of carrying is the man, second is a spanner
                    self.man = args[0]
                    self.spanners.add(args[1])
            elif predicate == 'at':
                if len(args) > 1: self.locations.add(args[1]) # Second arg is a location
            elif predicate == 'link':
                if len(args) > 1:
                    self.locations.add(args[0])
                    self.locations.add(args[1])
                    links.add(frozenset({args[0], args[1]}))

        # If man not found via 'carrying', try to infer from 'at' facts
        if self.man is None:
             locatables_at_start = {get_parts(f)[1] for f in self.initial_state if match(f, "at", "*", "*")}
             # Assume the man is the unique object at a location that is not a nut or spanner
             potential_men = locatables_at_start - self.nuts - self.spanners
             if len(potential_men) == 1:
                 self.man = list(potential_men)[0]
             # else: problem might be malformed or heuristic inference failed to find the man


        # 2. Build location graph adjacency list
        self.adj = {loc: set() for loc in self.locations}
        for link in links:
            l1, l2 = list(link)
            # Ensure locations from links were added to self.locations
            if l1 in self.adj and l2 in self.adj:
                self.adj[l1].add(l2)
                self.adj[l2].add(l1)
            # Handle cases where link refers to locations not found in 'at' facts initially
            # This shouldn't happen in valid PDDL but defensive coding is good.
            elif l1 in self.locations and l2 in self.locations:
                 if l1 not in self.adj: self.adj[l1] = {l2}
                 else: self.adj[l1].add(l2)
                 if l2 not in self.adj: self.adj[l2] = {l1}
                 else: self.adj[l2].add(l1)


        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = {loc: float('inf') for loc in self.locations}
            self.distances[start_node][start_node] = 0
            queue = collections.deque([start_node])

            while queue:
                u = queue.popleft()
                if u in self.adj: # Ensure location exists in adj list
                    for v in self.adj[u]:
                        if self.distances[start_node][v] == float('inf'):
                            self.distances[start_node][v] = self.distances[start_node][u] + 1
                            queue.append(v)

        # 4. Identify goal nuts (nuts that must be tightened)
        self.goal_nuts = {get_parts(g)[1] for g in self.goals if match(g, "tightened", "*")}


    def get_distance(self, loc1, loc2):
        """Returns the precomputed shortest distance between two locations."""
        # Return infinity if either location is not in our known locations
        if loc1 not in self.locations or loc2 not in self.locations:
             return float('inf')
        # Return the precomputed distance, defaulting to infinity if no path exists
        return self.distances[loc1].get(loc2, float('inf'))


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to tighten all goal nuts.
        """
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Ensure man object was identified during initialization
        if self.man is None:
             # Cannot compute heuristic without knowing the man object
             return float('inf')


        # 1. Extract current state information
        current_man_location = None
        man_carrying_spanner = None # Store spanner object name if carried
        loose_nuts_to_tighten = [] # List of (nut_name, nut_location)
        usable_spanners_on_ground = [] # List of (spanner_name, spanner_location)

        # Find man's current location
        for fact in state:
            if match(fact, "at", self.man, "*"):
                current_man_location = get_parts(fact)[2]
                break # Assuming only one man

        if current_man_location is None:
             # Man's location is unknown in this state - problem state is weird or unsolvable
             return float('inf')
        if current_man_location not in self.locations:
             # Man is at an unknown location
             return float('inf')


        # Check if man is carrying a usable spanner
        carried_spanner_name = None
        for fact in state:
            if match(fact, "carrying", self.man, "*"):
                carried_spanner_name = get_parts(fact)[2]
                break # Assuming man carries at most one spanner

        if carried_spanner_name and f'(usable {carried_spanner_name})' in state:
             man_carrying_usable_spanner = carried_spanner_name
        else:
             man_carrying_usable_spanner = None


        # Find loose nuts that are goals and their locations
        for nut in self.goal_nuts:
            if f'(tightened {nut})' not in state:
                # This nut needs tightening
                nut_location = None
                for fact in state:
                    if match(fact, "at", nut, "*"):
                        nut_location = get_parts(fact)[2]
                        break
                if nut_location:
                    if nut_location not in self.locations:
                         # Nut is at an unknown location
                         return float('inf')
                    loose_nuts_to_tighten.append((nut, nut_location))
                else:
                    # Nut is not 'at' any location in the state - unsolvable from here
                    return float('inf')


        # Find usable spanners on the ground and their locations
        for spanner in self.spanners:
             # Check if spanner is on the ground AND usable
             is_on_ground = False
             spanner_location = None
             for fact in state:
                 if match(fact, "at", spanner, "*"):
                     is_on_ground = True
                     spanner_location = get_parts(fact)[2]
                     break

             if is_on_ground and spanner_location and spanner_location in self.locations and f'(usable {spanner})' in state:
                 usable_spanners_on_ground.append((spanner, spanner_location))


        # 2. Check if enough usable spanners exist in total
        num_needed = len(loose_nuts_to_tighten)
        num_available = len(usable_spanners_on_ground) + (1 if man_carrying_usable_spanner else 0)

        if num_needed > num_available:
            return float('inf') # Not enough usable spanners to tighten all goal nuts


        # 3. Estimate cost using a greedy approach
        h = 0
        current_man_loc = current_man_location
        has_usable_spanner = (man_carrying_usable_spanner is not None)

        # Sort nuts by distance from current man location for greedy processing
        # This is a heuristic choice, other sorting might be better
        loose_nuts_to_tighten.sort(key=lambda item: self.get_distance(current_man_loc, item[1]))

        spanners_on_ground_remaining = list(usable_spanners_on_ground) # Copy for modification

        for nut, nut_loc in loose_nuts_to_tighten:
            # Cost to tighten this nut:
            # 1. Get a spanner if needed.
            if not has_usable_spanner:
                # Find closest available usable spanner on the ground
                if not spanners_on_ground_remaining:
                    # This case should be caught by the num_needed > num_available check,
                    # but as a fallback for safety or if the check was insufficient.
                    return float('inf')

                # Sort remaining spanners by distance from current man location
                spanners_on_ground_remaining.sort(key=lambda item: self.get_distance(current_man_loc, item[1]))

                closest_spanner, closest_spanner_loc = spanners_on_ground_remaining.pop(0)

                # Add cost to walk to spanner and pick it up
                dist_to_spanner = self.get_distance(current_man_loc, closest_spanner_loc)
                if dist_to_spanner == float('inf'): return float('inf') # Unreachable spanner
                h += dist_to_spanner # walk
                h += 1 # pickup

                current_man_loc = closest_spanner_loc # Man is now at spanner location
                has_usable_spanner = True # Man is now carrying a usable spanner

            # 2. Walk to the nut location.
            dist_to_nut = self.get_distance(current_man_loc, nut_loc)
            if dist_to_nut == float('inf'): return float('inf') # Unreachable nut
            h += dist_to_nut # walk
            current_man_loc = nut_loc # Man is now at nut location

            # 3. Tighten the nut.
            h += 1 # tighten action
            has_usable_spanner = False # Spanner is consumed/unusable

        return h
