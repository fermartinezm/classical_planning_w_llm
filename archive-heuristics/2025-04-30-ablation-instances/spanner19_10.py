# Helper functions (can be outside the class)
from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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

# Heuristic class
from heuristics.heuristic_base import Heuristic

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It counts the number of loose nuts (representing the required tighten actions)
    and adds an estimate of the movement and pickup costs required to perform
    the *first* tightening operation, considering the man's current location
    and whether he is already carrying a usable spanner.

    # Assumptions
    - The problem instances are solvable, implying there are always enough
      usable spanners available (either carried or on the ground) to tighten
      all loose nuts.
    - There is only one man object in the domain.
    - Locations are connected by bidirectional links forming a graph.

    # Heuristic Initialization
    - Extracts all locations and the graph of links from the static facts.
    - Computes all-pairs shortest path distances between all locations using BFS.
    - Identifies the name of the single man object.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all loose nuts and their current locations. Count the total number of loose nuts (`N_loose`).
    2. If `N_loose` is 0, the goal is reached, so the heuristic value is 0.
    3. Find the current location of the man.
    4. Determine if the man is currently carrying a usable spanner.
    5. Initialize the heuristic value with `N_loose` (representing the cost of the tighten actions).
    6. Calculate the minimum movement and/or pickup cost required to enable the *next* tighten action:
       - If the man is already carrying a usable spanner: The cost is the minimum distance from the man's current location to the location of any loose nut. Add this distance to the heuristic.
       - If the man is NOT carrying a usable spanner:
         - Identify the locations of all usable spanners that are currently on the ground.
         - If there are no usable spanners on the ground (and the man isn't carrying one), the problem is likely unsolvable from this state if `N_loose > 0`. Return a large value (representing infinity).
         - The man needs to go to a usable spanner location, pick it up (cost 1), and then go to a loose nut location. Calculate the minimum cost for this sequence: Find the minimum value of (distance from man's location to spanner location + 1 + distance from spanner location to nut location) over all usable spanner locations on the ground and all loose nut locations. Add this minimum cost to the heuristic.
    7. Return the calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing shortest path distances.

        @param task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract locations and build graph from static link facts
        self.locations = set()
        self.graph = {} # Adjacency list

        for fact in static_facts:
            if match(fact, "link", "?l1", "?l2"):
                l1, l2 = get_parts(fact)[1:]
                self.locations.add(l1)
                self.locations.add(l2)
                self.graph.setdefault(l1, []).append(l2)
                self.graph.setdefault(l2, []).append(l1) # Links are bidirectional

        # 2. Compute all-pairs shortest paths using BFS
        self.distance = {loc: {} for loc in self.locations}

        # Define a large value to represent infinity for distances
        # Any distance in an unweighted graph is at most V-1, where V is the number of nodes.
        self.infinity = len(self.locations) + 1

        for start_node in self.locations:
            self.distance[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node}

            while queue:
                u = queue.popleft()
                dist_u = self.distance[start_node][u]

                # Get neighbors, handle locations with no links (though graph should cover all locations)
                neighbors = self.graph.get(u, [])

                for v in neighbors:
                    if v not in visited:
                        visited.add(v)
                        self.distance[start_node][v] = dist_u + 1
                        queue.append(v)

        # Fill in infinity for unreachable pairs (BFS naturally leaves them unassigned or at initial infinity)
        for l1 in self.locations:
            for l2 in self.locations:
                if l2 not in self.distance[l1]:
                    self.distance[l1][l2] = self.infinity


        # 3. Find the name of the man (assuming there's only one man object)
        # We can find the man by looking for an object that is 'at' a location
        # and is not a spanner or nut based on typical naming conventions or
        # presence in 'carrying' facts.
        self.man_name = None
        potential_locatables = set()
        spanner_names = set()
        nut_names = set()

        # Collect all object names that appear as locatables or in relevant predicates
        for fact in task.initial_state | task.goals:
             parts = get_parts(fact)
             if not parts: continue

             predicate = parts[0]
             if predicate == 'at' and len(parts) == 3:
                 potential_locatables.add(parts[1])
             elif predicate == 'carrying' and len(parts) == 3:
                 # First arg is man, second is spanner
                 potential_locatables.add(parts[1]) # Man is locatable
                 spanner_names.add(parts[2]) # Spanner is locatable
             elif predicate in ['loose', 'tightened'] and len(parts) == 2:
                 nut_names.add(parts[1]) # Nut is locatable
             elif predicate == 'usable' and len(parts) == 2:
                 spanner_names.add(parts[1]) # Spanner is locatable

        # The man is a locatable that is not a spanner or nut
        candidate_men = potential_locatables - spanner_names - nut_names

        if len(candidate_men) == 1:
            self.man_name = list(candidate_men)[0]
        elif len(candidate_men) > 1:
             # More than one candidate man? This shouldn't happen based on problem description assumption.
             # Pick one arbitrarily.
             self.man_name = list(candidate_men)[0]
        else:
             # No candidate man found. Problem might be malformed or assumption violated.
             # Heuristic might not work correctly.
             pass # self.man_name remains None


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        @param node: The current search node containing the state.
        @return: The estimated heuristic cost (non-negative integer).
        """
        state = node.state

        # 1. Identify loose nuts and their locations
        loose_nuts = set()
        nut_locations = {}
        for fact in state:
            if match(fact, "loose", "?n"):
                nut_name = get_parts(fact)[1]
                loose_nuts.add(nut_name)
                # Find location of this nut
                for loc_fact in state:
                    if match(loc_fact, "at", nut_name, "?l"):
                        nut_locations[nut_name] = get_parts(loc_fact)[2]
                        break # Found location

        # 2. If N_loose is 0, goal is reached
        n_loose = len(loose_nuts)
        if n_loose == 0:
            return 0

        # Ensure man_name was identified in __init__
        if self.man_name is None:
             # Cannot compute heuristic without man's name.
             # This indicates an issue during initialization or problem definition.
             # Return a large value to signify an error or unsolvable state.
             return 1000000

        # 3. Find man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "?l"):
                man_loc = get_parts(fact)[2]
                break

        # If man_loc is None, the man is not at any location, which is an invalid state.
        # Return a large value.
        if man_loc is None:
             return 1000000

        # 4. Determine if man is carrying a usable spanner
        man_carrying_spanner = False
        carried_spanner_name = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "?s"):
                carried_spanner_name = get_parts(fact)[2]
                # Check if the carried spanner is usable
                if f"(usable {carried_spanner_name})" in state:
                     man_carrying_spanner = True
                break # Assuming man carries at most one spanner

        # 5. Initialize heuristic with N_loose
        h = n_loose

        # 6. Calculate movement/pickup cost for the next tightening operation
        if man_carrying_spanner:
            # Man has spanner, needs to go to a nut.
            # Cost is min distance from man_loc to any nut_loc.
            min_dist_man_to_nut = self.infinity
            for nut_name in loose_nuts:
                 l_n = nut_locations.get(nut_name) # Use .get for safety
                 if l_n is not None and man_loc in self.distance and l_n in self.distance[man_loc]:
                    min_dist_man_to_nut = min(min_dist_man_to_nut, self.distance[man_loc][l_n])

            # If min_dist_man_to_nut is still infinity, it means nuts are in unreachable locations
            # from the man's current location. This shouldn't happen in solvable problems.
            # Return large value.
            if min_dist_man_to_nut == self.infinity:
                 return 1000000 # Effectively infinity

            h += min_dist_man_to_nut

        else:
            # Man needs a spanner first, then go to a nut.
            # Identify usable spanners on the ground and their locations.
            usable_spanner_locations = set()
            for fact in state:
                 # Check for spanners at locations
                 if match(fact, "at", "?s", "?l") and get_parts(fact)[1].startswith("spanner"):
                     spanner_name = get_parts(fact)[1]
                     spanner_loc = get_parts(fact)[2]
                     # Check if this spanner is usable
                     if f"(usable {spanner_name})" in state:
                         usable_spanner_locations.add(spanner_loc)

            if not usable_spanner_locations:
                 # No usable spanners on the ground and man isn't carrying one.
                 # Problem is likely unsolvable from here if N_loose > 0.
                 # Return large value.
                 return 1000000 # Effectively infinity

            # Man needs to go to a spanner, pick it up, then go to a nut.
            # Cost = min (distance(man_loc, l_s) + 1 + distance(l_s, l_n)) over l_s, l_n
            min_cost_get_spanner_then_nut = self.infinity

            for l_s in usable_spanner_locations:
                for nut_name in loose_nuts:
                    l_n = nut_locations.get(nut_name) # Use .get for safety
                    if l_n is not None and man_loc in self.distance and l_s in self.distance[man_loc] and l_n in self.distance[l_s]:
                         cost = self.distance[man_loc][l_s] + 1 + self.distance[l_s][l_n]
                         min_cost_get_spanner_then_nut = min(min_cost_get_spanner_then_nut, cost)

            # If min_cost_get_spanner_then_nut is still infinity, it means no path exists
            # from man_loc to any spanner_loc and then to any nut_loc. Unsolvable.
            if min_cost_get_spanner_then_nut == self.infinity:
                 return 1000000 # Effectively infinity

            h += min_cost_get_spanner_then_nut

        return h
