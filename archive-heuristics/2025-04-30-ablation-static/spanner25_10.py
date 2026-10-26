# Assuming a base class 'Heuristic' exists in 'heuristics.heuristic_base'
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque
import math # For infinity

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the total cost to tighten all initially loose nuts.
    It uses an additive approach (similar to h_add), summing the estimated minimum
    cost for each remaining loose nut independently. The cost for a single nut
    includes walking to a usable spanner, picking it up, walking to the nut's
    location, and tightening it. Shortest path distances between locations are
    precomputed using BFS.

    # Assumptions
    - All nuts that are initially loose must be tightened to reach the goal.
    - Spanners are single-use (become unusable after one tighten action).
    - The man can only carry one spanner at a time.
    - Nuts do not move from their initial locations.
    - The graph of locations connected by 'link' predicates is connected, or at
      least all relevant locations (initial man location, spanner locations,
      nut locations) are in the same connected component.
    - There are enough usable spanners initially for all loose nuts in solvable problems.

    # Heuristic Initialization
    - Identify all objects (man, spanners, nuts, locations) based on initial state predicates.
    - Identify all nuts that are initially 'loose'. These are the nuts that need
      to be tightened to reach the goal. Store their initial locations.
    - Build the location graph based on 'link' facts using identified locations.
    - Compute all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the goal is reached. If yes, return 0.
    2. Identify the man's current location. If the man's location is unknown, return infinity.
    3. Identify all spanners that are currently 'usable' in this state and their locations
       (either 'at' a location or 'carrying' by the man).
    4. Initialize the total heuristic cost to 0.
    5. Iterate through the set of nuts that were initially loose (identified during initialization).
    6. For each initially loose nut:
       a. Check if this nut is already 'tightened' in the current state. If yes,
          it contributes 0 to the heuristic for this state. Continue to the next nut.
       b. If the nut is not yet 'tightened':
          i. Get the initial location of this nut (assuming it hasn't moved). If the initial location is unknown, return infinity.
          ii. Initialize the minimum cost to tighten this specific nut to infinity.
          iii. Find all spanners that are currently 'usable' in this state.
          iv. For each currently 'usable' spanner:
             - Calculate the cost for the man to get this spanner and bring it
               to the nut's location, plus the cost of the tighten action.
             - If the man is currently carrying this usable spanner:
               Cost = distance from man's current location to the nut's location + 1 (tighten).
             - If the usable spanner is currently at a location:
               Cost = distance from man's current location to spanner's location + 1 (pickup)
                      + distance from spanner's location to nut's location + 1 (tighten).
             - Update the minimum cost for this nut with the minimum of the calculated costs
               over all usable spanners.
          v. If after checking all usable spanners, the minimum cost for this nut is still infinity,
             it means this nut cannot be tightened with any currently usable spanner. This state
             is likely unsolvable. Return infinity.
          vi. Add the minimum cost found for this nut to the total heuristic cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing distances and identifying goal nuts."""
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # Identify objects and their types based on initial state and static facts
        all_objects = set()
        initial_at_facts = {} # obj -> loc
        initial_carrying_facts = {} # man -> spanner
        initial_usable_spanners = set()
        initial_loose_nuts = set()

        for fact in self.initial_state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            # Add the first argument of any fact as a potential object
            if len(parts) > 1:
                all_objects.add(parts[1])

            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                all_objects.add(loc) # Add location object
                initial_at_facts[obj] = loc
            elif parts[0] == 'carrying' and len(parts) == 3:
                man_obj, spanner_obj = parts[1], parts[2]
                all_objects.add(spanner_obj) # Add spanner object
                initial_carrying_facts[man_obj] = spanner_obj
            elif parts[0] == 'usable' and len(parts) == 2:
                spanner_obj = parts[1]
                all_objects.add(spanner_obj) # Add spanner object
                initial_usable_spanners.add(spanner_obj)
            elif parts[0] == 'loose' and len(parts) == 2:
                nut_obj = parts[1]
                all_objects.add(nut_obj) # Add nut object
                initial_loose_nuts.add(nut_obj)

        # Identify locations from static facts as well
        static_locations = set()
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts[0] == 'link' and len(parts) == 3:
                 l1, l2 = parts[1], parts[2]
                 all_objects.add(l1) # Add location object
                 all_objects.add(l2) # Add location object
                 static_locations.add(l1)
                 static_locations.add(l2)

        # Refined type identification based on predicate roles
        self.initial_loose_nuts = initial_loose_nuts
        # Spanners are objects that are usable or carried initially
        self.all_spanners = {obj for obj in all_objects if obj in initial_usable_spanners or any(obj == s for m, s in initial_carrying_facts.items())}
        # Man is the object that is carrying something initially
        self.man = {m for m in initial_carrying_facts.keys()}
        # Assuming there's exactly one man, pick the first one found
        if len(self.man) == 1:
             self.man = list(self.man)[0]
        elif len(self.man) > 1:
             # Handle multiple men if necessary, but domain implies one. Pick one.
             self.man = list(self.man)[0]
        else:
             # If no carrying facts, try to infer man as the single locatable that isn't a spanner/nut
             potential_men_at = {obj for obj, loc in initial_at_facts.items() if obj not in self.initial_loose_nuts and obj not in self.all_spanners}
             if len(potential_men_at) == 1:
                  self.man = list(potential_men_at)[0]
             else:
                  # Fallback: Assume the first object in initial_at_facts that isn't a known nut or spanner is the man.
                  found_man = False
                  for obj in initial_at_facts:
                       if obj not in self.initial_loose_nuts and obj not in self.all_spanners:
                            self.man = obj
                            found_man = True
                            break
                  if not found_man:
                       self.man = None # This indicates a problem parsing the instance

        # Locations are all objects that are not man, spanners, or nuts
        self.locations = {obj for obj in all_objects if obj != self.man and obj not in self.all_spanners and obj not in self.initial_loose_nuts}

        # Store initial locations of nuts (assuming they don't move)
        self.nut_initial_location = {nut: initial_at_facts.get(nut) for nut in self.initial_loose_nuts if nut in initial_at_facts}


        # 1. Build the graph based on 'link' facts using identified locations
        self.location_graph = {loc: [] for loc in self.locations} # Initialize with all identified locations
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts[0] == 'link' and len(parts) == 3:
                 l1, l2 = parts[1], parts[2]
                 # Only add links if both endpoints were identified as locations
                 if l1 in self.locations and l2 in self.locations:
                     self.location_graph[l1].append(l2)
                     self.location_graph[l2].append(l1) # Links are bidirectional
                 # else: print(f"Warning: Link involves non-location object or unknown location: {fact}") # Debugging

        # 2. Compute all-pairs shortest path distances using BFS
        self.distances = {} # {(loc1, loc2): distance, ...}
        for start_node in self.locations:
            self.distances[(start_node, start_node)] = 0
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_loc, dist = queue.popleft()

                # Ensure current_loc is a valid key in the graph before accessing neighbors
                if current_loc not in self.location_graph:
                     # This can happen if a location from initial_at_facts was added
                     # but wasn't in any link facts. It's an isolated location with no paths.
                     continue # No neighbors to explore from here

                for neighbor in self.location_graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, return infinity if no path or location unknown."""
        # Check if locations are valid and if distance was computed
        if loc1 not in self.locations or loc2 not in self.locations:
             return float('inf')
        return self.distances.get((loc1, loc2), float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check if goal is reached
        if self.goals <= state:
             return 0

        # Ensure man object was identified during initialization
        if self.man is None:
             # Should not happen in valid problems, indicates parsing issue
             return float('inf')

        # 2. Identify the man's current location.
        man_location = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_location = get_parts(fact)[2]
                break

        if man_location is None or man_location not in self.locations:
             # Man is not at a known location? This state is likely unreachable or invalid.
             return float('inf')

        # 3. Identify all spanners that are currently 'usable' and their locations/status.
        usable_spanners_info = [] # List of (spanner_obj, location_or_carrying_status)
        for spanner in self.all_spanners:
             if f"(usable {spanner})" in state:
                 # Check if carrying by the man
                 if f"(carrying {self.man} {spanner})" in state:
                     usable_spanners_info.append((spanner, "carrying"))
                 else:
                     # Check if at a location
                     found_at = False
                     for fact in state:
                         if match(fact, "at", spanner, "*"):
                             loc = get_parts(fact)[2]
                             # Ensure the location is one we know about
                             if loc in self.locations:
                                 usable_spanners_info.append((spanner, loc))
                                 found_at = True
                                 break # Found location, move to next spanner
                             # else: print(f"Warning: Usable spanner {spanner} at unknown location {loc} in state.") # Debugging
                     # If spanner is usable but not carrying and not at a known location, it's effectively unavailable for this heuristic.


        # 4. Initialize the total heuristic cost to 0.
        total_heuristic = 0

        # 5. Iterate through the set of nuts that were initially loose.
        loose_nuts_remaining = []
        for nut in self.initial_loose_nuts:
            # 6.a. Check if this nut is already 'tightened'.
            if f"(tightened {nut})" not in state:
                 # 6.b. If the nut is not yet 'tightened'.
                 loose_nuts_remaining.append(nut)

        # If no loose nuts remaining, the goal should have been reached (checked at start).
        # If loose_nuts_remaining is empty, the loop below is skipped, total_heuristic is 0, which is correct.

        for nut in loose_nuts_remaining:
            # 6.b.i. Get the initial location of this nut (assuming it hasn't moved).
            nut_location = self.nut_initial_location.get(nut)
            if nut_location is None or nut_location not in self.locations:
                 # Initial location unknown or not a valid location in our graph
                 return float('inf') # Invalid or unsolvable state

            # 6.b.ii. Initialize the minimum cost to tighten this specific nut to infinity.
            min_cost_for_nut = float('inf')

            # 6.b.iii-iv. For each currently 'usable' spanner:
            for spanner, status in usable_spanners_info:
                 cost_this_spanner = float('inf')
                 if status == "carrying":
                     # Man is carrying this usable spanner
                     # Cost = walk from man_location to nut_location + tighten
                     dist_m_to_n = self.get_distance(man_location, nut_location)
                     if dist_m_to_n != float('inf'):
                          cost_this_spanner = dist_m_to_n + 1 # +1 for tighten action
                 else: # status is a location L_S
                     spanner_location = status
                     # Cost = walk m to s + pickup + walk s to n + tighten
                     dist_m_to_s = self.get_distance(man_location, spanner_location)
                     dist_s_to_n = self.get_distance(spanner_location, nut_location)
                     if dist_m_to_s != float('inf') and dist_s_to_n != float('inf'):
                          cost_this_spanner = dist_m_to_s + 1 + dist_s_to_n + 1 # +1 for pickup, +1 for tighten

                 min_cost_for_nut = min(min_cost_for_nut, cost_this_spanner)

            # 6.b.v. If after checking all usable spanners, the minimum cost for this nut is still infinity,
            # it means this nut cannot be tightened with any currently usable spanner.
            if min_cost_for_nut == float('inf'):
                 return float('inf') # Unsolvable from here

            # 6.b.vi. Add the minimum cost found for this nut to the total heuristic cost.
            total_heuristic += min_cost_for_nut

        # 7. Return the total heuristic cost.
        return total_heuristic
