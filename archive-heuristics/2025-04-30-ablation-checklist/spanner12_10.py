from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

# Helper functions to parse PDDL facts
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain using greedy simulation.

    Estimates the cost by simulating the process of tightening each loose nut
    one by one, always choosing the nearest available usable spanner if needed,
    and accounting for travel and pickup costs.

    Preprocessing: Computes all-pairs shortest paths between locations.
    Heuristic Calculation:
    1. Identify loose nuts, man's location, and usable spanners.
    2. If no loose nuts, cost is 0.
    3. If not enough usable spanners exist for all loose nuts, return a large value.
    4. Sort loose nuts by estimated cost to tighten them first from the current state.
    5. Iterate through sorted loose nuts:
       - Add 1 for the tighten action.
       - If man is carrying a usable spanner: Add travel cost to the nut. Simulate spanner consumption.
       - If man needs a spanner: Find the nearest available usable spanner on the ground. Add travel to spanner, pickup cost (1), and travel from spanner location to nut location. Simulate spanner pickup and consumption. Update man's location.
    6. Sum costs for all nuts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting locations, objects, and
        computing all-pairs shortest paths.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # Collect all terms appearing in initial state and static facts
        all_terms = set()
        for fact in self.initial_state | self.static_facts:
            all_terms.update(get_parts(fact)[1:]) # Add all arguments

        # Classify terms based on predicate usage
        self.all_men = set()
        self.all_nuts = set()
        self.all_spanners = set()
        self.all_locations = set()

        for term in all_terms:
            # Check if it acts as a man
            if any(match(fact, "carrying", term, "*") for fact in self.initial_state):
                self.all_men.add(term)
            # Check if it acts as a spanner
            if any(match(fact, "carrying", "*", term) for fact in self.initial_state) or \
               any(match(fact, "usable", term) for fact in self.initial_state):
                self.all_spanners.add(term)
            # Check if it acts as a nut
            if any(match(fact, "loose", term) for fact in self.initial_state) or \
               any(match(fact, "tightened", term) for fact in self.goals): # Goals also define nuts
                self.all_nuts.add(term)
            # Check if it acts as a location
            if any(match(fact, "at", "*", term) for fact in self.initial_state) or \
               any(match(fact, "link", term, "*") for fact in self.static_facts) or \
               any(match(fact, "link", "*", term) for fact in self.static_facts):
                self.all_locations.add(term)

        # Handle the single man assumption
        self.the_man = list(self.all_men)[0] if self.all_men else None
        if self.the_man is None:
             # Fallback if no 'carrying' facts in initial state, assume first object in 'at' is man
             for fact in self.initial_state:
                  parts = get_parts(fact)
                  if parts[0] == "at":
                       obj = parts[1]
                       # Check if it's not a known nut, spanner, or location
                       if obj not in self.all_nuts and obj not in self.all_spanners and obj not in self.all_locations:
                            self.all_men.add(obj)
                            self.the_man = obj
                            break # Assume only one man


        # 2. Build location graph
        self.graph = {loc: [] for loc in self.all_locations}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                l1, l2 = parts[1], parts[2]
                if l1 in self.graph and l2 in self.graph: # Ensure locations were identified
                     self.graph[l1].append(l2)
                     self.graph[l2].append(l1)

        # 3. Compute all-pairs shortest paths using BFS from each node
        self.distances = {loc: {other_loc: math.inf for other_loc in self.all_locations} for loc in self.all_locations}

        for start_node in self.all_locations:
            self.distances[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node}

            while queue:
                u = queue.popleft()
                if u not in self.graph: continue # Should not happen if locations are from graph/state

                for v in self.graph.get(u, []): # Use .get for safety
                    if v not in visited:
                        visited.add(v)
                        self.distances[start_node][v] = self.distances[start_node][u] + 1
                        queue.append(v)

    def get_location(self, obj, state):
        """Find the current location of an object in the state."""
        # Check if the object is the man
        if obj == self.the_man:
             for fact in state:
                  if match(fact, "at", self.the_man, "*"):
                       return get_parts(fact)[2]
             return None # Man must have a location

        # Check if the object is carried by the man (only spanners can be carried by the man)
        if obj in self.all_spanners:
             if any(match(fact, "carrying", self.the_man, obj) for fact in state):
                  # Recursively find man's location
                  return self.get_location(self.the_man, state)

        # Check if the object is at a location on the ground
        for fact in state:
            if match(fact, "at", obj, "*"):
                return get_parts(fact)[2]

        return None # Object location unknown

    def dist(self, loc1, loc2):
         """Get the precomputed shortest distance between two locations."""
         if loc1 is None or loc2 is None or loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
              return math.inf # Handle cases where location is unknown or not in graph
         return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute the domain-dependent heuristic value for the given state."""
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
             return 0

        # Identify loose nuts in the current state
        loose_nuts = {n for n in self.all_nuts if f"(loose {n})" in state}
        num_loose_nuts = len(loose_nuts)

        if num_loose_nuts == 0:
             return 0 # Should be covered by goal check, but double check

        # Identify man's location
        man_loc = self.get_location(self.the_man, state)
        if man_loc is None:
             # Man's location unknown - problem state is likely invalid or unsolvable
             return math.inf

        # Identify usable spanners (carried or on ground) in the current state
        usable_spanners_total = {s for s in self.all_spanners if f"(usable {s})" in state}
        # Usable spanners currently on the ground (have an (at s l) fact)
        usable_spanners_on_ground_current = {s for s in usable_spanners_total if self.get_location(s, state) in self.all_locations}

        # Check if the man is currently carrying a usable spanner
        man_carrying_usable = any(match(fact, "carrying", self.the_man, s) for s in usable_spanners_total)

        # Check if enough usable spanners exist for remaining nuts
        if num_loose_nuts > len(usable_spanners_total):
             # Not enough usable spanners available in the entire state to tighten all nuts
             return math.inf # Problem is unsolvable from this state

        h = 0
        current_man_loc = man_loc
        # Use mutable sets for simulation
        current_usable_spanners_on_ground = set(usable_spanners_on_ground_current)
        current_man_carrying_usable = man_carrying_usable

        # Define the sorting key function inside __call__ to access simulation state variables
        def estimated_cost_to_tighten_first(nut):
             nut_loc = self.get_location(nut, state) # Get nut location from the actual state (nuts don't move)
             if nut_loc is None: return math.inf # Should not happen for loose nuts

             if current_man_carrying_usable:
                  # Cost if man already has spanner: walk from current sim loc to nut + tighten
                  walk_cost = self.dist(current_man_loc, nut_loc)
                  if walk_cost == math.inf: return math_cost # Cannot reach nut
                  return walk_cost + 1
             else:
                  # Cost if man needs spanner: walk from current sim loc to nearest available spanner + pickup + walk from spanner loc to nut + tighten
                  min_spanner_path_cost = math.inf
                  best_spanner_loc = None

                  # Find the nearest usable spanner *currently* on the ground in the simulation
                  for s in current_usable_spanners_on_ground:
                       s_loc = self.get_location(s, state) # Get spanner location from the actual state (spanners on ground don't move unless picked up)
                       if s_loc is None: continue # Should not happen if s is in current_usable_spanners_on_ground

                       # Cost: walk from current man loc to spanner, pickup, walk from spanner loc to nut loc
                       path_cost = self.dist(current_man_loc, s_loc) + 1 + self.dist(s_loc, nut_loc)
                       min_spanner_path_cost = min(min_spanner_path_cost, path_cost)

                  if min_spanner_path_cost == math.inf:
                       # No reachable usable spanners left on the ground in the simulation
                       return math.inf # Cannot tighten this nut

                  # Add the tighten cost (1) to the path cost
                  return min_spanner_path_cost + 1


        # Sort nuts by the estimated cost to tighten them *first* from the current simulation state
        # This greedy ordering might be important for heuristic performance
        loose_nuts_list_sorted = sorted(list(loose_nuts), key=estimated_cost_to_tighten_first)

        # Greedy simulation
        for nut in loose_nuts_list_sorted:
            nut_loc = self.get_location(nut, state) # Nut location is static

            cost_for_this_nut = 1 # Cost of the tighten action itself

            if current_man_carrying_usable:
                # Man has a usable spanner, just need to walk to the nut
                walk_cost = self.dist(current_man_loc, nut_loc)
                if walk_cost == math.inf: return math.inf # Cannot reach nut
                cost_for_this_nut += walk_cost

                # Simulate spanner consumption
                current_man_carrying_usable = False
                # The specific spanner carried is now unusable. It's still carried, but unusable.
                # We don't need to track *which* spanner is carried, just *if* a usable one is.

            else:
                # Man needs to get a usable spanner
                min_spanner_path_cost = math.inf
                best_spanner = None
                best_spanner_loc = None

                # Find the nearest available usable spanner on the ground *in the current simulation state*
                for s in current_usable_spanners_on_ground:
                    s_loc = self.get_location(s, state) # Get spanner location from the actual state (spanners on ground don't move unless picked up)
                    if s_loc is None: continue # Should not happen if s is in current_usable_spanners_on_ground

                    # Cost: walk from current man loc to spanner, pickup, walk from spanner loc to nut loc
                    path_cost = self.dist(current_man_loc, s_loc) + 1 + self.dist(s_loc, nut_loc)
                    if path_cost < min_spanner_path_cost:
                        min_spanner_path_cost = path_cost
                        best_spanner = s
                        best_spanner_loc = s_loc

                if best_spanner is None or min_spanner_path_cost == math.inf:
                    # No usable spanners left on the ground that are reachable in the simulation
                    return math.inf

                cost_for_this_nut += min_spanner_path_cost

                # Simulate spanner pickup and consumption
                current_usable_spanners_on_ground.remove(best_spanner)
                current_man_carrying_usable = True # Man is now carrying a usable spanner (the one just picked up)

            # Simulate man's location update after arriving at the nut location and tightening
            current_man_loc = nut_loc

            h += cost_for_this_nut

        return h
