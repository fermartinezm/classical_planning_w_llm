from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function
def bfs_shortest_path(graph, start):
    """Computes shortest path distances from start to all reachable nodes in a graph."""
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current in graph: # Handle nodes with no outgoing links (shouldn't happen with locations from links)
            for neighbor in graph[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances

# Define the heuristic class inheriting from the assumed base class
# class spannerHeuristic(Heuristic): # Uncomment this line in the target environment
class spannerHeuristic: # Use this for standalone testing/generation
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It sums the estimated cost for each individual loose goal nut. The cost for a
    single nut is estimated as the sum of:
    1. The cost of the tighten_nut action itself (always 1).
    2. The estimated cost to get the man from his current location to the nut's location with a usable spanner.

    # Assumptions:
    - There is only one man.
    - Spanners are single-use (become unusable after one tighten_nut action).
    - Enough usable spanners exist initially to tighten all goal nuts in solvable problems.
    - The man can only carry one spanner at a time (implied by the predicate structure).
    - There is no action to drop a spanner or make an unusable spanner usable again.
      If the man is carrying an unusable spanner, he effectively needs to find a *different*
      usable spanner on the ground. The heuristic models this by calculating the cost
      to acquire a usable spanner from the ground whenever the man is not carrying one
      that is usable.
    - The location graph defined by 'link' predicates is connected, allowing travel
      between any two relevant locations in solvable problems.

    # Heuristic Initialization
    - Identify the man object by inspecting initial state facts.
    - Identify all location objects involved in 'link' predicates and build the adjacency list graph.
    - Compute all-pairs shortest paths (distances) between all identified locations using BFS.
    - Identify the set of nuts that need to be tightened based on the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location (`man_loc`). If unknown, return infinity.
    2. Check if the man is currently carrying a spanner, and if so, whether it is usable (`man_carrying_usable`).
    3. Identify all usable spanners currently on the ground and their locations (`usable_spanners_on_ground`).
    4. Identify all nuts that are currently loose and are part of the goal set (`loose_goal_nuts`). If none, return 0.
    5. Initialize total heuristic cost to 0.
    6. For each loose goal nut `n` at its location `loc_n`:
        a. Add 1 to the total cost (for the `tighten_nut` action).
        b. Calculate the estimated cost to get the man to `loc_n` with a usable spanner:
           - If the man is currently carrying a usable spanner (`man_carrying_usable` is True):
             The cost is the travel distance from `man_loc` to `loc_n`: `distances[man_loc][loc_n]`. If unreachable, return infinity.
           - If the man is NOT currently carrying a usable spanner:
             He needs to acquire one from the ground. Find the usable spanner `s_ground` on the ground that is closest to `man_loc`.
             If no usable spanner is found on the ground, this nut cannot be tightened, return infinity.
             The estimated cost is the travel to the spanner, plus pickup, plus travel from spanner location to nut location: `distances[man_loc][loc_s_ground] + 1 + distances[loc_s_ground][loc_n]`. If any travel segment is unreachable, return infinity.
        c. Add this estimated travel+spanner cost to the total cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and building the location graph."""
        # In a real scenario inheriting from Heuristic, task is passed to super().__init__
        # and often stored there. Let's assume task is the input parameter.
        self.task = task # Store task for access to facts, goals, static

        # Identify objects by type
        self.men = set()
        self.spanners = set()
        self.nuts = set()
        self.locations = set()

        # Infer object types by looking at predicate arguments in all possible facts
        # This is a bit fragile, assumes predicates and argument positions are fixed
        for fact in self.task.facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == "carrying" and len(args) == 2:
                 self.men.add(args[0])
                 self.spanners.add(args[1])
            elif predicate == "usable" and len(args) == 1:
                 self.spanners.add(args[0])
            elif predicate == "link" and len(args) == 2:
                 self.locations.add(args[0])
                 self.locations.add(args[1])
            elif predicate == "tightened" and len(args) == 1:
                 self.nuts.add(args[0])
            elif predicate == "loose" and len(args) == 1:
                 self.nuts.add(args[0])
            # 'at' predicate args can be any locatable and location, need other predicates to type
            # We collect locations from 'link' and then refine using 'at' if needed, but 'link' should cover all locations.

        # Assume there is exactly one man based on examples
        # Find the man object name - look for the subject of 'at' that isn't a spanner or nut
        man_obj = None
        for fact in self.task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at" and len(parts) == 3:
                 obj = parts[1]
                 if obj not in self.spanners and obj not in self.nuts:
                     man_obj = obj
                     break
        self.man = man_obj
        if not self.man:
             # Fallback: try finding any object used with 'carrying' predicate
             for fact in self.task.facts:
                 if match(fact, "carrying", "*", "*"):
                     self.man = get_parts(fact)[1]
                     break
        if not self.man:
             # If still not found, this is unexpected for a valid spanner problem
             print("Warning: Could not identify the man object.")


        # Build location graph from static links
        self.location_graph = {loc: [] for loc in self.locations}
        for fact in self.task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                if loc1 in self.location_graph and loc2 in self.location_graph:
                    self.location_graph[loc1].append(loc2)
                    self.location_graph[loc2].append(loc1) # Links are bidirectional

        # Compute all-pairs shortest paths
        self.all_distances = {}
        for start_loc in self.locations:
            self.all_distances[start_loc] = bfs_shortest_path(self.location_graph, start_loc)

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.task.goals:
            if match(goal, "tightened", "*"):
                _, nut = get_parts(goal)
                self.goal_nuts.add(nut)

    def get_location(self, state, obj):
        """Find the current location of an object in the state."""
        for fact in state:
            if match(fact, "at", obj, "*"):
                return get_parts(fact)[2]
        return None # Object not found at any location (e.g., carried)

    def is_carrying(self, state, man, obj):
        """Check if the man is carrying the object."""
        return f"(carrying {man} {obj})" in state

    def is_usable(self, state, spanner):
         """Check if the spanner is usable."""
         return f"(usable {spanner})" in state

    def is_loose(self, state, nut):
         """Check if the nut is loose."""
         return f"(loose {nut})" in state

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.task.goals <= state:
            return 0

        # Find man's current location
        man_loc = self.get_location(state, self.man)
        if man_loc is None:
             # Man's location is unknown, problem likely unsolvable from here
             return float('inf')

        # Check if man is carrying a usable spanner
        man_carrying_usable = False
        for spanner in self.spanners:
            if self.is_carrying(state, self.man, spanner):
                if self.is_usable(state, spanner):
                    man_carrying_usable = True
                # Found the spanner he's carrying, no need to check others
                break

        # Find usable spanners on the ground
        usable_spanners_on_ground = []
        for spanner in self.spanners:
            # Check if spanner is on the ground AND usable
            spanner_loc = self.get_location(state, spanner)
            if spanner_loc and self.is_usable(state, spanner):
                 usable_spanners_on_ground.append((spanner, spanner_loc))

        # Identify loose goal nuts and their locations
        loose_goal_nuts = []
        for nut in self.goal_nuts:
            if self.is_loose(state, nut):
                nut_loc = self.get_location(state, nut)
                if nut_loc:
                    loose_goal_nuts.append((nut, nut_loc))
                else:
                     # Nut location unknown, problem likely unsolvable
                     return float('inf')


        # If there are no loose goal nuts, but goal is not reached, something is wrong
        # (e.g., goal includes other conditions), or it's a goal state.
        # We already checked for goal state. If loose_goal_nuts is empty, h=0.
        if not loose_goal_nuts:
             return 0

        total_cost = 0

        # Estimate cost for each loose goal nut independently
        for nut, nut_loc in loose_goal_nuts:
            # Cost for tighten_nut action
            cost_for_nut = 1

            # Cost to get man to nut location with a usable spanner
            travel_spanner_cost = 0

            if man_carrying_usable:
                # Man has a usable spanner, just needs to travel to the nut
                if man_loc not in self.all_distances or nut_loc not in self.all_distances[man_loc]:
                     # Nut location unreachable from man's location
                     return float('inf')
                travel_spanner_cost = self.all_distances[man_loc][nut_loc]
            else:
                # Man needs to acquire a usable spanner from the ground
                nearest_spanner_loc = None
                min_dist_to_spanner = float('inf')

                for spanner, s_loc in usable_spanners_on_ground:
                     if man_loc in self.all_distances and s_loc in self.all_distances[man_loc]:
                          dist = self.all_distances[man_loc][s_loc]
                          if dist < min_dist_to_spanner:
                               min_dist_to_spanner = dist
                               nearest_spanner_loc = s_loc

                if nearest_spanner_loc is not None:
                     # Cost is travel to spanner + pickup + travel from spanner to nut
                     if nearest_spanner_loc not in self.all_distances or nut_loc not in self.all_distances[nearest_spanner_loc]:
                          # Nut location unreachable from spanner location
                          return float('inf')
                     travel_spanner_cost = min_dist_to_spanner + 1 + self.all_distances[nearest_spanner_loc][nut_loc]
                else:
                     # No usable spanners available on the ground and man doesn't have one
                     # This state is unsolvable for this nut
                     return float('inf') # Cannot tighten this nut

            cost_for_nut += travel_spanner_cost
            total_cost += cost_for_nut

        return total_cost
