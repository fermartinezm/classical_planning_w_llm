from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at obj loc)" -> ["at", "obj", "loc"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must exactly match the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument (with wildcard support)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    goal nuts. It considers the number of nuts still loose, the number of
    spanners that need to be picked up, and the travel cost to reach the
    nearest required location (either a loose nut or a spanner).

    # Assumptions
    - There is only one man.
    - Nuts are static at their initial locations.
    - Spanners are used once per tightening action and become unusable.
    - The problem is solvable (there are enough usable spanners initially).
    - Locations are connected by an undirected graph defined by `link` predicates.

    # Heuristic Initialization
    - Identify the man object by looking for objects involved in 'carrying' or 'at' predicates that are not nuts or spanners.
    - Identify all location objects from 'at' and 'link' predicates.
    - Build the connectivity graph based on `link` predicates.
    - Precompute all-pairs shortest paths between locations using BFS.
    - Store the initial locations of all nuts from the initial state.
    - Store the set of spanners that are initially usable.
    - Store the set of nuts that are goals (need to be tightened).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the set of nuts that are goals (need to be tightened) but are currently loose. Let this set be `LooseGoalNuts`.
    2. If `LooseGoalNuts` is empty, the heuristic is 0 (goal state).
    3. The base cost is the number of `tighten_nut` actions needed, which is `|LooseGoalNuts|`.
    4. Determine if the man is currently carrying a usable spanner. A spanner is usable if the `(usable ?s)` predicate is true for it in the current state.
    5. Calculate the number of `pickup_spanner` actions needed. This is the number of loose goal nuts minus 1 if the man is already carrying a usable spanner (since he has one spanner ready), capped at a minimum of 0. Add this to the total cost.
    6. Find the man's current location.
    7. Get the locations of all loose goal nuts. These locations are static and stored during initialization.
    8. Get the locations of all usable spanners that are currently on the ground (not carried by the man).
    9. Determine the set of "target" locations the man needs to visit next. This set includes the locations of all loose goal nuts. If the man is *not* currently carrying a usable spanner, it also includes the locations of all usable spanners on the ground (as he needs to pick one up).
    10. Calculate the minimum travel cost from the man's current location to any of the target locations using the precomputed shortest paths.
    11. Add the minimum travel cost to the total cost.
    12. Return the total estimated cost. If any required location is unreachable, return infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = task.goals
        self.initial_state = task.initial_state
        static_facts = task.static

        # Identify objects by type and locations
        self.man = None
        self.spanners = set()
        self.nuts = set()
        self.locations = set()
        self.nut_location = {} # Stores initial location of nuts
        self.initial_usable_spanners = set() # Stores spanners initially usable

        # Goal nuts are those mentioned in tightened goals
        self.goal_nuts_set = {get_parts(g)[1] for g in self.goals if match(g, "tightened", "*")}

        # Collect objects and locations from initial state and static facts
        # Infer types based on common predicates
        potential_men = set()
        potential_spanners = set()
        potential_nuts = set()
        potential_locations = set()

        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                potential_locations.add(loc)
                # Tentative classification based on typical predicates
                if obj in self.goal_nuts_set or any(match(f, "loose", obj) for f in self.initial_state):
                     potential_nuts.add(obj)
                     self.nut_location[obj] = loc # Store initial nut location
                elif any(match(f, "usable", obj) for f in self.initial_state) or any(match(f, "carrying", "*", obj) for f in self.initial_state):
                     potential_spanners.add(obj)
                else:
                     potential_men.add(obj) # Assume it's a man if not nut/spanner
            elif parts[0] == "carrying":
                 m, s = parts[1], parts[2]
                 potential_men.add(m)
                 potential_spanners.add(s)
            elif parts[0] == "usable":
                 s = parts[1]
                 potential_spanners.add(s)
                 self.initial_usable_spanners.add(s)
            elif parts[0] == "loose":
                 n = parts[1]
                 potential_nuts.add(n)

        # Refine object sets based on potential classifications
        self.nuts = potential_nuts
        self.spanners = potential_spanners
        # Assume the single object identified as potential man is the man
        if len(potential_men) == 1:
            self.man = list(potential_men)[0]
        elif len(potential_men) > 1:
             # If multiple potential men, pick one. This is a heuristic assumption.
             print(f"Warning: Multiple potential men identified: {potential_men}. Picking one arbitrarily.")
             self.man = list(potential_men)[0]
        # else: self.man remains None, handled later if needed

        # Add locations from static links
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                potential_locations.add(l1)
                potential_locations.add(l2)

        self.locations = potential_locations

        # Build location graph
        self.graph = {loc: set() for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                # Ensure locations are in our collected set before adding to graph
                if l1 in self.graph and l2 in self.graph:
                    self.graph[l1].add(l2)
                    self.graph[l2].add(l1) # Links are bidirectional

        # Compute all-pairs shortest paths
        self.distances = {loc: self.bfs(loc, self.graph) for loc in self.locations}

        # Final check for man identification if not found yet
        if self.man is None:
             # Attempt to find man from initial 'at' fact if not found in 'carrying' etc.
             for fact in self.initial_state:
                 if match(fact, "at", "*", "*"):
                     obj = get_parts(fact)[1]
                     if obj not in self.nuts and obj not in self.spanners:
                         self.man = obj
                         break # Assume first such object is the man

        if self.man is None:
             print("Warning: Could not identify the man object during initialization.") # Should not happen in valid problems


    def bfs(self, start_loc, graph):
        """Perform BFS to find shortest distances from start_loc to all other locations."""
        dist = {loc: float('inf') for loc in graph}
        # If start_loc is not in the graph (e.g., malformed problem), return inf distances
        if start_loc not in dist:
             return dist

        dist[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            curr = queue.popleft()
            current_dist = dist[curr]

            # Ensure curr is a valid key in the graph
            if curr in graph:
                for neighbor in graph[curr]:
                    if dist[neighbor] == float('inf'):
                        dist[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return dist

    def get_distance(self, loc1, loc2):
        """Lookup shortest distance between two locations."""
        # Return infinity if locations are not in the precomputed distances map
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             return float('inf')
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nuts = {nut for nut in self.goal_nuts_set if f"(loose {nut})" in state}

        # 2. If no loose goal nuts, goal is reached
        if not loose_goal_nuts:
            return 0

        # 3. Base cost: tighten actions
        n_loose_goals = len(loose_goal_nuts)
        total_cost = n_loose_goals

        # 4. Check if man is carrying a usable spanner
        is_carrying_usable = False
        carried_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man, "*"):
                carried_spanner = get_parts(fact)[2]
                # A carried spanner is usable if the (usable S) predicate is true in the current state
                if f"(usable {carried_spanner})" in state:
                     is_carrying_usable = True
                break # Assuming man carries at most one spanner

        # 5. Calculate pickup actions needed
        # Man needs N spanners total (one per nut). He has 1 if carrying usable. Needs N - (1 or 0) pickups.
        pickup_actions_needed = max(0, n_loose_goals - (1 if is_carrying_usable else 0))
        total_cost += pickup_actions_needed

        # 6. Find man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
             # Man location not found, problem state is likely invalid or unsolvable
             return float('inf')

        # 7. Get locations of loose goal nuts
        nut_locations = set()
        for nut in loose_goal_nuts:
            if nut in self.nut_location:
                nut_locations.add(self.nut_location[nut])
            else:
                 # Location of a goal nut not found in initial state - malformed problem?
                 # Nuts are static, so initial location is the only one.
                 print(f"Warning: Location for goal nut {nut} not found in initial state.")
                 return float('inf') # Malformed problem

        # 8. Get locations of usable spanners on ground
        ground_spanner_locations = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                s, l = get_parts(fact)[1], get_parts(fact)[2]
                # Check if it's a spanner, is usable, and is not carried by the man
                if s in self.spanners and f"(usable {s})" in state and not (carried_spanner == s):
                     ground_spanner_locations.add(l)

        # 9. Determine target locations for travel
        target_locations = set(nut_locations)
        if not is_carrying_usable:
            # If man needs a spanner, he might go to a spanner location first
            target_locations.update(ground_spanner_locations)

        # 10. Calculate minimum travel cost
        min_travel_cost = float('inf')
        # Ensure man's location is in our graph before calculating distances
        if man_location in self.distances:
            for target_loc in target_locations:
                min_travel_cost = min(min_travel_cost, self.get_distance(man_location, target_loc))

        # If no reachable target location, problem is likely unsolvable from here
        if min_travel_cost == float('inf'):
             return float('inf')

        total_cost += min_travel_cost

        # 11. Return total estimated cost
        return total_cost
