from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

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
    # Check if the number of parts matches the number of args, unless args has wildcards that can match anything
    # This check can be complex with wildcards, simpler to rely on zip and fnmatch
    # if len(parts) != len(args) and '*' not in args:
    #      return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_node, adj):
    """
    Performs Breadth-First Search to find shortest distances from start_node
    to all reachable nodes in the graph represented by adj.
    Returns a dictionary {node: distance}. Unreachable nodes are not included.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        if current_node in adj:
            for neighbor in adj[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    goal nuts. It considers the cost of tightening each nut, the cost for the
    man to reach a nut location, and the cost for the man to acquire a usable
    spanner and bring it to a nut location if needed. The heuristic focuses
    on the cost to tighten the *first* remaining nut, plus the count of all
    remaining nuts.

    # Assumptions
    - Nuts do not move from their initial locations.
    - Spanners do not move unless carried by the man.
    - A usable spanner becomes unusable after one tightening action.
    - The location graph defined by 'link' predicates is undirected.
    - The problem is solvable (i.e., enough usable spanners exist and locations are connected).
    - There is exactly one man object.

    # Heuristic Initialization
    - Identify all goal nuts from the task goals.
    - Extract all location names from initial state and static facts ('at', 'link').
    - Build the undirected location graph from 'link' predicates.
    - Compute all-pairs shortest paths between all identified locations using BFS.
    - Identify the names of the man, nut, and spanner objects based on initial state predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of goal nuts that are currently 'loose' in the state. Let this set be U.
    2. If U is empty, the goal is reached, heuristic is 0.
    3. Initialize heuristic value `h` with the number of loose goal nuts (`len(U)`),
       representing the minimum number of 'tighten_nut' actions required.
    4. Determine the man's current location (L_M) from the state.
    5. Identify usable spanners the man is currently carrying (S_usable_carried) from the state.
    6. Identify usable spanners available on the ground and their locations (S_usable_available) from the state.
    7. Get the current locations of all loose goal nuts (NutLocations) from the state.
    8. Check if the man is currently carrying a usable spanner (`len(S_usable_carried) > 0`).
    9. If the man *is* carrying a usable spanner:
       - He needs to travel to the location of one of the loose nuts.
       - Add the minimum distance from the man's current location (L_M) to any location in NutLocations to `h`. If no nut locations are reachable, return infinity.
    10. If the man is *not* carrying a usable spanner:
        - He needs to acquire a usable spanner and bring it to a nut location.
        - Calculate the minimum cost to achieve this for the *first* spanner and *first* nut:
          - Find the minimum cost over all available usable spanners (s at L_s) and all nut locations (L_n) in NutLocations.
          - The cost for a specific (s, L_s, L_n) is:
            distance(L_M, L_s) (walk to spanner) + 1 (pickup action) + distance(L_s, L_n) (walk spanner to nut).
        - Add this minimum cost to `h`. If no usable spanners are available or no path exists, return infinity.
    11. Return the calculated value of `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # 1. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened":
                self.goal_nuts.add(parts[1])

        # 2. Extract locations and build location graph
        self.locations = set()
        self.adj = {} # Adjacency list {loc: [neighbor1, neighbor2, ...]}

        # Collect all locations mentioned in initial state and static facts
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 self.locations.add(parts[2])
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts[0] == 'link':
                 self.locations.add(parts[1])
                 self.locations.add(parts[2])
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'at': # Goal might specify object location
                 self.locations.add(parts[2])


        # Build adjacency list from link facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                 loc1, loc2 = parts[1], parts[2]
                 # Ensure locations are in our set (should be if collected above)
                 if loc1 in self.locations and loc2 in self.locations:
                     self.adj.setdefault(loc1, []).append(loc2)
                     self.adj.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # 3. Compute all-pairs shortest paths
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = bfs(loc, self.adj)

        # 4. Identify object names and types (man, nuts, spanners)
        self.man_name = None
        self.initial_nut_names = set()
        self.initial_spanner_names = set()

        # Collect all objects mentioned in initial 'at' facts
        initial_locatable_objects = {get_parts(f)[1] for f in task.initial_state if get_parts(f)[0] == 'at'}

        # Identify nuts based on loose/tightened status in initial state or being a goal nut
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ["loose", "tightened"]:
                 self.initial_nut_names.add(parts[1])
        self.initial_nut_names.update(self.goal_nuts) # Add goal nuts even if not loose/tightened initially

        # Identify spanners based on usable/carrying status in initial state
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == "usable":
                 self.initial_spanner_names.add(parts[1])
             elif parts[0] == "carrying":
                 # The carried object is a spanner
                 self.initial_spanner_names.add(parts[2])

        # The man is the locatable object that is not a nut or a spanner
        man_candidates = initial_locatable_objects - self.initial_nut_names - self.initial_spanner_names
        if len(man_candidates) == 1:
             self.man_name = list(man_candidates)[0]
        elif len(man_candidates) > 1:
             # Handle multiple men? Domain description implies one man.
             # Assume the first one found is the man.
             self.man_name = list(man_candidates)[0]
             # print(f"Warning: Found multiple potential man objects: {man_candidates}. Assuming {self.man_name} is the man.")
        else:
             # No man found? Invalid problem?
             self.man_name = None
             # print("Warning: Could not identify the man object.")


    def get_distance(self, loc1, loc2):
        """Looks up the shortest distance between two locations."""
        # Ensure locations exist in our precomputed distances
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
            # Locations are not connected or one/both are unknown.
            # Return infinity to indicate high cost/unreachability.
            return math.inf
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose goal nuts in the current state
        loose_goal_nuts = set()
        current_nut_locations = {} # {nut_name: location_name}
        current_spanner_locations = {} # {spanner_name: location_name} (on ground)
        current_usable_spanners = set() # {spanner_name}
        man_location = None
        spanners_carried = set() # {spanner_name}

        # Scan state to extract relevant facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "loose" and parts[1] in self.goal_nuts:
                loose_goal_nuts.add(parts[1])
            elif parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj == self.man_name:
                    man_location = loc
                # Check if obj is a nut (one of the initial nuts)
                if obj in self.initial_nut_names:
                     current_nut_locations[obj] = loc
                # Check if obj is a spanner on the ground (one of the initial spanners)
                elif obj in self.initial_spanner_names:
                      current_spanner_locations[obj] = loc

            elif parts[0] == "carrying":
                 carrier, obj = parts[1], parts[2]
                 if carrier == self.man_name:
                     spanners_carried.add(obj)
            elif parts[0] == "usable":
                 current_usable_spanners.add(parts[1])


        # 2. If U is empty, goal reached
        if not loose_goal_nuts:
            return 0

        # 3. Base cost: tighten actions
        h = len(loose_goal_nuts)

        # 4. Man's current location (already found)
        if man_location is None:
             # Man's location not found in state? Invalid state.
             return math.inf

        # 5. Usable spanners carried
        usable_spanners_carried = {s for s in spanners_carried if s in current_usable_spanners}

        # 6. Usable spanners available on ground
        usable_spanners_available_on_ground = {(s, loc) for s, loc in current_spanner_locations.items() if s in current_usable_spanners}

        # 7. Locations of loose goal nuts that are actually in the state
        nut_locations = {current_nut_locations[n] for n in loose_goal_nuts if n in current_nut_locations}

        # Handle case where nut locations are not found for loose goal nuts (shouldn't happen in valid states)
        if len(nut_locations) != len(loose_goal_nuts):
             # This implies a loose goal nut doesn't have an 'at' predicate in the state, or its location wasn't found.
             # Assuming valid states, this shouldn't occur.
             # If it did, these nuts are unreachable. Return infinity.
             return math.inf


        # 8. Check if man is carrying a usable spanner
        man_has_usable_spanner = len(usable_spanners_carried) > 0

        # 9. If man *is* carrying a usable spanner:
        if man_has_usable_spanner:
            # Man has spanner, needs to go to the closest nut
            min_dist_man_to_nut = math.inf
            for L_n in nut_locations:
                dist = self.get_distance(man_location, L_n)
                min_dist_man_to_nut = min(min_dist_man_to_nut, dist)

            if min_dist_man_to_nut == math.inf:
                 # Cannot reach any nut location from man's current location
                 return math.inf

            h += min_dist_man_to_nut

        # 10. If man is *not* carrying a usable spanner:
        else:
            # Man needs to get a spanner and go to a nut location
            min_cost_get_spanner_to_nut = math.inf

            if not usable_spanners_available_on_ground:
                # No usable spanners available anywhere, and man isn't carrying one.
                # Problem unsolvable from here.
                return math.inf

            # Find the minimum cost to get *a* spanner to *a* nut location
            for s, L_s in usable_spanners_available_on_ground:
                # Check if spanner location is reachable
                dist_man_to_spanner = self.get_distance(man_location, L_s)
                if dist_man_to_spanner == math.inf:
                     continue # Cannot reach this spanner

                for L_n in nut_locations:
                    # Check if nut location is reachable from spanner location
                    dist_spanner_to_nut = self.get_distance(L_s, L_n)
                    if dist_spanner_to_nut == math.inf:
                         continue # Cannot reach this nut from spanner location

                    # Cost = walk to spanner + pickup + walk spanner to nut
                    cost = dist_man_to_spanner + 1 + dist_spanner_to_nut
                    min_cost_get_spanner_to_nut = min(min_cost_get_spanner_to_nut, cost)

            if min_cost_get_spanner_to_nut == math.inf:
                 # Cannot get a spanner and bring it to any nut location
                 return math.inf

            h += min_cost_get_spanner_to_nut

        # 11. Return the calculated value of h
        return h
