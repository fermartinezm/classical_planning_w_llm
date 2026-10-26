import math
from collections import deque
from fnmatch import fnmatch
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
    # Ensure we don't try to match more args than parts in the fact
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start):
    """
    Perform a Breadth-First Search to find shortest distances from a start node.

    Args:
        graph: Adjacency list representation of the graph {node: [neighbor1, ...]}
        start: The starting node for the BFS.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from start.
    """
    distances = {node: math.inf for node in graph}
    distances[start] = 0
    queue = deque([start])

    while queue:
        u = queue.popleft()
        if u in graph: # Handle potential nodes in distances not in graph (e.g. types)
            for v in graph[u]:
                if distances[v] == math.inf:
                    distances[v] = distances[u] + 1
                    queue.append(v)
    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It considers the cost of tightening each loose goal nut sequentially,
    greedily choosing the closest nut and, if a spanner is needed, the closest
    available usable spanner. The cost for each nut includes walking to the
    spanner (if needed), picking it up, walking to the nut, and tightening it.

    # Assumptions
    - Nuts do not move. Their locations are static.
    - Spanners can be picked up and moved by the man.
    - A usable spanner is consumed after one tightening action.
    - The man can carry multiple spanners.
    - The graph of locations connected by 'link' predicates is undirected.

    # Heuristic Initialization
    - Build the graph of locations based on 'link' facts.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify all nuts that are goals.
    - Store the static locations of all nuts.
    - Identify the man object and all spanner objects.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify which goal nuts are currently loose.
    3. Identify which spanners are currently usable.
    4. Identify which usable spanners the man is currently carrying.
    5. Calculate the number of loose goal nuts (K).
    6. Calculate the number of usable spanners the man is carrying (C).
    7. Calculate the number of usable spanners available on the ground (A).
    8. If K is 0, the heuristic is 0 (goal state).
    9. If C + A < K, the problem is unsolvable with available resources, return infinity.
    10. Initialize heuristic value (h) to 0.
    11. Set the man's current location for calculation (`current_loc`) to his actual current location.
    12. Set the number of usable spanners the man has for calculation (`num_carried_usable`) to C.
    13. Create a list of usable spanners available on the ground.
    14. Get the locations of all loose goal nuts.
    15. Sort the loose goal nuts based on their distance from the man's initial location (`current_loc`). This defines the greedy order in which nuts are processed by the heuristic.
    16. Iterate through the sorted loose goal nuts:
        a. Get the location of the current nut (`loc_n`).
        b. If `num_carried_usable > 0`:
            i. Decrement `num_carried_usable`.
            ii. The cost for this nut is the distance from `current_loc` to `loc_n` (walk) + 1 (tighten).
            iii. Add this cost to h.
            iv. Update `current_loc` to `loc_n`.
        c. Else (`num_carried_usable == 0`, need to pick up a spanner):
            i. Find the closest available usable spanner (from the list created in step 13) to the current man location (`current_loc`).
            ii. Get the location of this closest spanner (`loc_s`).
            iii. The cost for this nut is the distance from `current_loc` to `loc_s` (walk to spanner) + 1 (pickup) + distance from `loc_s` to `loc_n` (walk to nut) + 1 (tighten).
            iv. Add this cost to h.
            v. Update `current_loc` to `loc_n`.
            vi. Remove the chosen spanner from the list of available usable spanners.
    17. Return the total heuristic value h.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information and computing distances."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Build location graph from link facts
        self.location_graph = {}
        all_locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1)

        # Ensure all locations mentioned in initial state or goals are in the graph nodes
        # even if they have no links (isolated locations)
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 all_locations.add(loc)
                 self.location_graph.setdefault(loc, []) # Add location node if not exists

        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 all_locations.add(loc)
                 self.location_graph.setdefault(loc, []) # Add location node if not exists


        # Compute all-pairs shortest paths
        self.dist = {}
        for start_loc in all_locations:
            self.dist[start_loc] = bfs(self.location_graph, start_loc)

        # Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Store static nut locations
        self.nut_locations = {}
        for fact in static_facts:
            if match(fact, "at", "*", "*"):
                 obj_type = None
                 # Find object type from initial state facts
                 for init_fact in initial_state:
                     if match(init_fact, get_parts(fact)[1], "*"): # Check if the object name matches
                         # This is a bit hacky, assumes type predicates are like (type object)
                         # A more robust parser would be better, but let's stick to the example style
                         # Look for type facts like (man bob), (spanner spanner1), (nut nut1)
                         # We need to find the object's type based on its name
                         # Let's iterate through initial state facts to find the type predicate
                         for type_fact in initial_state:
                             type_parts = get_parts(type_fact)
                             if len(type_parts) == 2 and type_parts[1] == get_parts(fact)[1]:
                                 obj_type = type_parts[0]
                                 break
                         if obj_type == 'nut':
                             self.nut_locations[get_parts(fact)[1]] = get_parts(fact)[2]
                             break # Found nut location, move to next static fact

        # Identify the man object
        self.man = None
        for fact in initial_state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == 'man': # Assuming type facts like (man bob)
                 self.man = parts[1]
                 break
            # Alternative: find object in (at ?m - man ?l - location)
            if match(fact, "at", "*", "*"):
                 obj, loc = parts[1], parts[2]
                 # Check if this object is a man type. This requires knowing types.
                 # A simpler approach is to assume there's only one man and find the object in (at man_obj loc)
                 # Let's rely on the (man bob) type fact if available, otherwise try to infer from 'at'
                 # Given the example, (man bob) exists.
                 pass # Man found above

        # Identify all spanner objects
        self.all_spanners = set()
        for fact in initial_state:
             parts = get_parts(fact)
             if len(parts) == 2 and parts[0] == 'spanner': # Assuming type facts like (spanner spanner1)
                 self.all_spanners.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # This should not happen in a valid planning state, but handle defensively
             return math.inf # Man is not located anywhere?

        # Find loose goal nuts
        loose_goal_nuts = {n for n in self.goal_nuts if '(loose ' + n + ')' in state}

        # Find usable spanners
        usable_spanners = {s for s in self.all_spanners if '(usable ' + s + ')' in state}

        # Find carried spanners
        carried_spanners = {s for s in self.all_spanners if '(carrying ' + self.man + ' ' + s + ')' in state}

        # Usable carried spanners
        carried_usable_spanners = usable_spanners.intersection(carried_spanners)

        # Usable available spanners (on ground)
        available_usable_spanners = usable_spanners - carried_spanners

        # Find current locations of available usable spanners
        available_spanner_locs = {}
        for s in available_usable_spanners:
            for fact in state:
                if match(fact, "at", s, "*"):
                    available_spanner_locs[s] = get_parts(fact)[2]
                    break # Found location for this spanner

        K = len(loose_goal_nuts)
        C = len(carried_usable_spanners)
        A = len(available_usable_spanners)

        # If no nuts need tightening, goal is reached
        if K == 0:
            return 0

        # If not enough usable spanners exist in total
        if C + A < K:
            return math.inf # Unsolvable

        h = 0
        current_loc = man_loc
        num_carried_usable = C
        remaining_available_spanners = list(available_usable_spanners) # Make a mutable list

        # Sort loose goal nuts by distance from the man's current location
        sorted_nuts = sorted(loose_goal_nuts, key=lambda n: self.dist[current_loc].get(self.nut_locations[n], math.inf))

        # Process nuts in greedy order
        for nut in sorted_nuts:
            loc_n = self.nut_locations[nut]

            # Check if nut location is reachable
            if self.dist[current_loc].get(loc_n, math.inf) == math.inf:
                 return math.inf # Nut location is unreachable

            if num_carried_usable > 0:
                # Use a carried spanner
                num_carried_usable -= 1
                # Cost is walk to nut + tighten
                cost = self.dist[current_loc][loc_n] + 1
                h += cost
                current_loc = loc_n # Man is now at the nut location

            else:
                # Need to pick up a spanner
                if not remaining_available_spanners:
                     # This case should be caught by C + A < K check, but defensive
                     return math.inf # No available spanners left

                # Find the closest available usable spanner to the current location
                closest_spanner = None
                min_dist_to_spanner = math.inf
                for s in remaining_available_spanners:
                    loc_s = available_spanner_locs[s]
                    # Check if spanner location is reachable
                    if self.dist[current_loc].get(loc_s, math.inf) == math.inf:
                         continue # Skip unreachable spanners

                    dist_to_spanner = self.dist[current_loc][loc_s]
                    if dist_to_spanner < min_dist_to_spanner:
                        min_dist_to_spanner = dist_to_spanner
                        closest_spanner = s

                if closest_spanner is None:
                     # No reachable available spanners
                     return math.inf

                loc_s = available_spanner_locs[closest_spanner]

                # Cost is walk to spanner + pickup + walk to nut + tighten
                # Check if path from spanner to nut is reachable
                if self.dist[loc_s].get(loc_n, math.inf) == math.inf:
                     return math.inf # Nut location is unreachable from spanner location

                cost = self.dist[current_loc][loc_s] + 1 + self.dist[loc_s][loc_n] + 1
                h += cost
                current_loc = loc_n # Man is now at the nut location
                remaining_available_spanners.remove(closest_spanner) # Spanner is now used

        return h
