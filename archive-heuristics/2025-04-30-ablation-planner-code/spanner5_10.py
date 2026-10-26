from fnmatch import fnmatch
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Perform Breadth-First Search to find shortest distances from a start node.

    Args:
        graph: Adjacency list representation of the graph {node: [neighbor1, neighbor2, ...]}
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from start_node.
    """
    distances = {start_node: 0}
    queue = [start_node]
    visited = {start_node}

    while queue:
        u = queue.pop(0) # Dequeue

        if u in graph: # Handle nodes with no outgoing links
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    distances[v] = distances[u] + 1
                    queue.append(v) # Enqueue
    return distances


class spannerHeuristic: # Inherit from Heuristic in the actual environment
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose
    goal nuts. It does this by calculating the cost to sequentially tighten each
    loose goal nut, always picking the next nut and available usable spanner
    that minimizes the immediate cost (walk to spanner, pickup, walk to nut, tighten).
    The cost includes walk actions (shortest path distance), pickup actions,
    and tighten actions.

    # Assumptions
    - There is only one man.
    - Nuts have fixed locations.
    - Spanners become unusable after one use.
    - The cost of each action (walk, pickup, tighten) is 1.
    - The heuristic assumes the man must pick up a spanner before tightening a nut
      if he is not already carrying a usable one.
    - The heuristic assumes the man moves directly from the spanner location to the nut location.
    - The heuristic assumes the man stays at the nut location after tightening it,
      which becomes the starting point for the next nut.

    # Heuristic Initialization
    - Identify the man object, nut objects, spanner objects, and location objects.
    - Extract the set of goal nuts that need to be tightened.
    - Build the location graph from `link` facts in `task.static`.
    - Precompute all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify the locations of all nuts and spanners.
    3. Identify which nuts are currently loose.
    4. Identify which spanners are currently usable.
    5. Determine if the man is currently carrying a spanner, and if it is usable.
    6. Filter the set of goal nuts to find those that are currently loose (`loose_goal_nuts_to_tighten`).
    7. Filter the set of usable spanners to find those not currently carried (`available_usable_spanners`).
    8. Initialize the total heuristic cost `h = 0`.
    9. Initialize the man's current location for the calculation (`current_man_location`) to his actual location in the state.
    10. Initialize the carrying status (`is_carrying_usable`) based on the state.
    11. Initialize the set of available usable spanners for the calculation (`remaining_usable_spanners_locations`).
    12. Initialize the set of nuts remaining to be tightened (`remaining_loose_goal_nuts`) to `loose_goal_nuts_to_tighten`.
    13. While there are nuts remaining in `remaining_loose_goal_nuts`:
        a. If the man is currently carrying a usable spanner (`is_carrying_usable` is True):
            i. Find the nut `n` in `remaining_loose_goal_nuts` whose location `l_n` is closest to `current_man_location` (using precomputed shortest paths).
            ii. Calculate the cost to tighten this nut: `cost = distance(current_man_location, l_n) + 1` (walk + tighten).
            iii. Set `is_carrying_usable` to False (spanner is used).
        b. If the man is not carrying a usable spanner (`is_carrying_usable` is False):
            i. Find the pair of (nut `n`, spanner `s`) where `n` is in `remaining_loose_goal_nuts` and `s` is in `remaining_usable_spanners_locations` that minimizes the cost: `distance(current_man_location, l_s) + 1 + distance(l_s, l_n) + 1` (walk to spanner + pickup + walk to nut + tighten).
            ii. If no usable spanners are available, return a large number (indicating likely unsolvability).
            iii. Calculate the minimum cost found: `cost = distance(current_man_location, l_s_best) + 1 + distance(l_s_best, l_n_best) + 1`.
            iv. Remove the chosen spanner `s_best` from `remaining_usable_spanners_locations`.
            v. Set `is_carrying_usable` to False (spanner is used).
            vi. Set `n` to `n_best`.
        c. Add the calculated `cost` for this nut to the total heuristic `h`.
        d. Update `current_man_location` to `l_n`.
        e. Remove nut `n` from `remaining_loose_goal_nuts`.
    14. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, object types, and computing shortest paths."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        objects = task.objects # Objects defined in the problem instance

        # Identify object types
        self.man_obj = None
        self.nut_objs = set()
        self.spanner_objs = set()
        self.location_objs = set()

        for obj_def in objects:
            parts = obj_def.split(' - ')
            if len(parts) == 2:
                obj_name, obj_type = parts
                if obj_type == 'man':
                    self.man_obj = obj_name
                elif obj_type == 'nut':
                    self.nut_objs.add(obj_name)
                elif obj_type == 'spanner':
                    self.spanner_objs.add(obj_name)
                elif obj_type == 'location':
                    self.location_objs.add(obj_name)

        # Store goal nuts (nuts that need to be tightened)
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened" and args and args[0] in self.nut_objs:
                self.goal_nuts.add(args[0])

        # Build location graph from link facts
        self.links = {}
        # Ensure all locations are keys in the graph, even if they have no links
        for loc in self.location_objs:
             self.links[loc] = []

        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations are valid objects before adding link
                if loc1 in self.location_objs and loc2 in self.location_objs:
                    self.links[loc1].append(loc2)
                    self.links[loc2].append(loc1) # Links are bidirectional

        # Compute all-pairs shortest paths
        self.distance = {}
        all_locations = list(self.location_objs)
        for start_loc in all_locations:
             self.distance[start_loc] = bfs(self.links, start_loc)

        # Fill in unreachable distances with infinity
        for loc1 in all_locations:
            for loc2 in all_locations:
                if loc2 not in self.distance[loc1]:
                    self.distance[loc1][loc2] = math.inf # Use infinity for unreachable

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Parse state to get current facts
        man_location = None
        nut_locations = {} # nut_name -> location
        spanner_locations = {} # spanner_name -> location
        loose_nuts = set() # set of nut_names
        usable_spanners = set() # set of spanner_names
        carried_spanner = None # spanner_name or None

        # Pre-populate locations for all objects from __init__ to handle cases
        # where an object might not have an (at) fact in the current state
        # (e.g., spanner being carried).
        # We will update these with actual state facts.
        for nut in self.nut_objs: nut_locations[nut] = None
        for spanner in self.spanner_objs: spanner_locations[spanner] = None


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at":
                if len(args) == 2:
                    obj, loc = args
                    if obj == self.man_obj:
                        man_location = loc
                    elif obj in self.nut_objs:
                        nut_locations[obj] = loc
                    elif obj in self.spanner_objs:
                        spanner_locations[obj] = loc
            elif predicate == "loose":
                if len(args) == 1 and args[0] in self.nut_objs:
                    loose_nuts.add(args[0])
            elif predicate == "usable":
                 if len(args) == 1 and args[0] in self.spanner_objs:
                    usable_spanners.add(args[0])
            elif predicate == "carrying":
                 if len(args) == 2 and args[0] == self.man_obj and args[1] in self.spanner_objs:
                    carried_spanner = args[1]

        # Identify loose goal nuts that still need tightening
        loose_goal_nuts_to_tighten = self.goal_nuts.intersection(loose_nuts)

        # If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts_to_tighten:
            return 0

        # Identify usable spanners that are not currently carried
        available_usable_spanners_locations = {
            s: spanner_locations[s]
            for s in usable_spanners
            if s != carried_spanner and spanner_locations.get(s) is not None # Must have a known location if not carried
        }

        # --- Heuristic Calculation ---
        h = 0
        current_man_location = man_location
        is_carrying_usable = (carried_spanner is not None) and (carried_spanner in usable_spanners)
        remaining_usable_spanners_locations = available_usable_spanners_locations.copy()
        remaining_loose_goal_nuts = loose_goal_nuts_to_tighten.copy()

        # Use infinity for unreachable locations or insufficient spanners
        UNREACHABLE_COST = math.inf

        # Check if man location is known and reachable from/to other locations
        if current_man_location is None or current_man_location not in self.distance:
             return UNREACHABLE_COST # Man is nowhere or in a disconnected component

        while remaining_loose_goal_nuts:
            best_nut = None
            best_spanner = None # Only relevant if not carrying usable
            min_cost_segment = UNREACHABLE_COST # Initialize with a large value

            if is_carrying_usable:
                # Find the nearest loose goal nut
                for nut in remaining_loose_goal_nuts:
                    nut_loc = nut_locations.get(nut)
                    # Check if nut location is known and reachable
                    if nut_loc is None or self.distance[current_man_location][nut_loc] == math.inf:
                         continue # Skip this nut, it's unreachable

                    walk_cost = self.distance[current_man_location][nut_loc]
                    cost_segment = walk_cost + 1 # walk + tighten

                    if cost_segment < min_cost_segment:
                        min_cost_segment = cost_segment
                        best_nut = nut

                if best_nut is None:
                     # No reachable loose goal nuts while carrying a usable spanner
                     return UNREACHABLE_COST # State is likely stuck/unsolvable

                # Apply the cost for the chosen nut
                h += min_cost_segment
                current_man_location = nut_locations[best_nut]
                is_carrying_usable = False # Spanner is used
                remaining_loose_goal_nuts.remove(best_nut)

            else: # Not carrying a usable spanner
                # Find the best (nut, spanner) pair to minimize cost
                if not remaining_usable_spanners_locations:
                    # No usable spanners left but nuts remain
                    return UNREACHABLE_COST # State is unsolvable

                for nut in remaining_loose_goal_nuts:
                    nut_loc = nut_locations.get(nut)
                    if nut_loc is None: continue # Skip nut with unknown location

                    for spanner, spanner_loc in remaining_usable_spanners_locations.items():
                         # Check reachability: man to spanner, spanner to nut
                         if self.distance[current_man_location][spanner_loc] == math.inf or self.distance[spanner_loc][nut_loc] == math.inf:
                             continue # Skip this spanner/nut pair if path is unreachable

                         walk_cost_1 = self.distance[current_man_location][spanner_loc]
                         pickup_cost = 1
                         walk_cost_2 = self.distance[spanner_loc][nut_loc]
                         tighten_cost = 1

                         cost_segment = walk_cost_1 + pickup_cost + walk_cost_2 + tighten_cost

                         if cost_segment < min_cost_segment:
                             min_cost_segment = cost_segment
                             best_nut = nut
                             best_spanner = spanner

                if best_nut is None:
                     # No reachable (nut, spanner) pair
                     return UNREACHABLE_COST # State is likely stuck/unsolvable

                # Apply the cost for the chosen nut and spanner
                h += min_cost_segment
                current_man_location = nut_locations[best_nut]
                # is_carrying_usable remains False after tightening
                remaining_loose_goal_nuts.remove(best_nut)
                del remaining_usable_spanners_locations[best_spanner]

        # If we successfully planned for all nuts, return the total cost
        return h
