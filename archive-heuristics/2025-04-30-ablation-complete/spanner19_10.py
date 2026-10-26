from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is available

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

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It uses a greedy strategy: the man repeatedly either goes to the nearest usable
    spanner on the ground (if not carrying one) or goes to the nearest loose nut
    (if carrying a usable spanner), performs the necessary action (pickup or tighten),
    and updates his state and location. The total cost is the sum of walk, pickup,
    and tighten actions in this greedy sequence.

    # Assumptions
    - There is a single man agent.
    - Spanners are single-use (become unusable after tightening one nut).
    - Nuts and their locations are static.
    - Location links are static and bidirectional.
    - The problem is solvable (i.e., there are enough usable spanners initially
      and all relevant locations are reachable). The heuristic returns infinity
      if a needed item is unreachable.

    # Heuristic Initialization
    - Parse static facts to build the location graph based on `link` predicates.
    - Identify all locations, the man, all spanners, and all nuts by inspecting
      static facts, initial state, and goals.
    - Compute all-pairs shortest path distances between all known locations
      using Breadth-First Search (BFS). These distances represent the minimum
      number of `walk` actions required to move between locations.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic simulates a greedy plan execution:

    1.  Identify the man's current location, the set of loose nuts and their locations,
        the set of usable spanners on the ground and their locations, and whether
        the man is currently carrying a usable spanner.
    2.  Initialize the total estimated cost `h` to 0.
    3.  Enter a loop that continues as long as there are loose nuts remaining:
        a.  If the man is currently carrying a usable spanner:
            i.  Find the loose nut whose location is nearest to the man's current location,
                using the precomputed shortest path distances.
            ii. If no loose nuts are reachable, the problem is unsolvable; return infinity.
            iii. Add the distance (number of walk actions) to the nearest nut's location to `h`.
            iv. Add 1 to `h` for the `tighten_nut` action.
            v.  Update the man's current location to the location of the tightened nut.
            vi. The spanner is now used; mark the man as no longer carrying a usable spanner.
            vii. Remove the tightened nut from the set of loose nuts that still need tightening.
        b.  If the man is NOT currently carrying a usable spanner:
            i.  Find the usable spanner on the ground whose location is nearest to the man's
                current location, using the precomputed shortest path distances.
            ii. If no usable spanners are available on the ground, but there are still
                loose nuts, the problem is unsolvable; return infinity.
            iii. Add the distance (number of walk actions) to the nearest spanner's location to `h`.
            iv. Add 1 to `h` for the `pickup_spanner` action.
            v.  Update the man's current location to the location where the spanner was picked up.
            vi. Mark the man as now carrying a usable spanner.
            vii. Remove the picked-up spanner from the set of usable spanners on the ground.
    4.  Once the loop finishes (all loose nuts are tightened), return the accumulated cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and computing distances.
        """
        self.goals = task.goals
        self.static_facts = task.static

        self.all_locations = set()
        self.location_graph = {} # Adjacency list {loc: [neighbor1, neighbor2, ...]}
        self.all_spanners = set()
        self.all_nuts = set()
        self.man = None

        # Parse static facts to build the location graph and find locations
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                l1, l2 = parts[1], parts[2]
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1) # Links are bidirectional
                self.all_locations.add(l1)
                self.all_locations.add(l2)

        # Identify all objects and their types from initial state and goals
        # This assumes objects appear in initial state or goals if they exist
        objects_with_potential_types = {}
        for fact in task.initial_state | task.goals:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]

            if pred == "at":
                obj, loc = args
                # Infer type based on typical predicates
                if obj.startswith('bob'): objects_with_potential_types[obj] = objects_with_potential_types.get(obj, set()) | {'man'}
                elif obj.startswith('spanner'): objects_with_potential_types[obj] = objects_with_potential_types.get(obj, set()) | {'spanner'}
                elif obj.startswith('nut'): objects_with_potential_types[obj] = objects_with_potential_types.get(obj, set()) | {'nut'}
                else: objects_with_potential_types[obj] = objects_with_potential_types.get(obj, set()) | {'locatable'} # Default

                self.all_locations.add(loc) # Ensure all locations mentioned are included

            elif pred == "carrying":
                man, spanner = args
                objects_with_potential_types[man] = objects_with_potential_types.get(man, set()) | {'man'}
                objects_with_potential_types[spanner] = objects_with_potential_types.get(spanner, set()) | {'spanner'}
            elif pred == "usable":
                spanner = args[0]
                objects_with_potential_types[spanner] = objects_with_potential_types.get(spanner, set()) | {'spanner'}
            elif pred == "tightened" or pred == "loose":
                nut = args[0]
                objects_with_potential_types[nut] = objects_with_potential_types.get(nut, set()) | {'nut'}

        # Assign objects to sets based on inferred types
        for obj, types in objects_with_potential_types.items():
            if 'man' in types:
                self.man = obj # Assuming one man
            if 'spanner' in types:
                self.all_spanners.add(obj)
            if 'nut' in types:
                self.all_nuts.add(obj)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = {}
            queue = deque([start_loc])
            dist = {start_loc: 0}
            visited = {start_loc}

            while queue:
                u = queue.popleft()
                self.distances[start_loc][u] = dist[u] # Store distance

                # Ensure u is a key in location_graph before iterating
                if u in self.location_graph:
                    for v in self.location_graph[u]:
                        if v not in visited:
                            visited.add(v)
                            dist[v] = dist[u] + 1
                            queue.append(v)

    def find_nearest_item(self, from_loc, target_items_with_loc):
        """
        Find the item (and its location) from target_items_with_loc that is nearest to from_loc.
        Returns (nearest_item, nearest_loc, min_dist) or (None, None, float('inf')).
        target_items_with_loc is a list of (item, location) tuples.
        """
        min_dist = float('inf')
        nearest_item = None
        nearest_loc = None

        for item, loc in target_items_with_loc:
            # Get distance from precomputed table, default to inf if unreachable
            d = self.distances.get(from_loc, {}).get(loc, float('inf'))
            if d < min_dist:
                min_dist = d
                nearest_item = item
                nearest_loc = loc

        return nearest_item, nearest_loc, min_dist


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Extract relevant information from the current state
        current_man_loc = None
        loose_nuts_with_loc = {} # {nut: loc}
        usable_spanners_on_ground_with_loc = {} # {spanner: loc}
        is_carrying_usable = False

        # Temporary storage to find locations of all nuts and spanners first
        nut_locations_in_state = {}
        spanner_locations_in_state = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj == self.man:
                    current_man_loc = loc
                elif obj in self.all_nuts:
                    nut_locations_in_state[obj] = loc
                elif obj in self.all_spanners:
                    spanner_locations_in_state[obj] = loc
            elif parts[0] == "carrying":
                m, s = parts[1], parts[2]
                if m == self.man and (f'(usable {s})' in state): # Check if the carried spanner is usable
                    is_carrying_usable = True

        # Populate loose_nuts_with_loc and usable_spanners_on_ground_with_loc
        for nut in self.all_nuts:
            if (f'(loose {nut})' in state) and (nut in nut_locations_in_state):
                 loose_nuts_with_loc[nut] = nut_locations_in_state[nut]

        for spanner in self.all_spanners:
             if (f'(usable {spanner})' in state) and (spanner in spanner_locations_in_state):
                 usable_spanners_on_ground_with_loc[spanner] = spanner_locations_in_state[spanner]


        h = 0 # Initialize heuristic cost

        # Simulate the greedy plan
        while loose_nuts_with_loc:
            if is_carrying_usable:
                # Need to go to a nut
                target_nuts_list = list(loose_nuts_with_loc.items()) # Convert dict items to list of (nut, loc) tuples
                _, nearest_nut_loc, dist_to_nut = self.find_nearest_item(current_man_loc, target_nuts_list)

                if dist_to_nut == float('inf'):
                    # Cannot reach any loose nut
                    return float('inf')

                h += dist_to_nut # Walk cost
                h += 1 # Tighten cost
                current_man_loc = nearest_nut_loc

                # Find and remove one nut at the nearest location
                nut_to_remove = None
                for nut, loc in loose_nuts_with_loc.items():
                    if loc == nearest_nut_loc:
                        nut_to_remove = nut
                        break
                if nut_to_remove:
                    del loose_nuts_with_loc[nut_to_remove]

                is_carrying_usable = False # Spanner is used up

            else: # Need a spanner
                # Go to nearest usable spanner on ground
                target_spanners_list = list(usable_spanners_on_ground_with_loc.items()) # Convert dict items to list of (spanner, loc) tuples
                if not target_spanners_list:
                    # No usable spanners left on the ground, but still loose nuts. Unsolvable.
                    return float('inf')

                nearest_spanner, nearest_spanner_loc, dist_to_spanner = self.find_nearest_item(current_man_loc, target_spanners_list)

                if dist_to_spanner == float('inf'):
                    # Cannot reach any usable spanner
                    return float('inf')

                h += dist_to_spanner # Walk cost
                h += 1 # Pickup cost
                current_man_loc = nearest_spanner_loc

                # Remove the picked-up spanner from the ground set
                if nearest_spanner in usable_spanners_on_ground_with_loc:
                     del usable_spanners_on_ground_with_loc[nearest_spanner]

                is_carrying_usable = True # Now carrying a usable spanner

        return h

