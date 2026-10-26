from fnmatch import fnmatch
from collections import deque
import math # Import math for infinity

# Assume Heuristic base class is defined elsewhere and imported
# from heuristics.heuristic_base import Heuristic

# Helper functions (can be inside the class or outside)
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

# Assume Heuristic base class is defined elsewhere and imported
# If running standalone or base class is not provided, remove the inheritance
# class spannerHeuristic(Heuristic):
class spannerHeuristic:
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It uses a greedy simulation approach: for each loose goal nut, it estimates the cost
    to acquire a usable spanner (if not already carrying one) and walk to the nut's
    location, plus the cost of the tighten action. Spanners are assumed to be consumed
    after each use. The heuristic sums these estimated costs for each nut, processing
    them in alphabetical order by nut name.

    # Assumptions:
    - The man can carry at most one spanner at a time.
    - Spanners are consumed (become unusable) after tightening one nut.
    - The problem is solvable (enough usable spanners exist).
    - The names of the man, spanners, and nuts can be reliably identified from the initial state and goals.
    - Locations are connected by `link` predicates forming an undirected graph.

    # Heuristic Initialization
    - Parses static facts (`link`) and initial state/goals (`at`, `carrying`, `usable`, `loose`, `tightened`)
      to identify locations, the man, spanners, and nuts.
    - Builds a graph of locations based on `link` facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the set of goal nuts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all goal nuts that are currently `loose` (i.e., not `tightened`). If this set is empty, the goal is reached, and the heuristic is 0.
    2. Get the man's current location from the state. If the man's location cannot be found, the state is invalid, return infinity.
    3. Determine if the man is currently carrying a usable spanner based on the current state.
    4. Identify all usable spanners that were initially on the ground at some location. This set forms the pool of spanners available for pickup throughout the simulation.
    5. Store the initial locations of all spanners for quick lookup during the simulation.
    6. Initialize the total heuristic cost `h` to 0.
    7. Initialize the man's simulated current location to his actual current location.
    8. Initialize a flag indicating if the man is carrying a spanner in the simulation, based on the actual current state.
    9. Initialize a mutable set `sim_available_spanners_for_pickup` containing the names of usable spanners that were initially on the ground.
    10. Iterate through the loose goal nuts. Processing order might affect the estimate; for determinism, process them in alphabetical order by nut name. For each nut `n`:
        a. Add 1 to `h` for the `tighten_nut` action required for this nut.
        b. Get the location of nut `n` from the current state. If the nut's location cannot be found, the state is invalid, return infinity.
        c. Check if the man is currently carrying a spanner in the simulation (`sim_is_carrying`).
        d. If the man IS carrying a spanner (`sim_is_carrying` is True):
            i. Add the shortest distance from the man's simulated current location (`sim_man_loc`) to the nut's location (`nut_loc`) to `h` (cost to walk to the nut). Use precomputed distances. If unreachable, return infinity.
            ii. Update the man's simulated current location to the nut's location.
            iii. Set the simulation's "carrying spanner" flag (`sim_is_carrying`) to false, as the spanner is consumed by the tighten action.
        e. If the man is NOT carrying a spanner (`sim_is_carrying` is False):
            i. Check if there are any spanners left in the `sim_available_spanners_for_pickup` pool. If not, the state is likely unsolvable, return infinity.
            ii. Find the spanner `s` in `sim_available_spanners_for_pickup` whose initial location is closest to the man's simulated current location (`sim_man_loc`).
            iii. Get the initial location (`s_loc`) of the chosen spanner `s`.
            iv. Add the shortest distance from `sim_man_loc` to `s_loc` to `h` (cost to walk to the spanner). If unreachable, return infinity.
            v. Add 1 to `h` for the `pickup_spanner` action.
            vi. Add the shortest distance from `s_loc` to `nut_loc` to `h` (cost to walk from spanner location to nut location). If unreachable, return infinity.
            vii. Update the man's simulated current location to the nut's location.
            viii. Remove the picked-up spanner `s` from the `sim_available_spanners_for_pickup` pool.
            ix. Set the simulation's "carrying spanner" flag (`sim_is_carrying`) to false, as the newly picked-up spanner is consumed by the tighten action (cost already added in step 10a).
    11. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        self.locations = set()
        self.links = {} # Adjacency list {loc: {neighbor1, neighbor2}}
        self.dist = {} # Shortest path distances {loc1: {loc2: dist}}

        self.man_name = None
        self.all_spanners = set()
        self.all_nuts = set()
        self.goal_nuts = set()

        # Collect objects and infer types from initial state, goals, and static facts
        all_obj_locatable = set()
        all_obj_location = set()
        all_obj_man = set()
        all_obj_spanner = set()
        all_obj_nut = set()

        for fact in self.initial_state | self.goals | self.static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at':
                obj, loc = parts[1], parts[2]
                all_obj_locatable.add(obj)
                all_obj_location.add(loc)
            elif predicate == 'carrying':
                man, spanner = parts[1], parts[2]
                all_obj_man.add(man)
                all_obj_spanner.add(spanner)
            elif predicate == 'usable':
                spanner = parts[1]
                all_obj_spanner.add(spanner)
            elif predicate == 'loose':
                nut = parts[1]
                all_obj_nut.add(nut)
            elif predicate == 'tightened':
                nut = parts[1]
                all_obj_nut.add(nut)
                if fact in self.goals:
                    self.goal_nuts.add(nut)
            elif predicate == 'link':
                l1, l2 = parts[1], parts[2]
                all_obj_location.add(l1)
                all_obj_location.add(l2)

        self.all_spanners = all_obj_spanner
        self.all_nuts = all_obj_nut

        # Identify the man: assume the unique object identified as a potential man is the man.
        # Prioritize objects mentioned in 'carrying' predicate.
        men_from_carrying = {get_parts(fact)[1] for fact in self.initial_state | self.goals | self.static_facts if get_parts(fact)[0] == 'carrying'}

        if len(men_from_carrying) == 1:
             self.man_name = men_from_carrying.pop()
        elif len(all_obj_man) == 1: # If no carrying, but only one potential man from other predicates
             self.man_name = all_obj_man.pop()
        elif 'bob' in all_obj_man: # Fallback to 'bob' if ambiguous but present
             self.man_name = 'bob'
        elif len(all_obj_man) > 0: # Still ambiguous, pick one
             # print(f"Warning: Could not uniquely identify man from {all_obj_man}. Picking one.")
             self.man_name = next(iter(all_obj_man))
        else: # No potential man found
             # print("Error: Could not identify the man.")
             self.man_name = None


        self.locations = all_obj_location

        # Compute all-pairs shortest paths
        self._compute_shortest_paths()


    def _compute_shortest_paths(self):
        """Computes all-pairs shortest paths using BFS."""
        # Initialize distances with infinity
        self.dist = {loc: {other_loc: float('inf') for other_loc in self.locations} for loc in self.locations}

        for start_node in self.locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.dist[start_node][start_node] = 0

            while q:
                current_loc, current_dist = q.popleft()

                # Ensure current_loc is in links before accessing
                if current_loc in self.links:
                    for neighbor in self.links[current_loc]:
                        if neighbor in self.locations and neighbor not in visited: # Ensure neighbor is a known location
                            visited.add(neighbor)
                            self.dist[start_node][neighbor] = current_dist + 1
                            q.append((neighbor, current_dist + 1))


    def get_object_location(self, state, obj_name):
        """Finds the current location of an object in the given state."""
        # Man's location
        if obj_name == self.man_name:
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[1] == obj_name:
                    return parts[2]
        # Spanner's location (either carried or on ground)
        elif obj_name in self.all_spanners:
            # Check if carried by the man
            if self.man_name and f'(carrying {self.man_name} {obj_name})' in state:
                 # Spanner is at the man's location
                 return self.get_object_location(state, self.man_name)
            # Check if on the ground
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[1] == obj_name:
                    return parts[2]
        # Nut's location (always on ground)
        elif obj_name in self.all_nuts:
             for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[1] == obj_name:
                    return parts[2]

        # Should not happen for objects relevant to the heuristic in a valid state
        # print(f"Warning: Location not found for object {obj_name} in state.")
        return None # Indicate location not found


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose goal nuts
        # Sort alphabetically for deterministic heuristic value
        nuts_to_tighten = sorted([n for n in self.goal_nuts if f'(tightened {n})' not in state])

        if not nuts_to_tighten:
            return 0 # Goal reached

        # Ensure man was identified during initialization
        if self.man_name is None:
             return float('inf') # Cannot solve without a man

        # 2. Get man's current location
        current_man_loc = self.get_object_location(state, self.man_name)
        if current_man_loc is None:
             return float('inf') # Man's location not found - state is invalid

        # 3. Identify usable spanners initially carried or on ground
        # These sets represent the *pool* of spanners available throughout the plan
        usable_spanners_initially_carried = {s for s in self.all_spanners if f'(carrying {self.man_name} {s})' in self.initial_state}
        usable_spanners_initially_on_ground = {s for s in self.all_spanners if f'(usable {s})' in self.initial_state and f'(carrying {self.man_name} {s})' not in self.initial_state}

        # Store initial locations of all spanners for quick lookup in simulation
        initial_spanner_locations = {s: self.get_object_location(self.initial_state, s) for s in self.all_spanners if self.get_object_location(self.initial_state, s) is not None}


        # 4. Initialize simulation state
        h = 0
        sim_man_loc = current_man_loc
        # The simulation starts with the actual carrying status from the current state
        sim_is_carrying = any(f'(carrying {self.man_name} {s})' in state for s in self.all_spanners)

        # The pool of available spanners for pickup in the simulation starts with those initially on the ground.
        # Spanners initially carried are not available for pickup from the ground.
        sim_available_spanners_for_pickup = set(usable_spanners_initially_on_ground)

        # 5. Greedy simulation loop
        for nut_name in nuts_to_tighten:
            # Cost for tighten action
            h += 1

            nut_loc = self.get_object_location(state, nut_name)
            if nut_loc is None:
                 return float('inf') # Nut location not found - state is invalid

            # Need a spanner for this nut
            if sim_is_carrying:
                # Man is carrying a spanner in the simulation, use it
                # Cost is just walking to the nut location
                dist_to_nut = self.dist[sim_man_loc].get(nut_loc, float('inf'))
                if dist_to_nut == float('inf'): return float('inf') # Cannot reach nut
                h += dist_to_nut
                sim_man_loc = nut_loc
                sim_is_carrying = False # Spanner is consumed
            else:
                # Man is not carrying a spanner in the simulation, need to pick one up from the ground pool
                if not sim_available_spanners_for_pickup:
                    # No more usable spanners available for pickup from the initial ground set
                    return float('inf') # Unsolvable state

                # Find the nearest available usable spanner on the ground from the simulated location
                nearest_s = None
                min_dist_to_s = float('inf')
                nearest_s_loc = None

                # Iterate through spanners still available for pickup in the simulation
                for s in sim_available_spanners_for_pickup:
                    s_loc = initial_spanner_locations.get(s) # Get initial location
                    if s_loc is None: continue # Should not happen if initial locations were correctly mapped

                    dist_to_s = self.dist[sim_man_loc].get(s_loc, float('inf'))
                    if dist_to_s < min_dist_to_s:
                        min_dist_to_s = dist_to_s
                        nearest_s = s
                        nearest_s_loc = s_loc

                if nearest_s is None or min_dist_to_s == float('inf'):
                     # Cannot reach any available spanner on the ground
                     return float('inf')

                # Cost to walk to spanner, pickup, walk to nut
                h += min_dist_to_s # Walk to spanner
                h += 1 # Pickup spanner

                dist_spanner_to_nut = self.dist[nearest_s_loc].get(nut_loc, float('inf'))
                if dist_spanner_to_nut == float('inf'): return float('inf') # Cannot reach nut from spanner location
                h += dist_spanner_to_nut # Walk from spanner location to nut location

                # Update state for next nut calculation
                sim_man_loc = nut_loc
                sim_available_spanners_for_pickup.remove(nearest_s)
                sim_is_carrying = False # Spanner is consumed by tighten (accounted for by adding 1 earlier)

        return h
