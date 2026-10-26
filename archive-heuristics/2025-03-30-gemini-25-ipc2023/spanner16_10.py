import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For infinity

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    if not fact or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to match facts against patterns
def match(fact, *args):
    """Check if a PDDL fact matches a pattern (supports '*' wildcard)."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts
    that are currently loose. It simulates a sequential process where the single man
    fetches usable spanners and travels to nut locations to tighten them one by one.
    Each tightening action consumes one usable spanner. The heuristic aims for accuracy
    over admissibility, suitable for Greedy Best-First Search.

    # Assumptions
    - There is exactly one man object in the problem instance.
    - The goal consists solely of a conjunction of `(tightened ?n)` predicates.
    - `(link ?l1 ?l2)` predicates define a static, undirected graph of locations.
    - Each `tighten_nut` action consumes one usable spanner, making it unusable.
    - All necessary objects (man, spanners, nuts, locations) are implicitly defined
      or discoverable from the initial state and static facts.

    # Heuristic Initialization
    - Extracts the set of nuts that need to be tightened from `task.goals`.
    - Identifies the unique man object based on initial state predicates like `(at ?m ?l)`
      or `(carrying ?m ?s)`, assuming it's not a spanner or nut.
    - Parses static `(link ?l1 ?l2)` facts to build an adjacency list representation
      of the location graph.
    - Precomputes all-pairs shortest path distances between all known locations using
      Breadth-First Search (BFS). Stores distances in `self.distances`. If locations
      are disconnected, the distance is infinity.
    - Identifies all spanner and nut objects from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract the current location of the man (`man_loc`),
        the spanner the man is carrying (`carried_spanner`, if any), the location
        (`loc`) and usability (`usable`) status for every spanner, and the location
        (`loc`) and looseness (`loose`) status for every nut. Also identify nuts
        already tightened.
    2.  **Identify Remaining Goals:** Determine the set of `goal_nuts` that are still
        `loose` according to the current state (`remaining_loose_goal_nuts`). If this
        set is empty, the heuristic value is 0, as the goal state is reached.
    3.  **Identify Available Resources:** Find all spanners that are currently `usable`.
        This includes spanners on the ground and the one potentially carried by the man.
    4.  **Check Resource Availability:** If the number of `remaining_loose_goal_nuts`
        exceeds the total number of `usable` spanners currently available, the goal
        might be unreachable. The simulation proceeds but will eventually run out of
        spanners, incurring a penalty.
    5.  **Simulate Tightening Sequence:**
        a. Initialize heuristic cost `h = 0`.
        b. Maintain the simulated state: man's location (`current_man_loc`), whether
           the carried spanner is usable (`sim_carried_spanner_is_usable`), the set
           of usable spanners on the ground (`sim_usable_spanners_on_ground`), and
           the set of nuts still needing tightening (`sim_nuts_to_tighten`).
        c. **Loop** while `sim_nuts_to_tighten` is not empty:
            i.   Check if any usable spanner is available (carried or on ground). If not, break the loop (add penalty later).
            ii.  **If man is carrying a usable spanner:**
                 - Find the nut `n` in `sim_nuts_to_tighten` that is closest to `current_man_loc`. Handle unreachable nuts.
                 - Add travel cost (`distance(current_man_loc -> nut_loc)`) to `h`.
                 - Add tighten cost (1 action) to `h`.
                 - Update `current_man_loc` to the nut's location.
                 - Mark the carried spanner as no longer usable in the simulation.
                 - Remove `n` from `sim_nuts_to_tighten`.
            iii. **If man needs a usable spanner:**
                 - Find the usable spanner `s` on the ground that is closest to `current_man_loc`. Handle unreachable spanners.
                 - Add travel cost (`distance(current_man_loc -> spanner_loc)`) to `h`.
                 - Add pickup cost (1 action) to `h`.
                 - Update `current_man_loc` to the spanner's location.
                 - Remove `s` from the simulation's available ground spanners.
                 - *Now the man holds spanner `s`.* Find the nut `n` in `sim_nuts_to_tighten` closest to the *new* `current_man_loc` (the spanner's location). Handle unreachable nuts.
                 - Add travel cost (`distance(spanner_loc -> nut_loc)`) to `h`.
                 - Add tighten cost (1 action) to `h`.
                 - Update `current_man_loc` to the nut's location.
                 - Mark spanner `s` as no longer usable in the simulation.
                 - Remove `n` from `sim_nuts_to_tighten`.
            iv.  If required objects (nuts/spanners) are unreachable at any step, add a large penalty and break the simulation.
    6.  **Apply Penalty (If Needed):** If the loop terminated because no more usable
        spanners were available or objects were unreachable, but `sim_nuts_to_tighten`
        was still not empty, add a penalty to `h`. A simple penalty is `2 * len(sim_nuts_to_tighten)`,
        estimating at least a pickup and tighten action are needed per remaining nut (which are now impossible).
    7.  **Return `h`**: The final calculated integer heuristic value.
    """
    LARGE_PENALTY_FACTOR = 100 # Factor for penalizing unreachable states/objects
    INFINITY = float('inf')

    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # --- Identify Objects (Man, Spanners, Nuts, Locations) ---
        self.locations = set()
        self.spanners = set()
        self.nuts = set()
        self.the_man = None

        all_facts = task.initial_state | task.static

        # Pass 1: Discover locations from links and 'at' predicates
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            if pred == 'link':
                self.locations.add(args[0])
                self.locations.add(args[1])
            elif pred == 'at':
                # Ensure the location part is added to the set of locations
                if len(args) > 1:
                    self.locations.add(args[1])

        # Pass 2: Discover spanners, nuts, and man candidates from initial state
        man_candidates = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             args = parts[1:]
             if pred == 'at':
                 obj = args[0]
                 # Use simple name check first (adjust if names vary significantly)
                 if 'spanner' in obj: self.spanners.add(obj)
                 elif 'nut' in obj: self.nuts.add(obj)
                 # If it's at a known location and not spanner/nut, consider it for man
                 elif len(args) > 1 and args[1] in self.locations:
                     man_candidates.add(obj)
             elif pred == 'usable': self.spanners.add(args[0])
             elif pred == 'loose': self.nuts.add(args[0])
             elif pred == 'carrying':
                 man, spanner = args
                 man_candidates.add(man)
                 self.spanners.add(spanner)

        # Identify the single man by excluding known spanners and nuts
        actual_man_candidates = man_candidates - self.spanners - self.nuts
        if len(actual_man_candidates) == 1:
            self.the_man = actual_man_candidates.pop()
        elif len(actual_man_candidates) > 1:
             # Fallback: Check for common names or just pick one
             if 'bob' in actual_man_candidates: self.the_man = 'bob'
             else: self.the_man = list(actual_man_candidates)[0]
        elif 'bob' in man_candidates: # Check original candidates if filtering failed
             self.the_man = 'bob'
        else:
             # Last resort: find any object 'at' a location not identified as spanner/nut
             found_man = False
             for fact in task.initial_state:
                 if match(fact, 'at', '*', '*'):
                     obj = get_parts(fact)[1]
                     if obj not in self.spanners and obj not in self.nuts and obj not in self.locations:
                         self.the_man = obj
                         found_man = True
                         break
             if not found_man:
                 raise ValueError("Could not uniquely identify the man object in the task.")

        # --- Goal Nuts ---
        self.goal_nuts = {get_parts(g)[1] for g in self.goals if match(g, "tightened", "*")}

        # --- Build Location Graph & Compute Distances ---
        adj = collections.defaultdict(list)
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                # Ensure locations from links are in the set (already done in Pass 1)
                adj[loc1].append(loc2)
                adj[loc2].append(loc1)

        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: self.INFINITY))

        if not self.locations:
             pass # No locations, distances remain infinite except for self (dist=0)

        for start_node in self.locations:
            self.distances[start_node][start_node] = 0
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                current_node, dist = queue.popleft()
                # Check neighbors only if the current node is actually linked
                if current_node in adj:
                    for neighbor in adj[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[start_node][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

    def _get_current_state_info(self, state):
        """ Parses the state frozenset to extract relevant information. """
        man_loc = None
        carried_spanner = None
        # Initialize states for all known objects
        spanner_states = {s: {'loc': None, 'usable': False} for s in self.spanners}
        nut_states = {n: {'loc': None, 'loose': False} for n in self.nuts}
        tightened_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            if pred == 'at':
                obj, loc = args
                if obj == self.the_man: man_loc = loc
                elif obj in self.spanners: spanner_states[obj]['loc'] = loc
                elif obj in self.nuts: nut_states[obj]['loc'] = loc
            elif pred == 'carrying':
                man, spanner = args
                # Ensure the carrying fact involves the identified man and a known spanner
                if man == self.the_man and spanner in self.spanners:
                    carried_spanner = spanner
                    spanner_states[spanner]['loc'] = self.the_man # Mark as carried
            elif pred == 'usable':
                if args[0] in self.spanners: spanner_states[args[0]]['usable'] = True
            elif pred == 'loose':
                if args[0] in self.nuts: nut_states[args[0]]['loose'] = True
            elif pred == 'tightened':
                 if args[0] in self.nuts: tightened_nuts.add(args[0])

        # Post-process: Ensure tightened nuts are not marked as loose
        for nut in tightened_nuts:
             if nut in nut_states: nut_states[nut]['loose'] = False

        # Ensure all goal nuts have a state entry, default to not loose if not found loose
        for n in self.goal_nuts:
             if n not in nut_states:
                 # Assume goal nut not mentioned is already achieved
                 nut_states[n] = {'loc': None, 'loose': False}
                 tightened_nuts.add(n)
             elif n in tightened_nuts:
                 nut_states[n]['loose'] = False # Ensure consistency

        # Check for missing man location which prevents travel calculation
        if man_loc is None and carried_spanner is not None:
             # This state might be technically valid but problematic for the heuristic
             pass # Will be handled in __call__

        return man_loc, carried_spanner, spanner_states, nut_states, tightened_nuts

    def __call__(self, node):
        state = node.state
        man_loc, carried_spanner, spanner_states, nut_states, tightened_nuts = self._get_current_state_info(state)

        # Identify nuts that are part of the goal and are currently loose
        remaining_loose_goal_nuts = {
            n for n in self.goal_nuts if n in nut_states and nut_states[n]['loose']
        }

        # If no goal nuts remain loose, the heuristic value is 0
        if not remaining_loose_goal_nuts:
            return 0

        # If the man's location is unknown, we cannot calculate travel costs.
        # Return INFINITY as the state is likely invalid or unreachable in a way the heuristic can model.
        if man_loc is None:
             return self.INFINITY

        # --- Identify Available Resources for Simulation ---
        # Copy usable spanners on the ground {spanner: location}
        _sim_usable_spanners_on_ground = {
            s: data['loc'] for s, data in spanner_states.items()
            if data['usable'] and data['loc'] is not None and data['loc'] != self.the_man
        }
        # Check if the currently carried spanner is usable
        _sim_carried_spanner_is_usable = (
            carried_spanner is not None and spanner_states[carried_spanner]['usable']
        )

        # --- Simulate Tightening Sequence ---
        h = 0
        current_man_loc = man_loc
        sim_nuts_to_tighten = set(remaining_loose_goal_nuts) # Copy set for simulation
        final_penalty = 0 # Penalty accumulated due to unreachability/lack of resources

        while sim_nuts_to_tighten:
            # Check if any usable spanner is available (carried or on ground)
            if not _sim_carried_spanner_is_usable and not _sim_usable_spanners_on_ground:
                # Ran out of usable spanners, add penalty for remaining nuts
                final_penalty = len(sim_nuts_to_tighten) * 2
                break # Exit simulation loop

            # Get valid locations of nuts still needing tightening for distance checks
            nut_locations = {n: nut_states[n]['loc'] for n in sim_nuts_to_tighten if n in nut_states and nut_states[n]['loc'] is not None}
            if len(nut_locations) != len(sim_nuts_to_tighten):
                # Some nuts have unknown locations, cannot plan for them
                missing_nuts = sim_nuts_to_tighten - set(nut_locations.keys())
                final_penalty += len(missing_nuts) * self.LARGE_PENALTY_FACTOR
                sim_nuts_to_tighten -= missing_nuts # Remove them from consideration
                if not sim_nuts_to_tighten: break # Stop if no more known nuts left

            if _sim_carried_spanner_is_usable:
                # --- Man has a usable spanner ---
                best_nut, min_dist, nut_loc_to_go = self._find_closest(current_man_loc, nut_locations)

                if best_nut is None or min_dist == self.INFINITY:
                    # Cannot reach any remaining nut with the current spanner
                    final_penalty += len(sim_nuts_to_tighten) * self.LARGE_PENALTY_FACTOR
                    break # Stop simulation

                # Add cost: walk distance + 1 tighten action
                h += min_dist + 1
                current_man_loc = nut_loc_to_go # Update simulated man location
                _sim_carried_spanner_is_usable = False # Spanner is consumed
                sim_nuts_to_tighten.remove(best_nut) # Nut is tightened

            else:
                # --- Man needs to pick up a spanner ---
                # Find the closest usable spanner on the ground
                best_spanner, min_dist_to_spanner, spanner_loc_to_go = self._find_closest(current_man_loc, _sim_usable_spanners_on_ground)

                if best_spanner is None or min_dist_to_spanner == self.INFINITY:
                    # Cannot reach any usable spanner
                    final_penalty += len(sim_nuts_to_tighten) * self.LARGE_PENALTY_FACTOR
                    break # Stop simulation

                # Add cost: walk to spanner + 1 pickup action
                h += min_dist_to_spanner + 1
                current_man_loc = spanner_loc_to_go # Update simulated man location
                del _sim_usable_spanners_on_ground[best_spanner] # Spanner is picked up

                # Now, find the closest nut from the spanner's location
                # Re-evaluate nut locations as some might have been tightened in previous iterations if logic changed
                nut_locations = {n: nut_states[n]['loc'] for n in sim_nuts_to_tighten if n in nut_states and nut_states[n]['loc'] is not None}
                if not nut_locations: # Check if nuts became unreachable or were all tightened
                     if sim_nuts_to_tighten: # If nuts remain but have no location
                         final_penalty += len(sim_nuts_to_tighten) * self.LARGE_PENALTY_FACTOR
                     break # No more reachable nuts

                best_nut, min_dist_to_nut, nut_loc_to_go = self._find_closest(current_man_loc, nut_locations)

                if best_nut is None or min_dist_to_nut == self.INFINITY:
                    # Picked up spanner, but cannot reach any remaining nut
                    final_penalty += len(sim_nuts_to_tighten) * self.LARGE_PENALTY_FACTOR
                    break # Stop simulation

                # Add cost: walk to nut + 1 tighten action
                h += min_dist_to_nut + 1
                current_man_loc = nut_loc_to_go # Update simulated man location
                sim_nuts_to_tighten.remove(best_nut) # Nut is tightened
                # Man is no longer carrying a usable spanner (it was just used)
                _sim_carried_spanner_is_usable = False

        # Add any accumulated penalty from resource exhaustion or unreachability
        h += final_penalty

        # Ensure heuristic returns a finite value. If INFINITY was involved, return a large number.
        if h == self.INFINITY:
             return 1000000

        return int(round(h)) # Return heuristic value as integer

    def _find_closest(self, start_loc, target_dict):
        """ Finds the closest target object from start_loc based on precomputed distances.
            target_dict format: {object_name: location}
            Returns: (best_object, min_distance, target_location) or (None, inf, None) if no reachable target.
        """
        best_target = None
        min_dist = self.INFINITY
        target_loc = None

        # Ensure start_loc is valid before querying distances
        if start_loc not in self.distances:
            # print(f"Warning: Start location '{start_loc}' not found in distance matrix.")
            return None, self.INFINITY, None

        for obj, loc in target_dict.items():
            # Ensure target location is valid
            if loc is None or loc not in self.distances[start_loc]:
                # print(f"Warning: Target location '{loc}' for object '{obj}' is invalid or unreachable from '{start_loc}'.")
                continue # Skip this target

            dist = self.distances[start_loc][loc]
            if dist < min_dist:
                min_dist = dist
                best_target = obj
                target_loc = loc

        # If min_dist is still INFINITY, no reachable target was found
        if min_dist == self.INFINITY:
            return None, self.INFINITY, None

        return best_target, min_dist, target_loc
