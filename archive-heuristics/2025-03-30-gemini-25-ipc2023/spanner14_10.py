import math
from collections import deque
from fnmatch import fnmatch
# Import the base class definition for Heuristic
# Assume it's available in the environment where this code will be used.
# If not, you might need to define a placeholder or import it differently.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a placeholder if the base class is not available
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

# --- Helper Functions ---

def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string.
    Example: "(at bob shed)" -> ["at", "bob", "shed"]
    Returns an empty list if the fact is malformed (e.g., not starting/ending with parentheses).
    """
    if not fact or not fact.startswith("(") or not fact.endswith(")"):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a PDDL fact matches a pattern using fnmatch for wildcards.
    Example: match("(at bob shed)", "at", "*", "shed") -> True
    """
    parts = get_parts(fact)
    if not parts or len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def _compute_distances(locations, link_facts):
    """Computes all-pairs shortest paths using BFS on the location graph.

    Args:
        locations (set): A set of all location names.
        link_facts (set): A set of static link facts, e.g., "(link loc1 loc2)".

    Returns:
        dict: A dictionary mapping (loc1, loc2) tuples to shortest path distance (int),
              or math.inf if unreachable.
    """
    distances = {}
    # Initialize adjacency list only for known locations
    adj = {loc: [] for loc in locations}
    processed_links = set()

    # Build adjacency list from link facts
    for fact in link_facts:
        if match(fact, "link", "*", "*"):
             parts = get_parts(fact)
             # Handle potential malformed facts gracefully
             if len(parts) == 3:
                 l1, l2 = parts[1:]
                 # Ensure locations are in the known set before adding links
                 if l1 in locations and l2 in locations:
                     # Ensure links are unique and add both directions
                     link_tuple = tuple(sorted((l1, l2)))
                     if link_tuple not in processed_links:
                         adj[l1].append(l2)
                         adj[l2].append(l1)
                         processed_links.add(link_tuple)

    # Run BFS from each location
    for start_node in locations:
        # Initialize distances from start_node to infinity
        for loc in locations:
            distances[(start_node, loc)] = math.inf
        # Distance to self is 0
        distances[(start_node, start_node)] = 0

        queue = deque([(start_node, 0)])
        # visited set is crucial for BFS correctness and performance
        visited = {start_node}

        while queue:
            current_node, dist = queue.popleft()

            # Explore neighbors using the adjacency list
            for neighbor in adj.get(current_node, []):
                # Process neighbor only if it's a known location and not visited yet in this BFS run
                if neighbor in locations and neighbor not in visited:
                     visited.add(neighbor)
                     new_dist = dist + 1
                     distances[(start_node, neighbor)] = new_dist
                     queue.append((neighbor, new_dist))

    return distances

# --- Heuristic Class ---

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts
    that are currently loose. It simulates a greedy strategy where the single man
    iteratively fetches usable spanners and tightens the nearest nuts. Each tighten
    action consumes one spanner's usability. The heuristic is designed for Greedy
    Best-First Search and is not necessarily admissible.

    # Assumptions
    - There is exactly one object identifiable as the 'man'.
    - Nuts do not change location. Their initial 'at' location is static.
    - Each 'tighten_nut' action requires one usable spanner and makes that spanner unusable.
    - The number of initially usable spanners is sufficient for the number of goal nuts.
      If not, the heuristic might return infinity, indicating potential unreachability.
    - The location graph defined by 'link' predicates might be disconnected.
    - Object types (man, spanner, nut, location) can be inferred reasonably well
      from predicate usage or object names (e.g., 'nut1', 'spanner1'). This is less
      robust than using explicit type information if available from the task parser.

    # Heuristic Initialization (`__init__`)
    - Extracts all locations, the man object, nut objects and their static locations,
      spanner objects, and the set of goal nuts from the task definition (initial state,
      static facts, goals).
    - Parses static 'link' facts to build an adjacency list representation of
      the location graph.
    - Pre-computes all-pairs shortest path distances between all known locations using BFS.
      Unreachable location pairs have infinite distance. Stores distances in `self.distances`.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Parse Current State:** Identify the man's current location (`man_loc`),
        any spanner the man is currently carrying (`carried_spanner`), the set of
        spanners that are currently `usable` (`usable_spanners_state`), the locations
        of all spanners currently on the ground (`spanner_locs`), and the set of
        nuts that are currently `tightened` or `loose`.
    2.  **Identify Remaining Work:** Determine the set of goal nuts that are not yet
        tightened (`remaining_goal_nuts`). Check if these remaining nuts are currently
        `loose`. If a goal nut is neither tightened nor loose, the state is considered
        inconsistent or the goal potentially unreachable (return infinity). Create a map
        `nuts_to_tighten_dict` of {loose goal nut -> static location}.
    3.  **Check Goal Completion:** If `nuts_to_tighten_dict` is empty (all goal nuts
        are tightened), the heuristic value is 0.
    4.  **Initialize Simulation:** Set heuristic cost `h = 0`. Use `current_man_loc`
        to track the simulated man's position. Determine if the man starts carrying
        a usable spanner (`current_usable_spanner_carried`). Create a dictionary of
        `available_ground_spanners` {spanner -> location} for usable spanners currently
        on the ground (make a copy for simulation).
    5.  **Iterative Tightening Loop:** While `nuts_to_tighten_dict` is not empty:
        a.  **Case 1: Man carries a usable spanner.**
            i.  Find the nut (`best_nut`) in `nuts_to_tighten_dict` that has the
                minimum travel distance from `current_man_loc`.
            ii. If no nut is reachable (distance is infinity), return infinity.
            iii.Add `distance(current_man_loc, best_nut_loc) + 1` (cost to walk and tighten)
                to `h`.
            iv. Update `current_man_loc` to the nut's location (`best_nut_loc`).
            v.  Mark the carried spanner as used by setting `current_usable_spanner_carried = None`.
            vi. Remove `best_nut` from `nuts_to_tighten_dict`.
        b.  **Case 2: Man does not carry a usable spanner.**
            i.  Check if `available_ground_spanners` is empty. If yes, return infinity
                (cannot fetch a spanner for remaining nuts).
            ii. Find the available ground spanner (`best_spanner`) closest to `current_man_loc`.
            iii.If no spanner is reachable, return infinity.
            iv. Find the nut (`best_nut`) in `nuts_to_tighten_dict` closest to the
                `best_spanner`'s location.
            v.  If no nut is reachable from the spanner location, return infinity.
            vi. Add `distance(current_man_loc, best_spanner_loc) + 1 +
                distance(best_spanner_loc, best_nut_loc) + 1` (cost to walk to spanner,
                pickup, walk to nut, tighten) to `h`.
            vii.Update `current_man_loc` to the nut's location (`best_nut_loc`).
            viii.Remove `best_spanner` from `available_ground_spanners`.
            ix. Remove `best_nut` from `nuts_to_tighten_dict`.
    6.  **Return Total Cost:** Once the loop finishes (all loose goal nuts accounted for),
        return the accumulated heuristic cost `h`.
    """
    def __init__(self, task):
        super().__init__(task) # Initialize base class if necessary
        self.goals = task.goals
        self.static = task.static # Set of static facts, like '(link loc1 loc2)'

        # --- Precomputation ---
        self.man = None
        self.nuts = {} # nut -> location (static)
        self.spanners = set()
        self.locations = set()
        link_facts = set() # Store only link facts for distance calculation

        # Combine facts for parsing objects and locations
        facts_to_parse = task.initial_state | self.static | self.goals

        # First pass: Identify locations and gather link facts
        for fact in facts_to_parse:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             args = parts[1:]

             if pred == 'at' and len(args) == 2:
                 obj, loc = args
                 self.locations.add(loc)
             elif pred == 'link' and len(args) == 2:
                 l1, l2 = args
                 self.locations.add(l1)
                 self.locations.add(l2)
                 link_facts.add(fact)
             # Add other predicates if they introduce locations
             elif pred == 'tighten_nut' and len(args) == 4: # From action def example
                 loc, s, m, n = args
                 self.locations.add(loc)

        # Second pass: Identify man, nuts, spanners from initial state
        potential_man_objects = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             args = parts[1:]

             if pred == 'at' and len(args) == 2:
                 obj, loc = args
                 # Use simple string checks as a fallback if types aren't available
                 if 'nut' in obj and obj not in self.locations:
                     self.nuts[obj] = loc
                 elif 'spanner' in obj and obj not in self.locations:
                     self.spanners.add(obj)
                 elif obj not in self.locations: # Could be the man
                     potential_man_objects.add(obj)

             elif pred == 'carrying' and len(args) == 2:
                 man_obj, spanner_obj = args
                 self.man = man_obj # Found the man via carrying
                 self.spanners.add(spanner_obj)
                 potential_man_objects.discard(man_obj) # Remove from potentials if found here

             elif pred == 'usable' and len(args) == 1:
                 spanner_obj = args[0]
                 self.spanners.add(spanner_obj)

        # Refine man identification
        if self.man is None:
            # Remove known nuts and spanners from potential men
            potential_man_objects -= self.nuts.keys()
            potential_man_objects -= self.spanners
            if len(potential_man_objects) == 1:
                self.man = potential_man_objects.pop()

        if self.man is None:
             # If still not found, raise error. A valid problem should define the man.
             raise ValueError("Failed to identify the unique man object. Check initial state for 'at' or 'carrying' predicates involving the man.")

        # Store goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                parts = get_parts(goal)
                if len(parts) == 2:
                    self.goal_nuts.add(parts[1])

        # Compute distances using the identified locations and link facts
        # Pass a copy of locations set as it might be modified by _compute_distances
        self.distances = _compute_distances(self.locations.copy(), link_facts)

    def _get_distance(self, loc1, loc2):
        """Safely retrieves distance, returning infinity for unreachable/unknown locations."""
        if loc1 not in self.locations or loc2 not in self.locations:
            # print(f"Warning: Querying distance for unknown location: {loc1} or {loc2}")
            return math.inf
        # Use .get for safety, defaulting to infinity
        return self.distances.get((loc1, loc2), math.inf)

    def __call__(self, node):
        """Calculates the heuristic value for a given state node."""
        state = node.state

        # --- State Parsing ---
        man_loc = None
        carried_spanner = None
        usable_spanners_state = set() # Usable spanners in this specific state
        spanner_locs = {} # Location of spanners on the ground
        tightened_nuts = set()
        loose_nuts = set() # All loose nuts in the state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            if pred == 'at' and len(args) == 2:
                obj, loc = args
                if obj == self.man:
                    man_loc = loc
                elif obj in self.spanners:
                    spanner_locs[obj] = loc
                # Nut locations are static, no need to track 'at nut loc' here
            elif pred == 'carrying' and len(args) == 2:
                man, spanner = args
                if man == self.man:
                    carried_spanner = spanner
            elif pred == 'usable' and len(args) == 1:
                spanner = args[0]
                usable_spanners_state.add(spanner)
            elif pred == 'tightened' and len(args) == 1:
                nut = args[0]
                tightened_nuts.add(nut)
            elif pred == 'loose' and len(args) == 1:
                 nut = args[0]
                 loose_nuts.add(nut)

        if man_loc is None:
            # If the man exists, he should always be 'at' some location in a valid state.
            # print(f"Error: Could not find man location for {self.man} in state.")
            return math.inf # Cannot compute heuristic without man location

        # --- Heuristic Calculation ---
        remaining_goal_nuts = self.goal_nuts - tightened_nuts
        if not remaining_goal_nuts:
            return 0 # Goal reached

        # Identify which of the remaining goal nuts are actually loose
        # Use a dictionary {nut: location} for nuts to be tightened
        nuts_to_tighten_dict = {}
        for nut in remaining_goal_nuts:
            if nut in loose_nuts:
                 if nut in self.nuts:
                     nuts_to_tighten_dict[nut] = self.nuts[nut] # Map nut -> static location
                 else:
                     # Initialization failed to find this nut's location
                     # print(f"Error: Location for goal nut {nut} unknown during heuristic calculation.")
                     return math.inf # Cannot proceed without nut location
            else:
                 # A goal nut is not tightened but also not loose. Inconsistent state?
                 # print(f"Warning: Goal nut {nut} is not tightened but not found in loose set: {loose_nuts}")
                 return math.inf # Treat as unreachable/inconsistent

        if not nuts_to_tighten_dict:
             # This case means remaining_goal_nuts was non-empty, but none were loose.
             # This implies an inconsistency, already handled above by returning inf.
             # If somehow reached here, return inf for safety.
             return math.inf


        h = 0
        current_man_loc = man_loc

        # Check if currently carried spanner is usable
        current_usable_spanner_carried = None
        if carried_spanner is not None and carried_spanner in usable_spanners_state:
            current_usable_spanner_carried = carried_spanner

        # Available spanners on the ground that are usable in this state
        # Make a copy to modify during simulation
        available_ground_spanners = {
            s: loc for s, loc in spanner_locs.items() if s in usable_spanners_state
        }

        # --- Simulation Loop ---
        # Continue while there are still nuts that need tightening
        while nuts_to_tighten_dict:
            if current_usable_spanner_carried:
                # Man has a usable spanner. Find the closest nut to tighten.
                best_nut = None
                min_dist = math.inf
                best_nut_loc = None

                # Iterate through the dictionary of nuts needing tightening
                for nut, nut_loc in nuts_to_tighten_dict.items():
                    dist = self._get_distance(current_man_loc, nut_loc)
                    # Update if this nut is closer
                    if dist < min_dist:
                        min_dist = dist
                        best_nut = nut
                        best_nut_loc = nut_loc

                # If no nut is reachable, the goal is unreachable from this state
                if best_nut is None or min_dist == math.inf:
                    # print(f"Heuristic Warning: Cannot reach any remaining nuts from {current_man_loc}")
                    return math.inf

                # Add cost: walk distance + 1 for tighten action
                h += min_dist + 1
                # Update simulated man location
                current_man_loc = best_nut_loc
                # Mark the carried spanner as used for the simulation
                current_usable_spanner_carried = None

                # Remove the 'tightened' nut from the dictionary
                del nuts_to_tighten_dict[best_nut]

            else:
                # Man needs to pick up a spanner.
                if not available_ground_spanners:
                    # No usable spanners left on the ground, but nuts remain. Goal unreachable.
                    # print(f"Heuristic Warning: No usable ground spanners left, but nuts {list(nuts_to_tighten_dict.keys())} remain.")
                    return math.inf

                # Find the available ground spanner closest to the man.
                best_spanner = None
                best_spanner_loc = None
                min_dist_to_spanner = math.inf

                for spanner, spanner_loc in available_ground_spanners.items():
                    dist = self._get_distance(current_man_loc, spanner_loc)
                    if dist < min_dist_to_spanner:
                        min_dist_to_spanner = dist
                        best_spanner = spanner
                        best_spanner_loc = spanner_loc

                # If no spanner is reachable, goal is unreachable.
                if best_spanner is None or min_dist_to_spanner == math.inf:
                    # print(f"Heuristic Warning: Cannot reach any available spanners from {current_man_loc}")
                    return math.inf

                # Now find the nut (among remaining) closest to this chosen spanner's location.
                best_nut = None
                best_nut_loc = None
                min_dist_spanner_to_nut = math.inf

                for nut, nut_loc in nuts_to_tighten_dict.items():
                    dist = self._get_distance(best_spanner_loc, nut_loc)
                    if dist < min_dist_spanner_to_nut:
                        min_dist_spanner_to_nut = dist
                        best_nut = nut
                        best_nut_loc = nut_loc

                # If no remaining nut is reachable from the spanner location, goal is unreachable.
                if best_nut is None or min_dist_spanner_to_nut == math.inf:
                    # print(f"Heuristic Warning: Cannot reach any remaining nuts from spanner location {best_spanner_loc}")
                    return math.inf

                # Add costs: walk to spanner, pickup(1), walk to nut, tighten(1)
                h += min_dist_to_spanner + 1 + min_dist_spanner_to_nut + 1
                # Update simulated man location to the nut's location
                current_man_loc = best_nut_loc

                # Remove the used spanner and tightened nut from available sets for the simulation
                del available_ground_spanners[best_spanner]
                del nuts_to_tighten_dict[best_nut]

        # If the loop completes, all goal nuts have been accounted for.
        return h
