import math
from collections import deque
from fnmatch import fnmatch
# Assuming heuristic_base is accessible, e.g., part of the same package or in PYTHONPATH
# If the heuristic base class is not provided, this heuristic can stand alone,
# but the `Heuristic` inheritance might need adjustment depending on the planner framework.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if Heuristic is not available
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact string by removing parentheses and splitting."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern. Wildcards (*) are allowed in args.
    Returns True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check each part against the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all goal nuts in the Spanner domain.
    It simulates a greedy strategy where the agent repeatedly performs two phases:
    1. If not holding a usable spanner, travel to the nearest available usable spanner on the ground and pick it up.
    2. Travel from the current location (with a usable spanner) to the nearest remaining loose goal nut and tighten it.
    The heuristic counts the number of walk, pickup, and tighten actions estimated by this greedy simulation. It aims for informativeness over admissibility, suitable for Greedy Best-First Search.

    # Assumptions
    - There is only one agent (man) in the problem.
    - Nuts do not move; their locations are static and derived from the initial state.
    - Links between locations are static and assumed to be bidirectional for distance calculation.
    - Each 'tighten_nut' action makes the used spanner unusable.
    - The heuristic needs to identify the man, nuts, spanners, and locations. It attempts to infer this from task object types if available, otherwise falling back to predicate analysis.

    # Heuristic Initialization (`__init__`)
    - Identifies the man, all nuts, spanners, and locations from the task definition (using object types if available, otherwise inferring from predicates). Robust identification is crucial.
    - Stores the locations of all nuts (assumed static from the initial state).
    - Identifies the set of nuts that need to be tightened based on the goal description `(tightened ?n)`.
    - Parses the static 'link' predicates to build a graph representation of the locations.
    - Computes all-pairs shortest path distances between locations using Breadth-First Search (BFS) and stores them in `self.distances`. Handles unreachable locations with infinite distance.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Parse the current state `node.state` (a frozenset of facts) to find:
       - The man's current location (`current_man_loc`).
       - The spanner the man is currently carrying (`carried_spanner`), if any.
       - The set of spanners that are currently `usable` (`usable_spanners_state`).
       - The locations of spanners currently on the ground (`spanner_locations_state`).
       - The set of goal nuts that are currently `loose` (`loose_goal_nuts`).
    2. If `loose_goal_nuts` is empty, the goal state is reached, return 0.
    3. Initialize heuristic value `h = 0`.
    4. Determine if the man starts the simulation step holding a usable spanner (`current_carried_spanner`).
    5. Create a dictionary `active_usable_spanners` mapping usable spanners currently on the ground to their locations.
    6. Start a loop that continues as long as there are `nuts_to_tighten` (initially the set of `loose_goal_nuts`):
       a. **Phase 1: Get Spanner (if needed)**
          - If `current_carried_spanner` is None (man doesn't have a usable spanner):
            i. Find the usable spanner (`chosen_spanner`) in `active_usable_spanners` (spanners on the ground) that is nearest to `current_man_loc` based on precomputed distances. The distance is `min_dist_to_spanner`.
            ii. If no usable spanners are left on the ground or none are reachable (`min_dist_to_spanner` is infinity), return `math.inf` (indicates a dead end or unsolvable state from the heuristic's perspective).
            iii. Add `min_dist_to_spanner` (walk actions) + 1 (pickup action) to `h`. Check for overflow to infinity.
            iv. Update `current_man_loc` to the chosen spanner's location.
            v. Set `current_carried_spanner` to `chosen_spanner`.
            vi. Remove `chosen_spanner` from `active_usable_spanners` (it's no longer on the ground).
       b. **Phase 2: Tighten Nut**
          - Now, the man is at `current_man_loc` holding a usable spanner (`current_carried_spanner`).
          i. Find the nut (`nut_to_tighten`) in `nuts_to_tighten` whose static location (`nut_loc`, retrieved from `self.nut_locations`) is nearest to `current_man_loc`. The distance is `min_dist_to_nut`.
          ii. If no remaining nuts are reachable (`min_dist_to_nut` is infinity), return `math.inf`.
          iii. Add `min_dist_to_nut` (walk actions) + 1 (tighten action) to `h`. Check for overflow to infinity.
          iv. Update `current_man_loc` to `nut_loc`.
          v. Remove `nut_to_tighten` from the set `nuts_to_tighten`.
          vi. Set `current_carried_spanner` to None (the spanner used is now conceptually unusable for the rest of the simulation).
    7. After the loop finishes (all goal nuts accounted for), return the total estimated cost `h`. If any step resulted in `math.inf`, the function returns `math.inf`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing the task definition.
        Extracts static information like object types, locations, links,
        goal nuts, and precomputes distances between locations.
        """
        super().__init__(task) # Initialize base class if necessary
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # --- Object Identification ---
        self.man = None
        self.all_nuts = set()
        self.all_spanners = set()
        self.all_locations = set()

        # Try to use task object type information if available (preferred)
        # This depends on the planner's Task object structure.
        # Example: Check for a 'types' attribute mapping names to types
        # Adjust based on the actual structure provided by the planner framework.
        if hasattr(task, 'types') and isinstance(task.types, dict):
            for name, type_ in task.types.items():
                # PDDL types can be hierarchical, check base type if needed
                # Assuming simple types 'man', 'nut', 'spanner', 'location'
                if type_ == 'man':
                    self.man = name # Assume only one man
                elif type_ == 'nut':
                    self.all_nuts.add(name)
                elif type_ == 'spanner':
                    self.all_spanners.add(name)
                elif type_ == 'location':
                    self.all_locations.add(name)
            # If locations weren't found via types, try inferring them
            if not self.all_locations:
                 print("Warning: No locations identified via types. Checking predicates.")
                 self._infer_locations_from_predicates(task)
        else:
             # Fallback if task.types info is missing or not structured as expected
             print("Warning: Task type information not available or not in expected format. Inferring from predicates.")
             self._infer_objects_from_predicates(task) # This includes location inference

        # Ensure man was identified
        if self.man is None:
            self._find_man_if_missing(task) # Try one more time
            if self.man is None:
                 raise ValueError("Could not identify the man object in the task. Heuristic cannot operate.")

        # Ensure locations were identified
        if not self.all_locations:
             print("Warning: No locations identified after all attempts. Distance calculations may fail.")


        # --- Static Information ---
        # Nut locations (assumed static, derived from initial state)
        self.nut_locations = {}
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Check for '(at nutX locY)'
             if len(parts) == 3 and parts[0] == 'at' and parts[1] in self.all_nuts:
                 self.nut_locations[parts[1]] = parts[2]
                 # Also ensure the location is known
                 if parts[2] not in self.all_locations:
                     print(f"Warning: Nut location '{parts[2]}' not in identified locations. Adding it.")
                     self.all_locations.add(parts[2])


        # Target nuts from goals
        self.target_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Check for '(tightened nutX)'
            if len(parts) == 2 and parts[0] == 'tightened':
                # Ensure the object is a known nut
                if parts[1] in self.all_nuts:
                    self.target_nuts.add(parts[1])
                else:
                     # This might indicate an issue with nut identification or the PDDL goal
                     print(f"Warning: Goal refers to tightening '{parts[1]}', which is not identified as a nut.")


        # Compute distances between all identified locations
        self.distances = self._compute_distances(self.all_locations, static_facts)


    def _infer_locations_from_predicates(self, task):
        """Helper to specifically infer locations if type info was incomplete."""
        locations = set()
        # Infer locations from 'link' predicates in static facts
        for fact in task.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'link':
                locations.add(parts[1])
                locations.add(parts[2])
        # Infer locations from 'at' predicates in initial state
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == 'at':
                 locations.add(parts[2]) # Add the location part
        # Update the main set of locations
        self.all_locations.update(locations)


    def _infer_objects_from_predicates(self, task):
        """Fallback method to infer objects and types from predicates if structured info is unavailable."""
        print("Warning: Inferring object types from predicates. This might be inaccurate.")
        locations = set()
        potential_spanners = set()
        potential_nuts = set()
        potential_men = set() # Use this to find the man later

        # Infer locations first
        self._infer_locations_from_predicates(task)
        locations = self.all_locations # Use the updated set

        # Gather all objects mentioned in 'at' predicates initially
        initial_objects_at = {}
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == 'at':
                 obj_name = parts[1]
                 initial_objects_at[obj_name] = parts[2] # Store object and its location

        # Infer types based on predicate usage across all relevant facts
        all_facts = task.initial_state.union(task.static).union(task.goals)
        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             args = parts[1:]

             if pred == 'usable' and len(args) == 1: potential_spanners.add(args[0])
             if pred == 'carrying' and len(args) == 2:
                 potential_men.add(args[0])
                 potential_spanners.add(args[1])
             if pred == 'loose' and len(args) == 1: potential_nuts.add(args[0])
             if pred == 'tightened' and len(args) == 1: potential_nuts.add(args[0])

        # Assign inferred types
        # Locations are already updated in self.all_locations
        self.all_nuts = potential_nuts
        self.all_spanners = potential_spanners

        # Try to identify the single man from 'carrying' predicates
        if len(potential_men) == 1:
            self.man = list(potential_men)[0]
        elif len(potential_men) > 1:
             # If carrying identified multiple, pick the first? Or error?
             self.man = list(potential_men)[0] # Take the first one found
             print(f"Warning: Multiple potential 'man' objects inferred from 'carrying' ({potential_men}). Using '{self.man}'.")
        # If man still not found, _find_man_if_missing will try other methods


    def _find_man_if_missing(self, task):
        """Attempt to find the man if initial methods failed, e.g., using 'at' facts."""
        if self.man: return # Already found

        potential_men_carrying = set()
        initial_objects_at = set()

        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == 'at': initial_objects_at.add(parts[1])
             if len(parts) == 3 and parts[0] == 'carrying': potential_men_carrying.add(parts[1])

        # Priority 1: Unique object involved in 'carrying'
        if len(potential_men_carrying) == 1:
            self.man = list(potential_men_carrying)[0]
            print(f"Info: Identified man '{self.man}' from 'carrying' predicate.")
            return

        # Priority 2: Unique object 'at' a location that isn't a known nut or spanner
        possible_men_at = initial_objects_at - self.all_nuts - self.all_spanners
        if len(possible_men_at) == 1:
            self.man = list(possible_men_at)[0]
            print(f"Info: Identified man '{self.man}' as unique object at a location (not nut/spanner).")
            return

        # Fallback: If multiple carrying, pick first (already handled in _infer_objects)
        # Fallback: If multiple possible from 'at', cannot uniquely identify.
        if potential_men_carrying:
             self.man = list(potential_men_carrying)[0] # Fallback: pick first from carrying
             print(f"Warning: Ambiguous man identification. Using '{self.man}' based on 'carrying'.")
             return

        print("Error: Could not uniquely identify the man object.")
        # self.man remains None, which will raise error later or cause issues.


    def _compute_distances(self, locations, links):
        """Computes all-pairs shortest paths using BFS. Returns dict mapping (loc1, loc2) to distance."""
        if not locations:
            print("Warning: No locations provided for distance calculation.")
            return {}

        adj = {loc: [] for loc in locations}
        link_facts = {link for link in links if match(link, "link", "*", "*")}

        # Build adjacency list
        for fact in link_facts:
            parts = get_parts(fact)
            if len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                # Add edges only if both locations are in our known set
                if l1 in locations and l2 in locations:
                    adj[l1].append(l2)
                    adj[l2].append(l1) # Assume bidirectional links
                else:
                    # This might happen if link refers to unknown locations
                    if l1 not in locations: print(f"Warning: Location '{l1}' in link not in known locations.")
                    if l2 not in locations: print(f"Warning: Location '{l2}' in link not in known locations.")


        distances = {}
        # Run BFS from each location to find shortest paths
        for start_node in locations:
            # Initialize distance to self as 0
            distances[(start_node, start_node)] = 0
            queue = deque([(start_node, 0)]) # Queue stores (node, distance)
            visited = {start_node: 0} # Keep track of visited nodes and their distances

            while queue:
                current_node, dist = queue.popleft()

                # Explore neighbors
                for neighbor in adj.get(current_node, []):
                    # Ensure neighbor is a known location and not visited yet
                    if neighbor in locations and neighbor not in visited:
                        visited[neighbor] = dist + 1
                        distances[(start_node, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

        # Set infinite distance for unreachable pairs
        all_locs_list = list(locations)
        for i in range(len(all_locs_list)):
            for j in range(len(all_locs_list)):
                l1 = all_locs_list[i]
                l2 = all_locs_list[j]
                # If a pair wasn't reached by BFS, set distance to infinity
                if (l1, l2) not in distances:
                    distances[(l1, l2)] = float('inf')

        return distances

    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        Estimates the cost to reach the goal by simulating a greedy plan.
        Returns the estimated cost (number of actions) or math.inf if the goal seems unreachable.
        """
        state = node.state

        # --- Parse Current State ---
        man_loc = None
        carried_spanner = None
        usable_spanners_state = set() # Names of all spanners currently usable
        spanner_locations_state = {} # name -> location (for spanners on the ground)
        loose_nuts_state = set() # Names of all nuts currently loose

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts
            pred = parts[0]
            args = parts[1:]

            # Extract relevant information based on predicates
            if pred == 'at' and len(args) == 2:
                obj, loc = args[0], args[1]
                if obj == self.man:
                    man_loc = loc
                # Record location only if it's a known spanner
                elif obj in self.all_spanners:
                    spanner_locations_state[obj] = loc
            elif pred == 'carrying' and len(args) == 2 and args[0] == self.man:
                # Ensure the object carried is a known spanner
                if args[1] in self.all_spanners:
                    carried_spanner = args[1]
            elif pred == 'usable' and len(args) == 1 and args[0] in self.all_spanners:
                usable_spanners_state.add(args[0])
            elif pred == 'loose' and len(args) == 1 and args[0] in self.all_nuts:
                loose_nuts_state.add(args[0])

        # Check if man's location was found (essential for calculation)
        if man_loc is None:
             # This indicates a potentially invalid state or issue with man identification
             print(f"Warning: Could not find location for man '{self.man}' in state. Returning inf.")
             return float('inf')

        # Identify which of the goal nuts are still loose
        loose_goal_nuts = self.target_nuts.intersection(loose_nuts_state)

        # --- Goal Check ---
        if not loose_goal_nuts:
            # All target nuts are tightened
            return 0

        # --- Prepare for Simulation ---
        # Identify usable spanners currently ON THE GROUND
        active_usable_spanners = {} # spanner_name -> location
        for spanner_name in usable_spanners_state:
            # Check if it's usable AND currently located somewhere (i.e., on the ground)
            if spanner_name in spanner_locations_state:
                 # Ensure it's not the one being carried (if any)
                 if spanner_name != carried_spanner:
                    active_usable_spanners[spanner_name] = spanner_locations_state[spanner_name]

        # Determine if the man starts this simulation step holding a usable spanner
        current_carried_spanner = carried_spanner if (carried_spanner is not None and carried_spanner in usable_spanners_state) else None

        # --- Start Heuristic Calculation (Greedy Simulation) ---
        h = 0 # Initialize heuristic cost (action count)
        current_man_loc = man_loc # Track simulated man location
        nuts_to_tighten = set(loose_goal_nuts) # Work on a copy of remaining goal nuts

        # Loop until all loose goal nuts are accounted for in the simulation
        while nuts_to_tighten:
            # --- Phase 1: Ensure man has a usable spanner ---
            if current_carried_spanner is None: # Man needs to acquire a usable spanner
                min_dist_to_spanner = float('inf')
                chosen_spanner = None
                spanner_loc = None # Location of the chosen spanner

                # Check if there are any usable spanners left on the ground
                if not active_usable_spanners:
                    # No usable spanners available to pick up
                    return float('inf') # Goal seems unreachable

                # Find the nearest usable spanner on the ground
                for s, loc in active_usable_spanners.items():
                    # Get distance from current simulated man location to spanner location
                    dist = self.distances.get((current_man_loc, loc), float('inf'))
                    if dist < min_dist_to_spanner:
                        min_dist_to_spanner = dist
                        chosen_spanner = s
                        spanner_loc = loc

                # Check if a reachable spanner was found
                if chosen_spanner is None or min_dist_to_spanner == float('inf'):
                     # No usable spanner is reachable from the current location
                     return float('inf')

                # Add cost: walk to spanner + pickup spanner
                h += min_dist_to_spanner
                # Check for infinity distance before adding action cost
                if h == float('inf'): return float('inf')
                h += 1 # Cost of pickup action

                # Update simulation state
                current_man_loc = spanner_loc # Man moves to the spanner's location
                current_carried_spanner = chosen_spanner # Man is now carrying this spanner
                # Remove the picked-up spanner from the available pool on the ground
                del active_usable_spanners[chosen_spanner]

            # --- Phase 2: Go tighten the nearest nut ---
            # At this point, man is at `current_man_loc` and carrying a usable `current_carried_spanner`
            min_dist_to_nut = float('inf')
            nut_to_tighten = None
            nut_loc = None # Location of the chosen nut

            # Should not happen if while loop condition is correct, but check for safety
            if not nuts_to_tighten: break

            # Find the nearest remaining loose goal nut
            for nut in nuts_to_tighten:
                # Get the static location of the nut
                n_loc = self.nut_locations.get(nut)
                if n_loc is None:
                    # This indicates an issue with initialization (nut location unknown)
                    print(f"Warning: Location for nut '{nut}' not found in precomputed map. Skipping.")
                    continue # Skip this nut if location is unknown

                # Get distance from current simulated man location to nut location
                dist = self.distances.get((current_man_loc, n_loc), float('inf'))
                if dist < min_dist_to_nut:
                    min_dist_to_nut = dist
                    nut_to_tighten = nut
                    nut_loc = n_loc

            # Check if a reachable nut was found
            if nut_to_tighten is None or min_dist_to_nut == float('inf'):
                # Cannot reach any of the remaining loose goal nuts
                return float('inf')

            # Add cost: walk to nut + tighten nut
            h += min_dist_to_nut
            # Check for infinity distance before adding action cost
            if h == float('inf'): return float('inf')
            h += 1 # Cost of tighten action

            # Update simulation state
            current_man_loc = nut_loc # Man moves to the nut's location
            nuts_to_tighten.remove(nut_to_tighten) # This nut is now accounted for
            current_carried_spanner = None # The spanner used becomes unusable for the next iteration

        # If the loop completes, return the calculated heuristic value
        # Ensure the returned value is finite if the calculation succeeded
        return h if h != float('inf') else float('inf') # Return infinity if any step failed

