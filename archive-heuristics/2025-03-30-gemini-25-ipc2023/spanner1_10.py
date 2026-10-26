import collections
import itertools
import heapq
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For infinity

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Check if fact is a string and has parentheses before slicing
    if isinstance(fact, str) and len(fact) > 1 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Return an empty list or raise error if format is unexpected
    # print(f"Warning: Unexpected fact format: {fact}")
    return []

# Helper function to match facts against patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if not parts or len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL 'spanner' domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It simulates a greedy strategy where the agent iteratively chooses the next action
    (walk, pickup, tighten) based on minimizing immediate costs (primarily distance).
    It considers the constraint that each nut requires a unique usable spanner. The
    heuristic aims for accuracy in estimation rather than admissibility.

    # Assumptions
    - There is only one 'man' agent in the problem.
    - Nuts do not move. Their locations are static throughout the plan execution.
    - Links between locations are bidirectional and static.
    - The goal is always a conjunction of '(tightened nut)' facts.
    - The heuristic assumes a greedy approach: always go to the nearest required item
      (spanner or nut) next. This is not necessarily optimal but provides an estimate.
    - Spanner and nut names can be identified via predicates like `(usable ?s)` or
      `(loose ?n)`, or potentially name patterns like 'spannerX' and 'nutX'.

    # Heuristic Initialization
    - Extracts the single man's name by checking 'at' or 'carrying' predicates in the initial state.
      Uses fallback mechanisms if direct identification fails.
    - Extracts the locations of all nuts from the initial state using the 'at' predicate,
      cross-referencing with 'loose'/'tightened' facts.
    - Extracts the set of goal nuts that need to be tightened from the task goals.
    - Builds a graph representation of the locations based on 'link' predicates in static facts.
      Also includes locations mentioned in the initial state 'at' predicates.
    - Pre-computes all-pairs shortest paths (APSP) using Breadth-First Search (BFS)
      for all reachable locations. Stores distances in `self.distances`. Unreachable pairs
      are assigned infinite distance.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Identify the man's current location (`current_man_loc`),
        whether the man is carrying a spanner (`carried_spanner`), which spanners are
        currently usable (`usable_spanner_names_in_state`), the locations of all spanners
        (`spanner_locations_in_state`), and the set of nuts that are currently loose (`loose_nuts`).
    2.  **Identify Remaining Goals:** Determine which goal nuts are still loose in the
        current state (`nuts_to_tighten = self.goal_nuts.intersection(loose_nuts)`).
        If this set is empty, check if the state satisfies all goal conditions (`self.goals <= state`).
        If yes, return 0. If no (e.g., other non-nut goals exist), return 1 as a minimum cost.
    3.  **Initialize Simulation:** Set heuristic cost `h = 0`. Keep track of the man's
        simulated location (`sim_man_loc`), whether the man is simulated to be
        carrying a usable spanner (`sim_carrying_usable`), the set of available usable
        spanners on the ground (`sim_available_spanners`), and the set of nuts still
        needing to be tightened (`sim_nuts_remaining`).
    4.  **Iterative Action Estimation:** Loop while `sim_nuts_remaining` is not empty:
        a.  **If Carrying Usable Spanner:**
            i.  Find the nut `n` in `sim_nuts_remaining` that is closest to `sim_man_loc`
                (minimum `self.get_dist(sim_man_loc, self.nut_locations[n])`).
            ii. If no reachable nut exists (all remaining nuts have infinite distance or location unknown), return infinity.
            iii.Add the distance (`min_dist_to_nut`) to `h` (cost of walking).
            iv. Add 1 to `h` (cost of `tighten_nut`).
            v.  Update `sim_man_loc` to the nut's location (`target_nut_loc`).
            vi. Remove `n` from `sim_nuts_remaining`.
            vii.Set `sim_carrying_usable` to `False` (spanner is now used).
        b.  **If Not Carrying Usable Spanner:**
            i.  Check if `sim_available_spanners` is empty. If yes, the goal is unreachable
                with this strategy (ran out of spanners); return infinity.
            ii. Find the usable spanner `s` in `sim_available_spanners` that is closest to
                `sim_man_loc` (minimum `self.get_dist(sim_man_loc, loc)`).
            iii.If no reachable spanner exists (all remaining spanners have infinite distance), return infinity.
            iv. Add the distance (`min_dist_to_spanner`) to `h` (cost of walking).
            v.  Add 1 to `h` (cost of `pickup_spanner`).
            vi. Update `sim_man_loc` to the spanner's location (`target_spanner_loc`).
            vii.Remove `s` from `sim_available_spanners` (it's now picked up).
            viii.Now the man is simulated to be carrying a usable spanner. Find the nut `n`
                in `sim_nuts_remaining` closest to the *new* `sim_man_loc` (the spanner's location).
            ix. If no reachable nut exists, return infinity.
            x.  Add the distance (`min_dist_to_nut`) to `h` (cost of walking).
            xi. Add 1 to `h` (cost of `tighten_nut`).
            xii.Update `sim_man_loc` to the nut's location (`target_nut_loc`).
            xiii.Remove `n` from `sim_nuts_remaining`.
            xiv.Set `sim_carrying_usable` to `False` (spanner is now used).
    5.  **Return Total Cost:** Return the accumulated heuristic value `h`. If at any point
        a required location is unreachable or resources (usable spanners) run out during
        the simulation, return `float('inf')` to indicate the state is considered unsolvable
        by this greedy strategy.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        # --- Heuristic Initialization ---

        # 1. Find the man (assume one)
        self.man_name = None
        # Try finding via 'carrying' predicate
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "carrying":
                 self.man_name = parts[1]
                 break
        # Fallback: Try finding via 'at' predicate for non-spanner/nut/location objects
        if self.man_name is None:
             # Gather all object names mentioned in 'at'
             objects_at_locs = set()
             for fact in self.initial_state:
                 parts = get_parts(fact)
                 if parts and parts[0] == "at":
                     objects_at_locs.add(parts[1])

             # Identify objects that are definitely spanners or nuts
             spanners_nuts = set()
             for fact in self.initial_state:
                 parts = get_parts(fact)
                 if not parts: continue
                 if parts[0] in ["usable", "loose", "tightened"]:
                     spanners_nuts.add(parts[1])
                 elif parts[0] == "at" and ("spanner" in parts[1] or "nut" in parts[1]):
                     spanners_nuts.add(parts[1])

             # Assume the man is an object at a location that isn't a known spanner/nut
             possible_men = objects_at_locs - spanners_nuts
             if len(possible_men) == 1:
                 self.man_name = possible_men.pop()
             # If ambiguity remains, we might need a better method or default

        if self.man_name is None:
            # Last resort based on example
            self.man_name = 'bob'
            # print(f"Warning: Could not reliably determine man's name. Assuming '{self.man_name}'.")


        # 2. Find nut locations (assume static)
        self.nut_locations = {}
        nut_names = set()
        # First pass: identify all nuts from relevant predicates
        for fact in self.initial_state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] in ["loose", "tightened"] or (parts[0] == "at" and "nut" in parts[1]):
                 nut_names.add(parts[1])
        # Second pass: find the location for each identified nut
        for nut in nut_names:
            for fact in self.initial_state:
                parts = get_parts(fact)
                if parts and parts[0] == "at" and parts[1] == nut:
                    self.nut_locations[nut] = parts[2]
                    break # Found location for this nut


        # 3. Find goal nuts
        self.goal_nuts = set()
        for goal_fact in self.goals:
            if match(goal_fact, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal_fact)[1])

        # 4. Build location graph and compute APSP
        self.locations = set()
        adj = collections.defaultdict(set)
        # Add locations from links
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                adj[loc1].add(loc2)
                adj[loc2].add(loc1)

        # Add locations only mentioned in 'at' predicates if not already included
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at":
                 self.locations.add(parts[2]) # Add the location part

        # Compute APSP using BFS from each location
        self.distances = {}
        for start_node in self.locations:
            # Initialize distances for this start node
            for target_node in self.locations:
                 self.distances[(start_node, target_node)] = float('inf')
            self.distances[(start_node, start_node)] = 0

            queue = collections.deque([(start_node, 0)])
            visited = {start_node} # Keep track of visited nodes for BFS

            while queue:
                curr_node, dist = queue.popleft()
                # Explore neighbors
                for neighbor in adj.get(curr_node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_dist = dist + 1
                        self.distances[(start_node, neighbor)] = new_dist
                        queue.append((neighbor, new_dist))


    def get_dist(self, loc1, loc2):
        """Returns the precomputed shortest distance between two locations."""
        # Handle cases where one location might be None (e.g., man location not found)
        if loc1 is None or loc2 is None:
            return float('inf')
        # Return infinity if the pair wasn't found (e.g., disconnected graph)
        return self.distances.get((loc1, loc2), float('inf'))


    def __call__(self, node):
        state = node.state

        # --- Parse Current State ---
        current_man_loc = None
        carried_spanner = None
        loose_nuts = set()
        tightened_nuts = set()
        usable_spanner_names_in_state = set()
        spanner_locations_in_state = {} # Tracks current location of all spanners

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at":
                obj, loc = args[0], args[1]
                if obj == self.man_name:
                    current_man_loc = loc
                # Use a simple check for spanner names - adjust if names vary
                elif "spanner" in obj:
                    spanner_locations_in_state[obj] = loc
            elif predicate == "carrying" and args[0] == self.man_name:
                carried_spanner = args[1]
            elif predicate == "usable":
                usable_spanner_names_in_state.add(args[0])
            elif predicate == "loose":
                loose_nuts.add(args[0])
            elif predicate == "tightened":
                 tightened_nuts.add(args[0])

        # Determine if carried spanner is usable
        sim_carrying_usable = False
        if carried_spanner and carried_spanner in usable_spanner_names_in_state:
            sim_carrying_usable = True

        # Find usable spanners on the ground
        sim_available_spanners = {} # {spanner_name: location}
        for spanner_name, loc in spanner_locations_in_state.items():
             if spanner_name in usable_spanner_names_in_state:
                  sim_available_spanners[spanner_name] = loc


        # --- Identify Remaining Goals ---
        nuts_to_tighten = self.goal_nuts.intersection(loose_nuts)

        if not nuts_to_tighten:
             # Check if all goal predicates are true in the state
             if self.goals <= state:
                 return 0 # State is a goal state
             else:
                 # Goal not fully met, but no more nuts to tighten?
                 # Return 1 as a minimum cost if not goal state.
                 return 1

        # --- Initialize Simulation ---
        h = 0
        sim_man_loc = current_man_loc
        # sim_carrying_usable already set
        # sim_available_spanners already set (make a copy for simulation)
        sim_available_spanners_copy = sim_available_spanners.copy()
        sim_nuts_remaining = nuts_to_tighten.copy()

        # --- Iterative Action Estimation ---
        while sim_nuts_remaining:
            if sim_man_loc is None: # Man's location unknown, cannot proceed
                 return float('inf')

            if sim_carrying_usable:
                # Find closest remaining nut
                best_nut = None
                min_dist_to_nut = float('inf')
                target_nut_loc = None

                for nut in sim_nuts_remaining:
                    nut_loc = self.nut_locations.get(nut)
                    if nut_loc is None:
                         # This indicates an issue with initialization or state consistency
                         return float('inf') # Nut location unknown
                    dist = self.get_dist(sim_man_loc, nut_loc)
                    if dist < min_dist_to_nut:
                        min_dist_to_nut = dist
                        best_nut = nut
                        target_nut_loc = nut_loc

                if best_nut is None or min_dist_to_nut == float('inf'):
                    # No reachable nuts left
                    return float('inf')

                # Simulate walk and tighten
                h += min_dist_to_nut  # Walk to nut
                h += 1               # Tighten nut
                sim_man_loc = target_nut_loc
                sim_nuts_remaining.remove(best_nut)
                sim_carrying_usable = False # Spanner is now used

            else: # Not carrying a usable spanner
                if not sim_available_spanners_copy:
                    # No usable spanner carried and none available on ground
                    return float('inf') # Cannot tighten remaining nuts

                # Find closest available spanner
                best_spanner = None
                min_dist_to_spanner = float('inf')
                target_spanner_loc = None

                for spanner, loc in sim_available_spanners_copy.items():
                    dist = self.get_dist(sim_man_loc, loc)
                    if dist < min_dist_to_spanner:
                        min_dist_to_spanner = dist
                        best_spanner = spanner
                        target_spanner_loc = loc

                if best_spanner is None or min_dist_to_spanner == float('inf'):
                    # No reachable spanners
                    return float('inf')

                # Simulate walk to spanner and pickup
                h += min_dist_to_spanner # Walk to spanner
                h += 1                  # Pick up spanner
                sim_man_loc = target_spanner_loc
                # Remove spanner from available copy for the simulation
                del sim_available_spanners_copy[best_spanner]

                # Now carrying a usable spanner, find the closest remaining nut
                best_nut = None
                min_dist_to_nut = float('inf')
                target_nut_loc = None

                for nut in sim_nuts_remaining:
                    nut_loc = self.nut_locations.get(nut)
                    if nut_loc is None:
                         return float('inf')
                    # Distance from spanner location (current sim_man_loc)
                    dist = self.get_dist(sim_man_loc, nut_loc)
                    if dist < min_dist_to_nut:
                        min_dist_to_nut = dist
                        best_nut = nut
                        target_nut_loc = nut_loc

                if best_nut is None or min_dist_to_nut == float('inf'):
                    # Cannot reach any remaining nut from this spanner location
                    return float('inf')

                # Simulate walk to nut and tighten
                h += min_dist_to_nut # Walk from spanner loc to nut loc
                h += 1              # Tighten nut
                sim_man_loc = target_nut_loc
                sim_nuts_remaining.remove(best_nut)
                # sim_carrying_usable remains False as the spanner was just used

        # If the loop finished, all goal nuts were processed by the simulation.
        # Return the calculated cost.
        return h
