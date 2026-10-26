import math
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts like "(predicate arg1 arg2)"
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    return fact[1:-1].split()

# Helper function to match facts against a pattern with exact arity
def match(fact, *args):
    """Checks if a fact matches a pattern with exact arity."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts
    that are currently loose. It simulates a greedy strategy where the man
    iteratively chooses the "cheapest" next nut to tighten. The cost calculation
    considers travel costs (number of 'walk' actions) to the required spanner
    and the nut, plus the 'pickup_spanner' and 'tighten_nut' actions.
    Each tighten action requires and consumes a unique usable spanner.

    # Assumptions
    - There is exactly one 'man' agent in the problem.
    - Nuts do not move; their locations are fixed as defined in the initial state.
    - 'link' predicates define symmetric paths between locations, and each 'walk'
      action between linked locations has a cost of 1.
    - The goal is solely defined by a conjunction of `(tightened nut)` predicates.
    - Each `(tightened nut)` goal requires one initially usable spanner. The problem
      is assumed solvable only if the number of initially usable spanners is at
      least the number of goal nuts.

    # Heuristic Initialization
    - Identifies the man object's name based on common usage ('bob') or predicates
      in the initial state.
    - Extracts the set of nuts that need to be tightened in the goal state (`self.goal_nuts`).
    - Extracts the static locations of all nuts from the initial state (`self.nut_locations`).
    - Builds a graph representation of all locations based on `link` predicates and
      locations found in the initial state.
    - Computes all-pairs shortest paths (number of walk actions) between locations
      using Breadth-First Search (BFS) and stores the distances in `self.distances`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify Goal Nuts: Determine which nuts need to be tightened based on the task goals (`self.goal_nuts`).
    2.  Identify Current State (in `__call__`):
        - Find the man's current location (`man_loc`).
        - Check if the man is carrying a spanner (`carrying_spanner`) and if that spanner is currently usable (`is_carried_spanner_usable`).
        - Find all nuts relevant to the goal that are currently loose (`loose_nuts`).
        - Find all spanners that are currently usable and their locations on the ground (`usable_spanners_at_locs`).
    3.  Determine Target Nuts: Create the set `target_nuts` containing nuts that are both in `self.goal_nuts` and `loose_nuts`.
    4.  Check Goal Completion: If `target_nuts` is empty, check if the current state satisfies all goal conditions (`self.goals <= state`).
        - If yes (state is a goal state), return 0.
        - If no (state is not a goal state, but all required nuts are tight), return 1 to indicate it's not the goal but requires no further nut-tightening actions according to this heuristic.
    5.  Initialize Cost: Set the heuristic estimate `h = 0`.
    6.  Initialize Simulation State:
        - `remaining_nuts`: A copy of `target_nuts`.
        - `available_spanners`: A copy of `usable_spanners_at_locs` (dictionary mapping spanner name to location).
        - `current_man_loc`: The man's location from the current state.
        - `man_has_usable_spanner`: Boolean, true if the man starts carrying a usable spanner.
    7.  Iterative Tightening Simulation:
        a. While `remaining_nuts` is not empty:
        b. Check Man's Status: Is `man_has_usable_spanner` true?
        c. Case 1: Man has a usable spanner.
            i.  Find the nut `n_next` in `remaining_nuts` that is closest to `current_man_loc`. Distance is measured in walk actions using precomputed `self.distances` based on static nut locations.
            ii. If no remaining nut is reachable from `current_man_loc`, return `math.inf` (dead end).
            iii.Calculate travel distance `dist = self.get_dist(current_man_loc, self.nut_locations[n_next])`.
            iv. Add `dist` (cost of walk actions) + 1 (cost of tighten action) to `h`.
            v.  Update `current_man_loc` to the location of `n_next`.
            vi. Remove `n_next` from `remaining_nuts`.
            vii.Set `man_has_usable_spanner = False` (as the spanner was just used).
        d. Case 2: Man does not have a usable spanner.
            i.  Check if `available_spanners` (usable spanners on the ground) is empty. If so, return `math.inf` (cannot tighten remaining nuts).
            ii. Find the combination of an available spanner `s` (at location `ls`) and a remaining nut `n` (at location `ln`) that minimizes the total estimated cost for the next cycle: `cost = self.get_dist(current_man_loc, ls) + self.get_dist(ls, ln)`.
            iii.If no such reachable combination exists (e.g., man is trapped, or remaining spanners/nuts are on disconnected graph components), return `math.inf`.
            iv. Let the best pair be `(best_s, best_n)` with locations `(best_ls, best_ln)`.
            v.  Calculate `dist1 = self.get_dist(current_man_loc, best_ls)` and `dist2 = self.get_dist(best_ls, best_ln)`.
            vi. Add `dist1` (walk) + 1 (pickup) + `dist2` (walk) + 1 (tighten) to `h`.
            vii.Update `current_man_loc` to `best_ln`.
            viii.Remove `best_n` from `remaining_nuts`.
            ix. Remove `best_s` from `available_spanners` (it was picked up and used).
            x.  Keep `man_has_usable_spanner = False` (the spanner was used immediately).
    8.  Return Total Cost: Return the accumulated heuristic value `h`. If `h` is 0 after the loop but the state wasn't identified as a goal state initially, the final check ensures 1 is returned.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Find the man object name
        # Prioritize finding 'bob' as it's common in examples
        self.man = None
        for fact in initial_state:
            # Assuming man is 'bob' if 'bob' is at some location initially
            if match(fact, "at", "bob", "?l"):
                self.man = "bob"
                break
        # If not bob, try finding from carrying predicates
        if not self.man:
            for fact in initial_state:
                 if match(fact, "carrying", "?m", "?s"):
                      # Assume the object carrying something is the man
                      self.man = get_parts(fact)[1]
                      break
        # If still not found, try finding from 'at' predicates (less reliable)
        if not self.man:
             for fact in initial_state:
                  if match(fact, "at", "?m", "?l"):
                       # Weak assumption: first object found at a location might be the man
                       # This is risky if types are mixed.
                       # Let's assume 'man' type objects appear first or are named distinctively.
                       potential_man = get_parts(fact)[1]
                       # Add checks for typical non-man names if needed, e.g. 'spanner', 'nut'
                       if not potential_man.startswith('spanner') and not potential_man.startswith('nut'):
                            self.man = potential_man
                            break # Take the first plausible candidate

        if not self.man:
             # If no man identified, raise error as the heuristic depends on it.
             raise ValueError("Could not determine the man's name from the initial state.")


        # 2. Find goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "?n"):
                self.goal_nuts.add(get_parts(goal)[1])

        # 3. Find nut locations (assume static from initial state)
        self.nut_locations = {}
        all_locations_from_init = set() # Track all locations mentioned in init state

        for fact in initial_state:
             parts = get_parts(fact)
             if match(fact, "at", "?obj", "?l"):
                 obj, loc = parts[1], parts[2]
                 all_locations_from_init.add(loc)
                 # If the object is one of the nuts needed for the goal, store its location
                 if obj in self.goal_nuts:
                     self.nut_locations[obj] = loc

        # Verify all goal nuts found locations
        for nut in self.goal_nuts:
            if nut not in self.nut_locations:
                 # This indicates an issue with the problem definition or parsing
                 raise ValueError(f"Initial location for goal nut '{nut}' not found in initial state.")


        # 4. Build location graph and compute distances
        locations_from_links = set()
        adj = {} # Adjacency list for the location graph
        for fact in static_facts:
            if match(fact, "link", "?l1", "?l2"):
                l1, l2 = get_parts(fact)[1:]
                locations_from_links.add(l1)
                locations_from_links.add(l2)
                # Add edges in both directions assuming symmetry
                adj.setdefault(l1, set()).add(l2)
                adj.setdefault(l2, set()).add(l1)

        # Combine all known locations from links, initial object positions, and nut goal positions
        self.all_locations = locations_from_links.union(all_locations_from_init).union(set(self.nut_locations.values()))

        # Ensure adjacency list covers all locations, initializing empty neighbor sets for isolated locations
        for loc in self.all_locations:
            if loc not in adj:
                adj[loc] = set()

        # Compute all-pairs shortest paths using BFS
        self.distances = self._compute_all_pairs_shortest_paths(adj)

    def _compute_all_pairs_shortest_paths(self, adj):
        """Computes shortest path distances between all pairs of known locations using BFS."""
        distances = {}
        locations = list(self.all_locations)
        for start_node in locations:
            # Initialize distances from start_node to all other locations as infinity
            distances[start_node] = {loc: float('inf') for loc in locations}
            # Distance from a node to itself is 0
            if start_node in self.all_locations: # Check if start_node is a valid location
                 distances[start_node][start_node] = 0
                 queue = deque([start_node]) # Queue for BFS
                 visited = {start_node} # Set of visited locations for this BFS run

                 while queue:
                     current_node = queue.popleft()
                     # If current_node somehow isn't in adj (e.g., isolated), skip neighbors part
                     if current_node not in adj: continue

                     current_dist = distances[start_node][current_node]

                     # Explore neighbors
                     for neighbor in adj.get(current_node, set()):
                         # Process neighbor only if it's a known location and not visited yet
                         if neighbor in self.all_locations and neighbor not in visited:
                             visited.add(neighbor)
                             distances[start_node][neighbor] = current_dist + 1
                             queue.append(neighbor)
        return distances

    def get_dist(self, loc1, loc2):
        """Returns the shortest distance (number of walk actions) between two locations."""
        # Handle cases where locations might be None or not in the map
        if loc1 is None or loc2 is None:
             return float('inf')
        if loc1 not in self.all_locations or loc2 not in self.all_locations:
             return float('inf') # Location doesn't exist in the precomputed map
        # If locations are the same, distance is 0
        if loc1 == loc2:
            return 0

        # Retrieve precomputed distance; default to infinity if path doesn't exist
        dist = self.distances.get(loc1, {}).get(loc2, float('inf'))
        return dist


    def __call__(self, node):
        """Calculates the heuristic value for a given state node."""
        state = node.state

        # --- Preliminary Goal Check ---
        # Check if the current state satisfies all goal conditions
        is_goal_state = self.goals <= state
        if is_goal_state:
            return 0 # Goal state reached, heuristic value is 0

        # --- 1. Parse current state information ---
        man_loc = None
        carrying_spanner = None # Name of the spanner being carried, if any
        is_carried_spanner_usable = False # Is the carried spanner usable?
        loose_nuts = set() # Set of goal nuts that are currently loose
        usable_spanners_at_locs = {} # Dict: usable spanner name -> location (if on ground)

        # Find names of all spanners marked as usable in the current state
        usable_spanner_names = {get_parts(f)[1] for f in state if match(f, "usable", "*")}

        # Iterate through facts in the state to extract relevant information
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj == self.man:
                    man_loc = loc # Found man's location
                elif obj in usable_spanner_names:
                    # Found a usable spanner on the ground
                    usable_spanners_at_locs[obj] = loc
            elif predicate == "carrying" and parts[1] == self.man:
                # Found the spanner the man is carrying
                carrying_spanner = parts[2]
                # Check if this carried spanner is in the set of usable spanners
                if carrying_spanner in usable_spanner_names:
                    is_carried_spanner_usable = True
            elif predicate == "loose":
                nut = parts[1]
                # Check if this loose nut is one of the goal nuts
                if nut in self.goal_nuts:
                    loose_nuts.add(nut)

        # --- 2. Identify target nuts for this state ---
        # Target nuts are those that are required by the goal AND are currently loose
        target_nuts = {n for n in self.goal_nuts if n in loose_nuts}

        # --- 3. Check if nut-tightening part of the task is done ---
        if not target_nuts:
            # No more goal nuts are loose. Since we already checked if it's a
            # full goal state and it wasn't, return 1. This distinguishes it
            # from the true goal state (h=0) but indicates no more nut actions needed.
            return 1

        # --- 4. Initialize simulation variables ---
        h = 0 # Accumulated heuristic cost
        current_man_loc = man_loc # Man's location for simulation steps
        # Copy resources for simulation to avoid modifying original dicts/sets
        available_spanners = usable_spanners_at_locs.copy() # Usable spanners on ground
        remaining_nuts = target_nuts.copy() # Nuts still needing tightening
        man_has_usable_spanner = is_carried_spanner_usable # Man's carrying status

        # --- 5. Simulate the tightening process greedily ---
        while remaining_nuts:
            # If man's location is unknown at any point, indicates an issue.
            if current_man_loc is None:
                 return float('inf') # Cannot proceed without man's location

            # Case 1: Man is currently holding a usable spanner
            if man_has_usable_spanner:
                best_n = None # The nut to tighten next
                min_dist = float('inf') # Minimum distance to a remaining nut

                # Find the closest remaining nut
                for n in remaining_nuts:
                    ln = self.nut_locations.get(n) # Get static location of the nut
                    if ln is None: continue # Skip if location unknown (shouldn't happen)

                    dist = self.get_dist(current_man_loc, ln)
                    if dist < min_dist: # Found a closer nut
                        min_dist = dist
                        best_n = n

                # If no nut is reachable, return infinity (dead end)
                if best_n is None or min_dist == float('inf'):
                    return float('inf')

                # Add cost: walk to the nut + tighten the nut
                ln_next = self.nut_locations[best_n]
                travel_dist = min_dist
                h += travel_dist  # Cost of walk actions
                h += 1           # Cost of tighten action

                # Update simulation state
                current_man_loc = ln_next # Man moves to nut's location
                remaining_nuts.remove(best_n) # Nut is tightened
                man_has_usable_spanner = False # Spanner is now used (becomes unusable)

            # Case 2: Man is not holding a usable spanner
            else:
                # Check if there are any usable spanners left on the ground
                if not available_spanners:
                    # No spanners left, but nuts remain. Cannot solve.
                    return float('inf')

                # Find the best (spanner, nut) pair to target next
                # "Best" minimizes: dist(man->spanner) + dist(spanner->nut)
                min_combined_cost = float('inf')
                best_s = None # Best spanner to pick up
                best_n = None # Best nut to tighten with that spanner
                best_ls = None # Location of best_s
                best_ln = None # Location of best_n

                # Iterate through available spanners and remaining nuts
                for s, ls in available_spanners.items():
                    dist_man_to_s = self.get_dist(current_man_loc, ls)
                    # If spanner is unreachable, skip it
                    if dist_man_to_s == float('inf'): continue

                    for n in remaining_nuts:
                        ln = self.nut_locations.get(n)
                        if ln is None: continue # Skip nut if location unknown

                        dist_s_to_n = self.get_dist(ls, ln)
                        # If nut is unreachable from spanner location, skip pair
                        if dist_s_to_n == float('inf'): continue

                        # Calculate combined cost for this spanner-nut pair
                        combined_cost = dist_man_to_s + dist_s_to_n
                        # If this pair is better than the current best, update
                        if combined_cost < min_combined_cost:
                            min_combined_cost = combined_cost
                            best_s = s
                            best_n = n
                            best_ls = ls
                            best_ln = ln

                # If no reachable spanner/nut combination was found
                if best_s is None:
                    return float('inf') # Dead end

                # Add costs for the chosen sequence: walk->pickup->walk->tighten
                travel_to_spanner = self.get_dist(current_man_loc, best_ls)
                travel_to_nut = self.get_dist(best_ls, best_ln)

                h += travel_to_spanner  # Cost of walk actions to spanner
                h += 1                 # Cost of pickup action
                h += travel_to_nut     # Cost of walk actions to nut
                h += 1                 # Cost of tighten action

                # Update simulation state
                current_man_loc = best_ln # Man moves to nut's location
                remaining_nuts.remove(best_n) # Nut is tightened
                del available_spanners[best_s] # Spanner is picked up and used
                # Man remains without a usable spanner after this sequence
                man_has_usable_spanner = False

        # --- 6. Return final heuristic value ---
        # If the loop finished, all target nuts were accounted for in the cost `h`.
        # Since we checked for the actual goal state at the very beginning,
        # if h > 0, it represents the estimated cost to reach the goal.
        # If h == 0 here, it implies target_nuts was empty initially, and the
        # initial check already returned 1 because it wasn't a goal state.
        return h
