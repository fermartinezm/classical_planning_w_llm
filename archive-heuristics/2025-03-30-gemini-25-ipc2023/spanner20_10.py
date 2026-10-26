import collections
import itertools
from fnmatch import fnmatch
# Assuming the heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

# Helper functions (can be placed outside the class or as static methods)
def get_parts(fact):
    """
    Extract the components of a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(at bob shed)" -> ["at", "bob", "shed"]
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: A sequence of strings representing the pattern. Wildcards (*) are allowed.
    - Returns `True` if the fact structure and content match the pattern, `False` otherwise.
    """
    parts = get_parts(fact)
    # Check if the number of parts in the fact matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument (considering wildcards)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'spanner'.

    # Summary
    This heuristic estimates the remaining cost to tighten all goal nuts by simulating a sequential plan.
    It assumes a single man agent. For each loose nut that is part of the goal, it calculates the cost
    for the man to potentially fetch the nearest available usable spanner (if not already carrying one),
    travel to the nut's location, and perform the tighten action. The heuristic accounts for the
    consumption of usable spanners, ensuring one usable spanner is allocated per required tightening action.

    # Assumptions
    - There is exactly one agent of type 'man' in the problem instance.
    - 'link' predicates define a static, bidirectional graph of locations. Travel cost between linked locations is 1.
    - Spanners become permanently unusable after one 'tighten_nut' action. They are consumed by the process.
    - Nuts do not move from their initial locations. Spanners only move when picked up by the man.
    - The goal is solely defined by a conjunction of '(tightened nut)' predicates for specific nuts.
    - Object names might be used for basic type inference (e.g., 'spanner', 'nut') if full PDDL typing info isn't available during heuristic initialization.

    # Heuristic Initialization
    - The constructor identifies the single 'man' object, typically by looking at parameters of actions like 'walk' or 'pickup_spanner' or by checking the initial state.
    - It parses the static 'link' facts from the task definition (`task.static`) to build an adjacency list representation of the location graph.
    - It computes all-pairs shortest path distances between all locations using Breadth-First Search (BFS) on the graph and stores these distances (`self.dist`) for efficient lookup during heuristic evaluation.
    - It identifies the set of specific nuts (`self.goal_nuts`) that need to be tightened to satisfy the goal conditions (`task.goals`).

    # Step-By-Step Thinking for Computing Heuristic
    1.  Parse the current state (`node.state`) to extract dynamic information:
        - The man's current location (`man_loc`).
        - The spanner the man is carrying, if any (`carried_spanner`).
        - The set of spanners currently marked as `usable`.
        - The locations of all spanners currently on the ground (`spanner_locations`).
        - The locations of all nuts (`nut_locations`).
        - The set of nuts currently marked as `loose`.
    2.  Identify the set of `loose_goal_nuts` by intersecting the set of `loose_nuts` with the set of `goal_nuts` identified during initialization.
    3.  If `loose_goal_nuts` is empty, the goal state (with respect to tightening nuts) is reached, so return 0.
    4.  Determine the set of usable spanners currently available on the ground (`current_usable_spanners_on_ground`).
    5.  Check if the `carried_spanner` (if one exists) is in the set of `usable_spanners`. Let this be `is_carrying_usable`.
    6.  Calculate the `total_available_usable_spanners` = count of `current_usable_spanners_on_ground` + 1 (if `is_carrying_usable`).
    7.  If `total_available_usable_spanners` is less than the number of `loose_goal_nuts`, the goal is unreachable from this state due to insufficient spanners, so return infinity (`float('inf')`).
    8.  Initialize the heuristic estimate `h = 0`.
    9.  Initialize simulation variables to track the state during the estimation process:
        - `current_man_loc` = `man_loc` (from the actual state).
        - `sim_is_carrying_usable` = `is_carrying_usable` (from the actual state).
        - `sim_available_spanners` = A copy of the dictionary `current_usable_spanners_on_ground`.
    10. Create a sorted list of `loose_goal_nuts` (e.g., alphabetically) to process them in a deterministic order. This ensures the heuristic value is consistent for the same state.
    11. Iterate through each `nut` in the sorted list:
        a.  Get the location `nut_loc` of the current `nut`. Handle potential errors if the nut location isn't found.
        b.  Initialize costs for processing this nut: `cost_walk_to_spanner = 0`, `cost_pickup = 0`.
        c.  If `sim_is_carrying_usable` is False (man needs to acquire a usable spanner):
            i.   Check if `sim_available_spanners` is empty. If so, return infinity (indicates an inconsistency, as the initial check should have caught this).
            ii.  Find the `best_spanner` among those in `sim_available_spanners` that minimizes the walking distance from `current_man_loc`. Use the precomputed `self.dist`.
            iii. If no usable spanner is reachable (all distances are infinity), return infinity.
            iv.  Calculate `cost_walk_to_spanner` = shortest distance to `best_spanner`.
            v.   Set `cost_pickup = 1` (for the `pickup_spanner` action).
            vi.  Update the simulated man location: `current_man_loc` = location of `best_spanner`.
            vii. Update the simulated state: `sim_is_carrying_usable = True`.
            viii.Remove `best_spanner` from `sim_available_spanners` (it's now notionally picked up).
        d.  (At this point, the simulation assumes the man is at the location where he obtained/had a usable spanner) Calculate `cost_walk_to_nut` = shortest distance from `current_man_loc` to `nut_loc`. Use `self.dist`. If unreachable, return infinity.
        e.  Set `cost_tighten = 1` (for the `tighten_nut` action).
        f.  Add the costs incurred for this nut (`cost_walk_to_spanner + cost_pickup + cost_walk_to_nut + cost_tighten`) to the total heuristic value `h`.
        g.  Update the simulated man location: `current_man_loc` = `nut_loc`.
        h.  Update the simulated state: `sim_is_carrying_usable = False` (the spanner was just used and became unusable).
    12. After iterating through all `loose_goal_nuts`, return the final accumulated heuristic value `h`. If `h` became infinity at any point during the calculation, infinity is returned.
    """

    def __init__(self, task):
        """
        Initializes the heuristic. Precomputes distances and identifies goals.
        """
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # 1. Find the man (assuming one man)
        self.man = self._find_man(task)
        if self.man is None:
            # If man cannot be identified, the heuristic cannot function.
            raise ValueError("SpannerHeuristic: Could not identify the 'man' object in the task definition.")

        # 2. Build location graph and compute distances
        self.adj, self.locations = self._build_graph(self.static)
        if not self.locations:
             # Handle case with no locations or links - perhaps return inf always?
             print("Warning: SpannerHeuristic found no locations or links in static facts.")
             self.dist = {}
        else:
            # Precompute all-pairs shortest paths
            self.dist = self._compute_distances(self.locations, self.adj)

        # 3. Identify goal nuts from the task's goal specification
        self.goal_nuts = set()
        for goal_fact in self.goals:
             # Check if the goal fact is of the form (tightened ?n)
             if match(goal_fact, "tightened", "*"):
                 # Extract the nut name (second part of the fact)
                 self.goal_nuts.add(get_parts(goal_fact)[1])

    def _find_man(self, task):
        """
        Identifies the single 'man' object in the planning task.
        Tries to infer from operator parameters, falls back to checking
        initial state or common names like 'bob'.
        Returns the name of the man object or None if identification fails.
        """
        man_candidates = set()
        # Look through grounded operators provided by the task object
        for op in task.operators:
             # Operator name string includes parameters, e.g., "(walk shed location1 bob)"
             op_parts = get_parts(op.name)
             action_name = op_parts[0]
             params = op_parts[1:]
             # Infer man from parameter positions in known actions
             if action_name == 'walk' and len(params) == 3:
                 man_candidates.add(params[2]) # Man is the 3rd parameter
             elif action_name == 'pickup_spanner' and len(params) == 3:
                 man_candidates.add(params[2]) # Man is the 3rd parameter
             elif action_name == 'tighten_nut' and len(params) == 4:
                 man_candidates.add(params[2]) # Man is the 3rd parameter (m)

        if len(man_candidates) == 1:
            return man_candidates.pop()
        elif len(man_candidates) > 1:
             print(f"Warning: Found multiple potential man candidates: {man_candidates}. Heuristic might be incorrect.")
             # Arbitrarily pick one if multiple found, or could raise error
             return list(man_candidates)[0]

        # Fallback 1: Check initial state for known man names or patterns
        for fact in task.initial_state:
            if match(fact, "at", "bob", "*"): # Check for common name 'bob'
                return "bob"
            # Check if an object is involved in 'carrying' predicate in init state
            if match(fact, "carrying", "*", "*"):
                # Assume the first argument of carrying is the man
                return get_parts(fact)[1]

        # Fallback 2: Check task objects if available (not standard in this interface)

        print("Warning: SpannerHeuristic could not reliably determine man's name.")
        return None # Indicate failure

    def _build_graph(self, static_facts):
        """
        Builds an adjacency list representation of the location graph
        from static 'link' predicates.
        Returns a tuple: (adjacency_list, set_of_locations).
        """
        adj = collections.defaultdict(list)
        locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2 = parts[1], parts[2]
                # Add edges for both directions assuming links are bidirectional
                adj[loc1].append(loc2)
                adj[loc2].append(loc1)
                # Add locations to the set
                locations.add(loc1)
                locations.add(loc2)
        return adj, locations

    def _compute_distances(self, locations, adj):
        """
        Computes all-pairs shortest paths using BFS for an unweighted graph.
        Returns a dictionary `dist[start_loc][end_loc]` containing distances.
        Distance is `float('inf')` if locations are disconnected.
        """
        # Initialize distance matrix with infinity
        dist = {loc: {other: float('inf') for other in locations} for loc in locations}

        for start_node in locations:
            # Check if start_node is valid (should be if locations set is correct)
            if start_node not in dist: continue
            dist[start_node][start_node] = 0 # Distance to self is 0
            queue = collections.deque([start_node])
            # visited should be reset for each BFS starting from a new node
            visited = {start_node}

            while queue:
                u = queue.popleft()
                # Explore neighbors
                for v in adj.get(u, []):
                    # Ensure neighbor is a valid location and not visited in this BFS run
                    if v in locations and v not in visited:
                        visited.add(v)
                        # Update distance based on distance to predecessor
                        dist[start_node][v] = dist[start_node][u] + 1
                        queue.append(v)
        return dist

    def __call__(self, node):
        """
        Calculate the heuristic value (estimated cost to goal) for the given state node.
        """
        state = node.state

        # --- State Parsing ---
        man_loc = None
        carried_spanner = None
        usable_spanners = set()
        spanner_locations = {} # Location of spanners on the ground
        nut_locations = {}     # Location of nuts
        loose_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            # Use match for robustness against spacing issues etc.
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj == self.man:
                    man_loc = loc
                # Basic type inference by object name prefix (improve if full type info available)
                # This assumes standard naming conventions from examples.
                elif "spanner" in obj:
                    spanner_locations[obj] = loc
                elif "nut" in obj:
                    nut_locations[obj] = loc
            elif match(fact, "carrying", "*", "*"):
                # Ensure the man carrying is the one we identified
                if parts[1] == self.man:
                    carried_spanner = parts[2]
            elif match(fact, "usable", "*"):
                usable_spanners.add(parts[1])
            elif match(fact, "loose", "*"):
                loose_nuts.add(parts[1])

        # --- Pre-computation Checks ---
        # Check if man's location was found; critical for heuristic calculation.
        if man_loc is None:
             print(f"Error: SpannerHeuristic could not find location for man '{self.man}' in state: {state}")
             return float('inf') # Cannot compute heuristic without man's location

        # Check if distance matrix is available
        if not self.dist and self.locations:
             print("Error: SpannerHeuristic distance matrix is missing or empty.")
             return float('inf')


        # --- Heuristic Calculation ---
        # Identify which of the currently loose nuts are relevant to the goal
        loose_goal_nuts = {n for n in loose_nuts if n in self.goal_nuts}

        # If no goal nuts are loose, the goal is achieved (or this part of it)
        if not loose_goal_nuts:
            return 0

        # Identify currently available usable spanners on the ground
        current_usable_spanners_on_ground = {
            s: l for s, l in spanner_locations.items() if s in usable_spanners
        }
        # Check if the spanner carried by the man (if any) is usable
        is_carrying_usable = (carried_spanner is not None and carried_spanner in usable_spanners)

        # Calculate the total number of usable spanners available now
        total_available_usable_spanners = len(current_usable_spanners_on_ground) + (1 if is_carrying_usable else 0)
        num_needed = len(loose_goal_nuts)

        # Check if there are enough spanners to achieve the goal
        if total_available_usable_spanners < num_needed:
            # Not enough usable spanners left to tighten all remaining goal nuts
            return float('inf')

        # --- Simulation of sequential tightening ---
        h = 0 # Initialize heuristic cost estimate
        current_man_loc = man_loc # Track simulated man location
        # Make copies of mutable state for the simulation
        sim_available_spanners = current_usable_spanners_on_ground.copy()
        sim_is_carrying_usable = is_carrying_usable

        # Process nuts one by one in a deterministic order (e.g., sorted alphabetically)
        # This ensures the heuristic is a stable function of the state.
        sorted_nuts = sorted(list(loose_goal_nuts))

        for nut in sorted_nuts:
            nut_loc = nut_locations.get(nut)
            # If nut location is somehow missing, we cannot proceed.
            if nut_loc is None:
                print(f"Warning: SpannerHeuristic - Location for nut '{nut}' not found in state. State: {state}")
                return float('inf')

            cost_pickup = 0
            cost_walk_to_spanner = 0
            cost_walk_to_nut = 0
            spanner_loc_for_pickup = None # Track where the spanner was picked up

            # --- Step 1: Acquire a usable spanner if needed ---
            if not sim_is_carrying_usable:
                # Man needs to find and pick up a usable spanner from the ground.
                if not sim_available_spanners:
                    # This condition should theoretically be prevented by the initial check,
                    # but include as a safeguard against potential logic errors.
                    print("Error: SpannerHeuristic inconsistency - No available spanners during simulation loop.")
                    return float('inf')

                best_spanner = None
                min_dist_to_spanner = float('inf')

                # Find the usable spanner on the ground closest to the current simulated man location
                for spanner, spanner_loc in sim_available_spanners.items():
                    # Ensure locations are valid keys in the precomputed distance matrix
                    if current_man_loc not in self.dist or spanner_loc not in self.dist[current_man_loc]:
                         # This might happen if a location exists in the state but wasn't in static 'link' facts
                         print(f"Warning: SpannerHeuristic - Location '{current_man_loc}' or '{spanner_loc}' not found in distance matrix.")
                         dist_to_spanner = float('inf')
                    else:
                        dist_to_spanner = self.dist[current_man_loc][spanner_loc]

                    # Update if this spanner is closer
                    if dist_to_spanner < min_dist_to_spanner:
                        min_dist_to_spanner = dist_to_spanner
                        best_spanner = spanner
                        spanner_loc_for_pickup = spanner_loc # Store location for updating man's pos

                # Check if a reachable spanner was found
                if best_spanner is None or min_dist_to_spanner == float('inf'):
                    # Cannot reach any available usable spanner from current simulated location
                    return float('inf')

                # Add costs for getting the spanner
                cost_walk_to_spanner = min_dist_to_spanner
                cost_pickup = 1 # Cost of the pickup action

                # Update simulation state: man moves, picks up, spanner removed from ground
                current_man_loc = spanner_loc_for_pickup
                sim_is_carrying_usable = True
                del sim_available_spanners[best_spanner]

            # --- Step 2: Move to the nut ---
            # Man is now assumed to be carrying a usable spanner (either started with one or just picked one up).
            # Calculate cost to walk from current location to the nut's location.
            if current_man_loc not in self.dist or nut_loc not in self.dist[current_man_loc]:
                 print(f"Warning: SpannerHeuristic - Location '{current_man_loc}' or '{nut_loc}' not found in distance matrix.")
                 dist_to_nut = float('inf')
            else:
                dist_to_nut = self.dist[current_man_loc][nut_loc]

            # If the nut location is unreachable from the current simulated location
            if dist_to_nut == float('inf'):
                return float('inf')

            cost_walk_to_nut = dist_to_nut

            # --- Step 3: Tighten the nut ---
            cost_tighten = 1 # Cost of the tighten action

            # --- Update total cost and simulation state ---
            h += cost_walk_to_spanner + cost_pickup + cost_walk_to_nut + cost_tighten

            # Update simulated man location to the nut's location
            current_man_loc = nut_loc
            # Mark that the man is no longer carrying a usable spanner (it was just used)
            sim_is_carrying_usable = False

        # After iterating through all required nuts, return the total estimated cost
        # Final check for infinity just in case (though intermediate checks should handle it)
        if h == float('inf'):
             return float('inf')

        return h
