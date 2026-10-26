import collections
from fnmatch import fnmatch
# Assuming the Heuristic base class is available in the specified path
# If the execution environment differs, this might need adjustment.
from heuristics.heuristic_base import Heuristic
import itertools # Keep import in case needed later, though not used now.


# Helper function (can be defined inside the class or outside)
def get_parts(fact):
    """Extract the components of a PDDL fact string by removing parentheses
    and splitting by space."""
    # Handles facts like '(at bob shed)' -> ['at', 'bob', 'shed']
    return fact[1:-1].split()

# Helper function (can be defined inside the class or outside)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    The fact is a string like '(predicate obj1 obj2)'.
    Args are strings representing the pattern, e.g., "predicate", "obj1", "*".
    Wildcards (*) are supported via fnmatch.
    Returns True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    # Check if the number of parts in the fact matches the pattern length
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern element using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It simulates a greedy strategy where the single man repeatedly fetches the
    "best" available usable spanner (if needed) and uses it to tighten the
    "closest" remaining loose goal nut. The cost includes walking ('walk' actions),
    picking up spanners ('pickup_spanner' actions), and tightening nuts
    ('tighten_nut' actions).

    # Assumptions
    - There is exactly one man object in the problem instance. The heuristic identifies
      this man during initialization.
    - The `link` predicates define a potentially disconnected graph of locations.
      It's assumed that paths exist between relevant locations in solvable instances.
    - Links are treated as bidirectional (e.g., `(link l1 l2)` implies movement
      is possible from l1 to l2 and l2 to l1).
    - The goal is solely defined by `(tightened nut)` predicates for a subset of nuts.
    - Each `tighten_nut` action requires and consumes one `usable` spanner, making
      that specific spanner instance unusable afterwards.

    # Heuristic Initialization
    - Identifies the single man, all locations, nuts, and spanners by parsing
      the initial state, goals, and static facts (like `link`). It makes inferences
      based on predicate usage (e.g., the first argument of `carrying` is the man).
    - Parses the static `link` predicates to build an adjacency list representation
      of the location graph (`self.links`).
    - Computes all-pairs shortest paths (APSP) using Breadth-First Search (BFS)
      starting from each location. Distances (number of `walk` actions) are stored
      in `self.dist` (using `float('inf')` for unreachable pairs).
    - Identifies the set of nuts that must be in the `(tightened ?n)` state
      according to the task goal (`self.goal_nuts`).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Get Current State Info**: Parse the current state (`node.state`, a frozenset
        of facts) to determine:
        - The man's current location (`man_loc`).
        - The spanner carried by the man, if any (`carried_spanner`).
        - The set of all spanners currently marked as `usable` (`usable_spanners_state`).
        - The locations of all spanners currently on the ground (`spanners_at`).
        - The locations of all nuts (`nuts_at`).
        - The set of nuts currently marked as `loose` (`loose_nuts_state`).
    2.  **Identify Task**: Determine the set of goal nuts that are currently loose
        and their locations (`loose_goal_nuts_locs`). This represents the
        remaining subgoals to achieve.
    3.  **Goal Check**: If `loose_goal_nuts_locs` is empty, it means all goal nuts
        are already tightened (or were never loose), so the estimated cost to
        reach the goal is 0.
    4.  **Resource Check**: Determine which usable spanners are currently on the
        ground (`usable_spanners_on_ground_locs`) and whether the man is currently
        carrying a usable one (`man_has_usable_spanner_now`). Calculate the total
        number of usable spanners available (`num_usable_total`). If this number
        is less than the number of `loose_goal_nuts_locs`, the goal is unreachable
        from this state due to insufficient spanners, so return infinity.
    5.  **Initialize Simulation**: Set `total_cost = 0`. Initialize the simulation's
        man location `current_loc = man_loc`. Keep track of whether the simulated
        man `has_usable` spanner, initialized based on the actual current state.
        Create mutable copies of the dictionaries for `available_spanners` (on ground)
        and `remaining_nuts` (loose goal nuts) to track resource consumption
        during the simulation.
    6.  **Iterative Tightening Loop**: Loop as long as `remaining_nuts` is not empty:
        a.  **Select Nut**: Choose the `target_nut` from `remaining_nuts`. The
            greedy strategy is to pick the nut whose location (`target_loc`) is
            closest to the man's current simulated location (`current_loc`), based
            on the pre-computed shortest path distances (`self.dist`). Crucially,
            only consider nuts that are reachable (`distance != float('inf')`).
            If no remaining nuts are reachable from `current_loc`, the goal is
            unreachable from this state, return infinity.
        b.  **Check if Spanner Needed**: If the simulated man `has_usable` spanner
            is currently false:
            i.  **Check Availability**: If `available_spanners` (on the ground)
                is empty, return infinity (this indicates a potential logic error
                or an unsolvable state missed by the initial check).
            ii. **Find Best Spanner**: Search through the `available_spanners`.
                For each spanner `s` at location `l_s`, calculate the estimated
                cost of the route: man walks from `current_loc` to `l_s`, picks
                up `s`, then walks from `l_s` to the `target_loc`. The path cost
                component used for selection is `distance(current_loc, l_s) +
                distance(l_s, target_loc)`. Select the spanner `best_s` at
                `best_l_s` that minimizes this path cost, ensuring both path
                segments are reachable (`distance != float('inf')`).
            iii. **Check Reachability**: If no spanner provides a fully reachable
                 route (i.e., `best_s` remains `None`), return infinity.
            iv. **Update Cost & State**: Add the cost of walking to the chosen
                 spanner and picking it up (`distance(current_loc, best_l_s) + 1`)
                 to `total_cost`. Update the simulated man's location
                 `current_loc = best_l_s`. Set the simulation flag `has_usable = True`.
                 Remove `best_s` from the `available_spanners` dictionary, as it's
                 now notionally held by the man.
        c.  **Go Tighten Nut**: At this point, the simulated man is at `current_loc`
            and `has_usable` spanner is true.
            i.  **Calculate Cost**: The cost to walk to the `target_loc` and
                perform the tighten action is `distance(current_loc, target_loc) + 1`.
            ii. **Check Reachability**: If `target_loc` is unreachable from the
                 current `current_loc` (`distance == float('inf')`), return infinity.
                 (This reachability should implicitly be guaranteed by the spanner
                 selection logic if a spanner was just fetched, but it's checked
                 for robustness).
            iii. **Update Cost & State**: Add the calculated cost (walk + tighten)
                 to `total_cost`. Update the simulated man's location
                 `current_loc = target_loc`. Set the simulation flag `has_usable = False`
                 (as the tighten action consumes the spanner's usability). Remove
                 `target_nut` from the `remaining_nuts` dictionary.
    7.  **Return Total Cost**: After the loop finishes (all nuts initially in
        `loose_goal_nuts_locs` have been processed), return the final
        accumulated `total_cost` as the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information and pre-calculating
        shortest path distances between locations.
        """
        self.task = task
        self.goal_nuts = set()
        self.locations = set()
        self.spanners = set()
        self.nuts = set()
        self.man = None
        self.links = collections.defaultdict(list)
        # dist[loc1][loc2] will store the shortest distance (number of walks)
        self.dist = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))

        # --- Object and Static Fact Identification ---
        obj_types = collections.defaultdict(set)
        all_objects = set()

        # Combine static facts (if available) and initial state for parsing
        static_facts = getattr(task, 'static', frozenset())
        facts_to_parse = static_facts.union(task.initial_state).union(task.goals)

        # First pass: identify objects and their potential types, locations, links
        for fact in facts_to_parse:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]

            if pred == 'link' and len(args) == 2:
                loc1, loc2 = args
                self.links[loc1].append(loc2)
                self.links[loc2].append(loc1) # Assume bidirectional links
                self.locations.add(loc1)
                self.locations.add(loc2)
                obj_types['location'].add(loc1)
                obj_types['location'].add(loc2)
                all_objects.add(loc1)
                all_objects.add(loc2)
            elif pred == 'at' and len(args) == 2:
                obj, loc = args
                all_objects.add(obj)
                all_objects.add(loc)
                # Infer loc is a location
                self.locations.add(loc)
                obj_types['location'].add(loc)
            elif pred == 'carrying' and len(args) == 2:
                 # Infer types from usage: man carries spanner
                 man, spanner = args
                 self.man = man # Assume the first argument is the man
                 obj_types['man'].add(man)
                 obj_types['spanner'].add(spanner)
                 all_objects.add(man)
                 all_objects.add(spanner)
            elif pred in ('tightened', 'loose') and len(args) == 1:
                 # Infer type from usage: nuts are tightened/loose
                 nut = args[0]
                 obj_types['nut'].add(nut)
                 all_objects.add(nut)
            elif pred == 'usable' and len(args) == 1:
                 # Infer type from usage: spanners are usable
                 spanner = args[0]
                 obj_types['spanner'].add(spanner)
                 all_objects.add(spanner)

        # Assign identified objects to instance variables
        self.locations.update(obj_types['location'])
        self.spanners.update(obj_types['spanner'])
        self.nuts.update(obj_types['nut'])

        # Attempt to identify the man if not found via 'carrying' predicate
        if not self.man:
             # Objects that are not locations, spanners, or nuts are potential men
             potential_men = all_objects - self.locations - self.spanners - self.nuts
             if len(potential_men) == 1:
                 self.man = list(potential_men)[0]
             elif obj_types['man']: # Check if type was inferred e.g. from domain file (not parsed here)
                 self.man = list(obj_types['man'])[0]
             else:
                 # Fallback: Check 'at' predicates in init state for potential men
                 found_man_in_init = False
                 for fact in task.initial_state:
                     # Check facts like (at obj loc) where obj is in potential_men
                     if match(fact, "at", "*", "*"):
                         obj = get_parts(fact)[1]
                         if obj in potential_men:
                             self.man = obj
                             found_man_in_init = True
                             break
                 if not found_man_in_init and potential_men:
                     # If still ambiguous, print warning and pick one arbitrarily
                     print(f"Warning: Could not definitively identify the man among {potential_men}. Selecting '{list(potential_men)[0]}'.")
                     self.man = list(potential_men)[0]

        # Final checks for essential components identified during initialization
        if not self.man:
            raise ValueError("Heuristic Initialization Error: Could not identify the man object.")
        if not self.locations:
             # Try to gather locations from 'at' predicates if links didn't define any
             for fact in task.initial_state:
                 if match(fact, "at", "*", "*"):
                     self.locations.add(get_parts(fact)[2])
             if not self.locations:
                 raise ValueError("Heuristic Initialization Error: Could not identify any locations.")

        # Identify the specific nuts that need to be tightened for the goal
        for goal in task.goals:
            # Ensure the goal fact is '(tightened nut_name)'
            if match(goal, "tightened", "*") and len(get_parts(goal)) == 2:
                self.goal_nuts.add(get_parts(goal)[1])

        # --- Compute All-Pairs Shortest Paths (APSP) using BFS ---
        # Initialize distances for all pairs of known locations
        for l1 in self.locations:
            for l2 in self.locations:
                if l1 == l2:
                    self.dist[l1][l2] = 0
                else:
                    self.dist[l1][l2] = float('inf')

        # Run BFS from each location to find shortest paths to all other locations
        for start_node in self.locations:
            # Optimization: If start_node distance to itself is already 0, BFS is done.
            # The check `if self.dist[start_node][start_node] == 0:` is implicit.

            queue = collections.deque([(start_node, 0)]) # Queue stores (node, distance_from_start)
            # visited_dist map stores the shortest distance found so far from start_node
            visited_dist = {start_node: 0}

            while queue:
                current_node, distance = queue.popleft()

                # Explore neighbors connected by links
                for neighbor in self.links.get(current_node, []):
                    # Process neighbor only if it's a known location
                    if neighbor in self.locations:
                        # If neighbor not visited, or found a shorter path
                        if neighbor not in visited_dist or visited_dist[neighbor] > distance + 1:
                             visited_dist[neighbor] = distance + 1
                             # Update the global distance map
                             self.dist[start_node][neighbor] = distance + 1
                             queue.append((neighbor, distance + 1))


    def __call__(self, node):
        """
        Calculate the heuristic value (estimated cost to goal) for the given state node.
        """
        state = node.state

        # 1. Get Current State Info from the state frozenset
        man_loc = None
        carried_spanner = None
        usable_spanners_state = set() # Set of spanner names that are usable
        spanners_at = {} # Maps spanner name -> location (if on ground)
        nuts_at = {}     # Maps nut name -> location
        loose_nuts_state = set() # Set of nut names that are loose

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]

            if pred == 'at' and len(args) == 2:
                obj, loc = args
                if obj == self.man:
                    man_loc = loc
                elif obj in self.spanners:
                    spanners_at[obj] = loc
                elif obj in self.nuts:
                    nuts_at[obj] = loc
            elif pred == 'carrying' and len(args) == 2:
                # Check if it's the known man carrying the object
                if args[0] == self.man:
                    # Assume the carried object is a spanner based on domain actions
                    carried_spanner = args[1]
            elif pred == 'usable' and len(args) == 1:
                usable_spanners_state.add(args[0])
            elif pred == 'loose' and len(args) == 1:
                loose_nuts_state.add(args[0])

        # If man's location is somehow unknown in the state, heuristic is undefined.
        if man_loc is None:
             print(f"Warning: Man '{self.man}' location not found in state. Returning inf.")
             return float('inf')

        # 2. Identify Task: Find goal nuts that are currently loose
        loose_goal_nuts_locs = {n: l for n, l in nuts_at.items()
                                if n in self.goal_nuts and n in loose_nuts_state}

        # 3. Goal Check: If the set is empty, goal is reached.
        if not loose_goal_nuts_locs:
            return 0

        # 4. Resource Check: Ensure enough usable spanners exist.
        usable_spanners_on_ground_locs = {s: l for s, l in spanners_at.items()
                                          if s in usable_spanners_state}
        man_has_usable_spanner_now = (carried_spanner is not None and
                                      carried_spanner in usable_spanners_state)
        num_usable_total = len(usable_spanners_on_ground_locs) + (1 if man_has_usable_spanner_now else 0)

        # If fewer usable spanners than loose goal nuts, goal is impossible.
        if len(loose_goal_nuts_locs) > num_usable_total:
            return float('inf')

        # 5. Initialize Simulation state variables
        cost = 0 # Accumulated estimated cost
        current_loc = man_loc # Simulated man's location
        # Track if the simulated man is holding a usable spanner
        has_usable = man_has_usable_spanner_now
        # Create mutable copies for simulation: track available resources
        available_spanners = dict(usable_spanners_on_ground_locs)
        remaining_nuts = dict(loose_goal_nuts_locs)

        # 6. Iterative Tightening Loop: Simulate tightening nuts one by one
        while remaining_nuts:
            # 6a. Select Nut: Find the closest reachable nut to current_loc
            target_nut = None
            target_loc = None
            min_dist_to_nut = float('inf')

            # Iterate through remaining nuts to find the best target
            for n, l in remaining_nuts.items():
                dist_to_l = self.dist[current_loc][l]
                # Check reachability (dist < infinity) and find minimum distance
                if dist_to_l < min_dist_to_nut:
                    min_dist_to_nut = dist_to_l
                    target_nut = n
                    target_loc = l

            # If no remaining nuts are reachable from current_loc, goal is unreachable.
            if target_nut is None:
                 return float('inf')

            # 6b. Check if Spanner Needed and Fetch if Necessary
            if not has_usable:
                # If no spanners left on ground, indicates an issue.
                if not available_spanners:
                    # This case should ideally be caught by the initial resource check.
                    print("Error: Ran out of available spanners during simulation unexpectedly.")
                    return float('inf')

                # Find Best Spanner: minimize travel dist(current->spanner->target)
                best_s = None # Name of the best spanner to pick up
                best_l_s = None # Location of the best spanner
                min_pickup_route_cost = float('inf') # Min cost for the path segment

                # Evaluate each available spanner on the ground
                for s, l_s in available_spanners.items():
                    cost_to_spanner = self.dist[current_loc][l_s]
                    cost_spanner_to_nut = self.dist[l_s][target_loc]

                    # Check if path current_loc -> l_s -> target_loc is possible
                    if cost_to_spanner != float('inf') and cost_spanner_to_nut != float('inf'):
                        # Calculate the path cost (walking only) for comparison
                        route_cost = cost_to_spanner + cost_spanner_to_nut
                        if route_cost < min_pickup_route_cost:
                            min_pickup_route_cost = route_cost
                            best_s = s
                            best_l_s = l_s

                # If no spanner provides a reachable path to the target nut
                if best_s is None:
                    return float('inf')

                # Simulate walking to the best spanner and picking it up
                # Cost = walk actions + 1 pickup action
                cost_to_pickup = self.dist[current_loc][best_l_s] + 1
                cost += cost_to_pickup
                current_loc = best_l_s # Update simulated man location
                has_usable = True      # Man now holds a usable spanner in simulation
                del available_spanners[best_s] # Remove spanner from available ground spanners

            # 6c. Go Tighten Nut
            # At this point, the simulated man is at current_loc and has_usable is True.
            # Cost = walk actions + 1 tighten action
            cost_to_tighten = self.dist[current_loc][target_loc] + 1

            # Check reachability to target nut location
            if cost_to_tighten == float('inf'):
                 # This might happen if graph is disconnected in unexpected ways
                 return float('inf')

            # Simulate walking to the nut and tightening it
            cost += cost_to_tighten
            current_loc = target_loc # Update simulated man location
            has_usable = False       # Spanner becomes unusable after tightening
            del remaining_nuts[target_nut] # Remove this nut from the set of tasks

        # 7. Return Total Cost accumulated during simulation
        return cost
