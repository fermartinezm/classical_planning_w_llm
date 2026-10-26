import collections
from fnmatch import fnmatch
import itertools

# Try to import the base class - handle if it doesn't exist for standalone execution
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a mock base class if the import fails (e.g., for testing)
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

# Helper functions
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(pred arg1 arg2)" -> ["pred", "arg1", "arg2"]
    Handles potential surrounding whitespace.
    Returns an empty list if the fact is malformed (e.g., not starting/ending with parentheses).
    """
    fact = fact.strip()
    if not fact.startswith("(") or not fact.endswith(")"):
        return []
    return fact[1:-1].split()

def match(fact_parts, *args):
    """
    Checks if a list of fact parts matches a pattern (supports '*' wildcard).
    `fact_parts` should be the result of `get_parts(fact)`.
    """
    if len(fact_parts) != len(args):
        return False
    # Use fnmatch for wildcard support if needed, otherwise simple equality check
    # Using simple equality check here as wildcards are used explicitly with '*'
    return all(p == a or a == '*' for p, a in zip(fact_parts, args))


class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the total number of actions required to tighten all
    loose nuts specified in the goal. It calculates the minimum estimated cost
    for each potential pairing of a loose goal nut and a usable spanner. It then
    uses a greedy assignment strategy (minimum cost pairing first) to match each
    loose nut to a unique usable spanner and sums the costs of these assignments.
    The cost reflects the sequence of actions: walk, pickup (if needed), walk, tighten.

    # Assumptions
    - There is exactly one man object in the problem instance. The heuristic identifies this man.
    - Nuts do not change location; their locations are considered static information derived from the initial state or static facts.
    - The `link` predicates define the connectivity between locations. Links are assumed to be bidirectional. The graph might be disconnected.
    - Each `tighten_nut` action requires one usable spanner and consumes its usability (`(not (usable ?s))`). The heuristic respects this by assigning each loose nut to a unique usable spanner.
    - The heuristic calculates costs based on the current state. For each nut-spanner assignment, it assumes the man starts at his current location and performs the necessary sequence of actions. It does not model the man's location change after completing one task when calculating costs for subsequent potential tasks (standard for state-based heuristics).

    # Heuristic Initialization
    - Identifies all relevant objects: the single man, nuts, spanners, and locations by analyzing the initial state, static facts, and operator definitions.
    - Parses static `link` facts to build an adjacency list representation of the location graph.
    - Computes all-pairs shortest path distances between locations using Breadth-First Search (BFS). Stores `float('inf')` for unreachable pairs.
    - Identifies the set of goal nuts (those needing the `(tightened ?n)` predicate) and stores their fixed locations.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unsatisfied Goals:** Find all goal nuts `?n` for which `(tightened ?n)` is a goal but `(loose ?n)` is true in the current state. If no such nuts exist, the heuristic value is 0.
    2.  **Locate Man:** Determine the current location of the man (`man_loc`) from the `(at man loc)` fact in the state.
    3.  **Find Usable Resources:** Identify all currently usable spanners. Note which one (if any) is being carried by the man (`carried_spanner`) and which are on the ground at specific locations (`usable_spanners_at_loc`).
    4.  **Calculate Potential Task Costs:** Create a list of potential 'tightening tasks'. For every pair `(nut, spanner)` where `nut` is a loose goal nut and `spanner` is a usable spanner:
        a.  Calculate the estimated cost to tighten `nut` using `spanner`, assuming the man starts at `man_loc`.
        b.  **If `spanner` is carried:** The cost is `distance(man_loc, nut_loc) + 1` (1 walk action sequence + 1 tighten action). Check if `nut_loc` is reachable from `man_loc`.
        c.  **If `spanner` is on the ground at `spanner_loc`:** The cost is `distance(man_loc, spanner_loc) + 1 + distance(spanner_loc, nut_loc) + 1` (walk to spanner + 1 pickup + walk to nut + 1 tighten). Check reachability for both path segments.
        d.  If the calculated cost is finite (paths exist), add the task `{'cost': cost, 'nut': nut, 'spanner': spanner}` to a list.
    5.  **Check Feasibility:** If no tasks with finite cost were generated but there are loose goal nuts, it implies the goals are unreachable from the current state; return `float('inf')`. Also, check if the number of unique usable spanners available is less than the number of loose goal nuts; if so, return `float('inf')`.
    6.  **Greedy Assignment:** Sort the list of potential tasks by `cost` in ascending order. Initialize `total_heuristic_cost = 0`, `assigned_nuts = set()`, `assigned_spanners = set()`.
    7.  Iterate through the sorted tasks: For each task `(cost, nut, spanner)`:
        a.  If `nut` is not in `assigned_nuts` AND `spanner` is not in `assigned_spanners`:
            i.  Mark `nut` as assigned by adding it to `assigned_nuts`.
            ii. Mark `spanner` as assigned by adding it to `assigned_spanners`.
            iii.Add `cost` to `total_heuristic_cost`.
    8.  **Final Check:** After iterating through all tasks, verify if the number of assigned nuts equals the total number of loose goal nuts identified in step 1.
        a.  If yes, return `total_heuristic_cost`.
        b.  If no (meaning some loose nuts could not be assigned a spanner due to resource contention or earlier reachability issues missed), return `float('inf')`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information from the task.
        """
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Identify Objects ---
        self.locations = set()
        self.nuts = set()
        self.spanners = set()
        potential_men = set()
        locatables = set() # Objects that can have a location

        # Gather objects and types from initial state and static facts
        all_facts = initial_state.union(static_facts)
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            if pred == 'at' and len(args) == 2:
                locatables.add(args[0])
                self.locations.add(args[1])
            elif pred == 'carrying' and len(args) == 2:
                potential_men.add(args[0])
                self.spanners.add(args[1])
                locatables.add(args[0]) # Man is locatable
                locatables.add(args[1]) # Spanner is locatable (when not carried)
            elif pred == 'usable' and len(args) == 1:
                self.spanners.add(args[0])
                locatables.add(args[0]) # Spanner is locatable
            elif pred == 'link' and len(args) == 2:
                self.locations.add(args[0])
                self.locations.add(args[1])
            elif (pred == 'tightened' or pred == 'loose') and len(args) == 1:
                self.nuts.add(args[0])
                locatables.add(args[0]) # Nut is locatable

        # Determine the man: the locatable object that isn't a nut, spanner, or location
        men_candidates = locatables - self.nuts - self.spanners - self.locations

        if len(men_candidates) != 1:
             # Fallback: Check parameters of actions like 'walk' or 'carrying' predicates
             action_men = set()
             for op in task.operators:
                 # Check 'carrying' preconditions/effects
                 for fact_str in op.preconditions.union(op.add_effects).union(op.del_effects):
                     fact_parts = get_parts(fact_str)
                     if fact_parts and fact_parts[0] == 'carrying' and len(fact_parts) == 3:
                         # Check if the first argument is not a known spanner/nut/location
                         potential_man = fact_parts[1]
                         if potential_man not in self.spanners and \
                            potential_man not in self.nuts and \
                            potential_man not in self.locations:
                             action_men.add(potential_man)
                 # This logic might still be ambiguous if types aren't strict
             men_candidates = action_men

        if len(men_candidates) != 1:
            raise ValueError(f"SpannerHeuristic requires exactly one man. Auto-detection failed. Found candidates: {men_candidates}")
        self.man = list(men_candidates)[0]

        # --- Store Goal Nuts and Their Locations ---
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "tightened" and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        self.nut_locations = {}
        for fact in all_facts:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] in self.nuts:
                 # Store nut location, ensuring consistency if mentioned multiple times
                 if parts[1] not in self.nut_locations:
                      self.nut_locations[parts[1]] = parts[2]
                 elif self.nut_locations[parts[1]] != parts[2]:
                      raise ValueError(f"Inconsistent location for nut {parts[1]}: {self.nut_locations[parts[1]]} vs {parts[2]}")

        # Verify all goal nuts have known locations
        for nut in self.goal_nuts:
            if nut not in self.nut_locations:
                 raise ValueError(f"Location of goal nut '{nut}' not found in initial/static state.")
            # Also ensure the nut's location is in the set of known locations
            self.locations.add(self.nut_locations[nut])

        # --- Build Location Graph and Compute Distances using BFS ---
        adj = collections.defaultdict(list)
        found_locations_from_links = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) == 3:
                u, v = parts[1], parts[2]
                # Add locations from links to the known set
                self.locations.add(u)
                self.locations.add(v)
                adj[u].append(v)
                adj[v].append(u) # Assume links are symmetric/bidirectional

        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))

        # Run BFS from each known location
        for start_node in self.locations:
            self.distances[start_node][start_node] = 0
            queue = collections.deque([start_node])
            visited_dist = {start_node: 0} # Store distance in visited dict

            while queue:
                u = queue.popleft()
                current_dist = visited_dist[u]

                # Check neighbors from adjacency list
                if u in adj:
                    for v in adj[u]:
                        # Important: Check if neighbor v is a known location
                        if v in self.locations and v not in visited_dist:
                            visited_dist[v] = current_dist + 1
                            self.distances[start_node][v] = current_dist + 1
                            queue.append(v)

    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        """
        state = node.state

        # --- Identify state specifics ---
        loose_goal_nuts = set()
        for nut in self.goal_nuts:
            # Use exact string matching for state facts
            if f"(loose {nut})" in state:
                loose_goal_nuts.add(nut)

        if not loose_goal_nuts:
            return 0 # Goal reached

        # Find man's location
        man_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3 and parts[1] == self.man:
                man_loc = parts[2]
                break
        if man_loc is None or man_loc not in self.locations:
             # Man's location not found or invalid
             return float('inf')

        # Find usable spanners (location or carried)
        usable_spanners_at_loc = {}
        all_usable_spanners = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            # Check for usable spanner predicate first
            if parts[0] == "usable" and len(parts) == 2 and parts[1] in self.spanners:
                usable_spanner = parts[1]
                all_usable_spanners.add(usable_spanner)
                # Now find its location (if not carried)
                found_at_loc = False
                for loc_fact in state:
                    loc_parts = get_parts(loc_fact)
                    if loc_parts and loc_parts[0] == "at" and len(loc_parts) == 3 and loc_parts[1] == usable_spanner:
                        usable_spanners_at_loc[usable_spanner] = loc_parts[2]
                        found_at_loc = True
                        break
                # If usable but not found at a location, it might be carried (check later)

        carried_spanner = None
        is_carried_spanner_usable = False
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "carrying" and len(parts) == 3 and parts[1] == self.man:
                carried_spanner = parts[2]
                if carried_spanner in all_usable_spanners:
                    is_carried_spanner_usable = True
                    # Remove from ground spanners if it was listed there erroneously
                    if carried_spanner in usable_spanners_at_loc:
                        del usable_spanners_at_loc[carried_spanner]
                break # Man carries at most one spanner

        # --- Feasibility Check ---
        current_usable_spanners = set(usable_spanners_at_loc.keys())
        if is_carried_spanner_usable:
            current_usable_spanners.add(carried_spanner)

        if not current_usable_spanners and loose_goal_nuts:
             return float('inf') # No usable spanners available at all
        if len(loose_goal_nuts) > len(current_usable_spanners):
             return float('inf') # Not enough unique usable spanners for the loose nuts

        # --- Calculate costs for potential tightening tasks ---
        tasks = []
        for nut in loose_goal_nuts:
            nut_loc = self.nut_locations[nut]
            if nut_loc not in self.locations: continue # Skip if nut location is invalid

            # Cost using carried spanner (if usable)
            if is_carried_spanner_usable:
                dist_man_nut = self.distances[man_loc][nut_loc]
                if dist_man_nut != float('inf'): # Check reachability
                    cost = dist_man_nut + 1 # walk + tighten
                    tasks.append({'cost': cost, 'nut': nut, 'spanner': carried_spanner})

            # Cost using ground spanners
            for spanner, spanner_loc in usable_spanners_at_loc.items():
                if spanner_loc not in self.locations: continue # Skip if spanner location is invalid

                dist_man_spanner = self.distances[man_loc][spanner_loc]
                dist_spanner_nut = self.distances[spanner_loc][nut_loc]

                if dist_man_spanner != float('inf') and dist_spanner_nut != float('inf'):
                    cost = dist_man_spanner + 1 + dist_spanner_nut + 1 # walk + pickup + walk + tighten
                    tasks.append({'cost': cost, 'nut': nut, 'spanner': spanner})

        # --- Greedy Assignment ---
        if not tasks and loose_goal_nuts:
             # No reachable path found for any nut-spanner pair
             return float('inf')

        tasks.sort(key=lambda x: x['cost'])

        assigned_nuts = set()
        assigned_spanners = set()
        total_assigned_cost = 0
        assignments_made = 0

        for task in tasks:
            nut = task['nut']
            spanner = task['spanner']
            cost = task['cost']

            # Check if both nut and spanner are available for assignment
            if nut in loose_goal_nuts and nut not in assigned_nuts and \
               spanner in current_usable_spanners and spanner not in assigned_spanners:

                assigned_nuts.add(nut)
                assigned_spanners.add(spanner)
                total_assigned_cost += cost
                assignments_made += 1

            # Optimization: if all nuts are assigned, stop.
            if assignments_made == len(loose_goal_nuts):
                break

        # If not all loose nuts could be assigned a spanner path, return infinity
        if assignments_made < len(loose_goal_nuts):
            return float('inf')

        return total_assigned_cost

