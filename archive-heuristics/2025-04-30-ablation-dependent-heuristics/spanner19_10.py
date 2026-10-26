from collections import deque
import math

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal state (all goal nuts tightened)
    by simulating a greedy process. The simulation involves repeatedly
    finding the nearest usable spanner (if not carrying one), picking it up,
    then finding the nearest loose goal nut, walking to it, and tightening it.
    The heuristic sums the walk distances and the costs of pickup (1) and
    tighten (1) actions performed in this simulation. This provides an estimate
    of the total number of actions required.

    Assumptions:
    - There is exactly one man object in the domain.
    - Links between locations defined by the 'link' predicate are treated as
      bidirectional for the purpose of calculating shortest path distances.
    - Nut locations are static (do not change during planning).
    - The goal only involves tightening a specific set of nuts.
    - A spanner becomes unusable after tightening one nut.
    - The heuristic assumes unit cost for walk, pickup, and tighten actions.

    Heuristic Initialization:
    In the constructor (__init__), the heuristic precomputes the following:
    1. Identifies all goal nuts from the task's goal conditions (`(tightened ?n)`).
    2. Builds a graph of locations based on 'link' facts in the static information.
       Links are treated as bidirectional edges with weight 1.
    3. Computes all-pairs shortest path distances between all locations using BFS.
       These distances represent the minimum number of 'walk' actions required
       to travel between any two locations.
    4. Stores the static locations of all nuts based on 'at' facts found in the
       static or initial state information.
    5. Infers object types (man, spanner, nut, location) by examining all facts
       in the initial state, static information, and goal, based on the predicates
       objects appear in. This is necessary as type information is not directly
       available in the Task object. It also identifies the single man object.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Check if the state is a goal state using `self.task.goal_reached(state)`.
       If yes, return 0.
    2. Extract state-dependent information by iterating through the facts in the state:
       - The man's current location (`man_loc`).
       - The spanner the man is carrying, if any (`man_carried_spanner`).
       - The set of spanners that are currently usable (`usable_spanners_in_state`).
       - The locations of all spanners that are currently at a location (`spanner_locations_in_state`).
       - The set of nuts that are currently loose (`loose_nuts_in_state`).
    3. Identify the set of goal nuts that are currently loose (`loose_goal_nuts`).
    4. If `loose_goal_nuts` is empty, it means all goal nuts are tightened. Since
       step 1 already checked `task.goal_reached`, reaching this point with
       an empty `loose_goal_nuts` implies the goal was reached, so return 0.
    5. Identify the locations of usable spanners that are currently at a location
       (`available_spanner_locs`).
    6. Determine if the man is currently carrying a usable spanner (`man_carries_usable`).
    7. Check if the total number of available usable spanners (carried + at locations)
       is less than the number of loose goal nuts. If so, the state is likely
       unsolvable, return `float('inf')`.
    8. Get the list of locations for the loose goal nuts (`loose_goal_nut_locs`).
    9. Initialize the heuristic value `h` to 0.
    10. Initialize simulation variables: `current_loc` (starts at `man_loc`),
        `spanners_needed` (starts at `len(loose_goal_nuts)`), `spanner_carried`
        (starts at `man_carries_usable`), and mutable copies of
        `available_spanner_locs` and `loose_goal_nut_locs` to track items used
        within the simulation (`available_spanner_locs_sim`, `loose_goal_nut_locs_sim`).
    11. Enter a loop that continues as long as `spanners_needed > 0`:
        a. If the man is not currently carrying a spanner (`not spanner_carried`):
           - Find the location of an available usable spanner in `available_spanner_locs_sim`
             that has the minimum shortest path distance from the `current_loc`.
           - If no reachable spanner is found, the state is unsolvable; return `float('inf')`.
           - Add the minimum distance to `h` (cost of walk).
           - Add 1 to `h` for the 'pickup_spanner' action.
           - Update `current_loc` to the location of the picked-up spanner.
           - Mark the chosen spanner location as "used" in `available_spanner_locs_sim`
             (by setting the entry to None).
           - Set `spanner_carried` to True.
        b. If the man is currently carrying a spanner (`spanner_carried` is True):
           - Find the location of a loose goal nut in `loose_goal_nut_locs_sim`
             that has the minimum shortest path distance from the `current_loc`.
           - If no reachable nut is found, the state is unsolvable; return `float('inf')`.
           - Add the minimum distance to `h` (cost of walk).
           - Add 1 to `h` for the 'tighten_nut' action.
           - Update `current_loc` to the location of the tightened nut.
           - Mark the chosen nut location as "used" in `loose_goal_nut_locs_sim`
             (by setting the entry to None).
           - Set `spanner_carried` to False (spanner is used up).
           - Decrement `spanners_needed` by 1.
    12. Return the final value of `h`.
    """

    def __init__(self, task):
        self.task = task
        self.goal_nuts = set()
        self.nut_locations = {}
        self.location_graph = {} # Adjacency list {loc: {neighbor1, neighbor2}}
        self.all_locations = set()
        self._object_types = {
            'man': set(),
            'spanner': set(),
            'nut': set(),
            'location': set(),
            'locatable': set()
        }

        # Parse static facts, initial state, and goals to build graph, get nut locations, and infer types
        all_relevant_facts = set(task.static) | set(task.initial_state) | set(task.goals)

        for fact_str in all_relevant_facts:
            pred, *args = self._parse_fact(fact_str)
            if pred == 'link':
                l1, l2 = args
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Assume bidirectional for distance
                self.all_locations.add(l1)
                self.all_locations.add(l2)
            elif pred == 'at':
                 obj, loc = args
                 # Infer types based on position/predicate roles
                 self._object_types['locatable'].add(obj)
                 self._object_types['location'].add(loc)
                 # Store potential nut locations (will refine type later)
                 # This assumes nut locations are in initial state or static
                 # A simple name check is a weak inference, but works for examples
                 if obj.startswith('nut'):
                      self.nut_locations[obj] = loc
            elif pred == 'carrying':
                 m, s = args
                 self._object_types['man'].add(m)
                 self._object_types['spanner'].add(s)
            elif pred == 'usable':
                 s = args[0]
                 self._object_types['spanner'].add(s)
            elif pred == 'loose' or pred == 'tightened':
                 n = args[0]
                 self._object_types['nut'].add(n)

        # Refine nut locations using inferred types (if needed, though initial guess is likely sufficient)
        # This step is redundant if nut locations are reliably in static/initial 'at' facts.
        # Keeping it simple based on initial guess is likely fine for this domain.

        # Parse goal facts to find goal nuts
        for goal_fact in task.goals:
             pred, *args = self._parse_fact(goal_fact)
             if pred == 'tightened':
                 self.goal_nuts.add(args[0])

        # Compute all-pairs shortest paths
        self.dist = self._compute_shortest_paths()

        # Identify the single man object (assuming only one based on examples)
        # Use next(iter(set), None) to get an element safely from a set
        self.man_obj = next(iter(self._object_types['man']), None)
        # If self.man_obj is None, the domain might be malformed or different from examples.
        # The heuristic will likely fail or return inf if man_loc cannot be found.


    def __call__(self, state):
        # Check if goal is reached first
        if self.task.goal_reached(state):
            return 0

        # Extract state-dependent info
        man_loc = None
        man_carried_spanner = None
        usable_spanners_in_state = set()
        spanner_locations_in_state = {}
        loose_nuts_in_state = set()

        # Assuming there is exactly one man object identified in __init__
        if self.man_obj is None:
             # Cannot find the man object, heuristic is undefined
             return float('inf')

        for fact in state:
            pred, *args = self._parse_fact(fact)
            if pred == 'at':
                obj, loc = args
                if obj == self.man_obj:
                    man_loc = loc
                elif obj in self._object_types['spanner']:
                    spanner_locations_in_state[obj] = loc
            elif pred == 'carrying':
                 m, spanner = args
                 if m == self.man_obj:
                     man_carried_spanner = spanner
            elif pred == 'usable':
                 usable_spanners_in_state.add(args[0])
            elif pred == 'loose':
                 loose_nuts_in_state.add(args[0])

        # Man's location is essential
        if man_loc is None:
             return float('inf') # Man's location is unknown

        # Filter for loose goal nuts
        loose_goal_nuts = loose_nuts_in_state.intersection(self.goal_nuts)

        # If loose_goal_nuts is empty, all goal nuts are tightened.
        # This case should be covered by the initial task.goal_reached(state) check,
        # but as a safeguard, return 0 if somehow reached here.
        if not loose_goal_nuts:
             return 0

        # Find available usable spanners at locations
        available_spanner_locs = [
            loc for spanner, loc in spanner_locations_in_state.items()
            if spanner in usable_spanners_in_state
        ]

        # Check if man carries a usable spanner
        man_carries_usable = (man_carried_spanner is not None and man_carried_spanner in usable_spanners_in_state)

        # Check if enough spanners exist in total (carried + at locations)
        total_usable_spanners = len(available_spanner_locs) + (1 if man_carries_usable else 0)
        if total_usable_spanners < len(loose_goal_nuts):
            return float('inf') # Unsolvable from this state

        # Get locations of loose goal nuts
        loose_goal_nut_locs = [self.nut_locations[nut] for nut in loose_goal_nuts]

        # --- Heuristic Calculation (Greedy Simulation) ---
        h = 0
        current_loc = man_loc
        spanners_needed = len(loose_goal_nuts)
        spanner_carried = man_carries_usable
        available_spanner_locs_sim = list(available_spanner_locs) # Copy for simulation
        loose_goal_nut_locs_sim = list(loose_goal_nut_locs) # Copy for simulation

        while spanners_needed > 0:
            if not spanner_carried:
                # Need to get a spanner
                min_dist_to_spanner = float('inf')
                best_spanner_loc_idx = -1
                for i, s_loc in enumerate(available_spanner_locs_sim):
                    if s_loc is not None: # Check if this spanner location is still available in simulation
                        d = self.dist.get((current_loc, s_loc), float('inf'))
                        if d == float('inf'): # Cannot reach this spanner location
                            continue
                        if d < min_dist_to_spanner:
                            min_dist_to_spanner = d
                            best_spanner_loc_idx = i

                # If best_spanner_loc_idx is -1, it means no reachable spanners are available in simulation.
                # This implies the state is unsolvable (e.g., spanners exist but are unreachable).
                if best_spanner_loc_idx == -1:
                     return float('inf')

                h += min_dist_to_spanner # Walk to spanner
                h += 1 # Pickup action cost
                current_loc = available_spanner_locs_sim[best_spanner_loc_idx]
                available_spanner_locs_sim[best_spanner_loc_idx] = None # Mark as used in simulation
                spanner_carried = True

            # Now spanner_carried is True
            # Need to go to a loose nut
            min_dist_to_nut = float('inf')
            best_nut_loc_idx = -1
            for i, n_loc in enumerate(loose_goal_nut_locs_sim):
                if n_loc is not None: # Check if this nut location is still untightened in simulation
                    d = self.dist.get((current_loc, n_loc), float('inf'))
                    if d == float('inf'): # Cannot reach this nut location
                        continue
                    if d < min_dist_to_nut:
                        min_dist_to_nut = d
                        best_nut_loc_idx = i

            # If best_nut_loc_idx is -1, it means no reachable nuts are left in simulation,
            # which contradicts spanners_needed > 0. This state is unsolvable.
            if best_nut_loc_idx == -1:
                 return float('inf')

            h += min_dist_to_nut # Walk to nut
            h += 1 # Tighten action cost
            current_loc = loose_goal_nut_locs_sim[best_nut_loc_idx]
            loose_goal_nut_locs_sim[best_nut_loc_idx] = None # Mark as used in simulation
            spanner_carried = False # Spanner is used up
            spanners_needed -= 1

        return h

    def _parse_fact(self, fact_str):
        # Helper to parse PDDL fact string into (predicate, arg1, arg2, ...)
        # Removes surrounding parentheses and splits by space
        return tuple(fact_str[1:-1].split())

    def _compute_shortest_paths(self):
        # Compute all-pairs shortest paths using BFS from each node
        dist = {}
        locations = list(self.all_locations)

        for start_node in locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            dist[(start_node, start_node)] = 0

            while q:
                current_node, current_dist = q.popleft()

                if current_node in self.location_graph:
                    for neighbor in self.location_graph[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            dist[(start_node, neighbor)] = current_dist + 1
                            q.append((neighbor, current_dist + 1))

        return dist

