from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic
import collections # For BFS queue

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose goal nuts. It considers the costs of walking, picking up usable
    spanners, and tightening nuts. It uses a greedy simulation approach
    to estimate travel and spanner acquisition costs: the man sequentially
    acquires a needed spanner (if not already carrying one) by going to the
    nearest available usable spanner on the ground, and then travels to the
    nearest loose goal nut to tighten it, repeating until all goal nuts are
    tightened.

    # Assumptions
    - The man can carry multiple spanners simultaneously (as implied by PDDL).
    - Each tighten_nut action consumes one usable spanner.
    - Nuts do not move.
    - Spanners do not move unless carried by the man.
    - The graph of locations connected by 'link' predicates is undirected.
    - The problem is solvable (i.e., there are enough usable spanners).
    - There is exactly one man object.

    # Heuristic Initialization
    - Identify the man object by looking for objects involved in 'carrying' or 'at' predicates.
    - Identify all location objects from 'at' and 'link' predicates.
    - Identify all spanner objects from 'carrying', 'usable', and 'at' predicates.
    - Identify all nut objects from 'loose', 'tightened', and 'at' predicates.
    - Build the graph of locations based on 'link' facts.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify the set of goal nuts from the goal condition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract relevant information from the current state: man's location,
       locations of all objects, usable spanners, carried spanners, and loose nuts.
    2. Identify the set of loose goal nuts. If this set is empty, the heuristic is 0.
    3. Identify the set of usable spanners (both carried and on the ground) present in the current state.
    4. Check if the number of loose goal nuts exceeds the total number of usable
       spanners in the current state. If so, the problem is likely unsolvable, return infinity.
    5. Initialize the total estimated cost to 0.
    6. Initialize the man's current location to his actual location in the state.
    7. Initialize the set of usable spanners currently carried by the man (that are also usable in the state).
    8. Initialize the set of usable spanners available on the ground (that are usable and at a location in the state).
    9. Create a list of remaining loose goal nuts to process.
    10. Enter a loop that continues as long as there are remaining loose goal nuts:
        a. If the man is not currently carrying any usable spanners (set is empty):
           i. Find the nearest usable spanner available on the ground whose location is known in the current state.
           ii. If no such spanner exists (should not happen if step 4 passed and graph is connected), return infinity.
           iii. Add the distance to this spanner's location plus 1 (for the pickup action) to the total cost.
           iv. Update the man's current location to the spanner's location.
           v. Add the picked-up spanner to the set of conceptually carried usable spanners.
           vi. Remove the picked-up spanner from the set of available ground spanners.
        b. The man is now conceptually carrying at least one usable spanner. Find the nearest remaining loose goal nut whose location is known in the current state.
        c. If no such nut exists (should not happen if the loop condition is correct), break the loop.
        d. Add the distance to this nut's location plus 1 (for the tighten action) to the total cost.
        e. Update the man's current location to the nut's location.
        f. Remove the tightened nut from the list of remaining loose goal nuts.
        g. Remove an arbitrary spanner from the set of conceptually carried usable spanners (as one is consumed).
    11. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        # Identify object types and names from initial state and goals
        self.locations = set()
        self.spanners = set()
        self.nuts = set()
        self.man_name = None

        all_relevant_facts = set(task.initial_state) | set(task.static) | set(task.goals)

        # Infer types based on predicates found in initial/static/goal state
        potential_men = set()
        potential_spanners = set()
        potential_nuts = set()
        potential_locations = set()

        for fact_str in all_relevant_facts:
            parts = get_parts(fact_str)
            if not parts: continue # Skip empty facts if any
            pred = parts[0]
            if pred == 'at' and len(parts) == 3:
                obj, loc = parts[1:]
                potential_locations.add(loc)
            elif pred == 'carrying' and len(parts) == 3:
                man, spanner = parts[1:]
                potential_men.add(man) # First arg of carrying is a man
                potential_spanners.add(spanner) # Second arg of carrying is a spanner
            elif pred in ['tightened', 'loose'] and len(parts) == 2:
                nut = parts[1]
                potential_nuts.add(nut) # Arg of tightened/loose is a nut
            elif pred == 'usable' and len(parts) == 2:
                spanner = parts[1]
                potential_spanners.add(spanner) # Arg of usable is a spanner
            elif pred == 'link' and len(parts) == 3:
                loc1, loc2 = parts[1:]
                potential_locations.add(loc1)
                potential_locations.add(loc2) # Args of link are locations

        # Prioritize finding the man from 'carrying' predicates
        if potential_men:
            # Assuming only one man, pick the first one found
            self.man_name = list(potential_men)[0]
        else:
             # Fallback: Look for the first object in any 'at' fact in initial state.
             # This is less reliable but necessary if 'carrying' isn't in initial/static/goal.
             for fact in task.initial_state:
                 if match(fact, "at", "*", "*"):
                     self.man_name = get_parts(fact)[1]
                     break
             if self.man_name is None:
                 # Should not happen in valid spanner problems
                 print("Warning: Could not identify man object.")
                 # Heuristic might return inf later if man_location cannot be found.


        self.locations = potential_locations
        self.spanners = potential_spanners
        self.nuts = potential_nuts

        # Identify goal nuts
        self.goal_nuts = set()
        for goal_fact in self.goals:
            if match(goal_fact, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal_fact)[1])

        # Build location graph and compute distances
        self.location_graph = {}
        for loc in self.locations:
            self.location_graph[loc] = set()

        for fact in self.static:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                # Ensure locations are in our identified set before adding link
                if loc1 in self.locations and loc2 in self.locations:
                     self.location_graph[loc1].add(loc2)
                     self.location_graph[loc2].add(loc1) # Assuming links are bidirectional

        self.dist_map = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS from each location.
        """
        distances = {}
        for start_node in self.locations: # Iterate over identified locations
            distances[start_node] = {}
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                (current_node, current_dist) = queue.popleft()
                distances[start_node][current_node] = current_dist

                # Get neighbors from the built graph, handle nodes with no links
                for neighbor in self.location_graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, current_dist + 1))

        # For any pair of locations (l1, l2) where l2 was not reached from l1,
        # the distance remains undefined in the inner dict.
        # The self.dist method handles this by returning float('inf').
        return distances

    def dist(self, loc1, loc2):
        """Helper to get precomputed distance."""
        # Check if both locations are known and if a path exists
        return self.dist_map.get(loc1, {}).get(loc2, float('inf'))


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        using a greedy simulation.
        """
        state = node.state  # Current world state (frozenset of facts)

        # 1. Extract relevant information from the current state
        obj_locations = {} # Map object name to its location string
        usable_spanners_in_state = set() # Set of spanner names that are usable
        carried_spanners_in_state = set() # Set of spanner names carried by the man
        loose_nuts_in_state = set() # Set of nut names that are loose
        man_location = None # Man's current location string

        # Use the pre-identified man_name
        if self.man_name is None:
             # Cannot compute heuristic if man is not identified
             return float('inf')

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at' and len(parts) == 3:
                obj, loc = parts[1:]
                obj_locations[obj] = loc
                if obj == self.man_name:
                    man_location = loc
            elif pred == 'usable' and len(parts) == 2:
                spanner = parts[1]
                usable_spanners_in_state.add(spanner)
            elif pred == 'carrying' and len(parts) == 3:
                man, spanner = parts[1:]
                if man == self.man_name:
                    carried_spanners_in_state.add(spanner)
            elif pred == 'loose' and len(parts) == 2:
                nut = parts[1]
                loose_nuts_in_state.add(nut)

        # Ensure man_location is found (should always be the case in a valid state)
        if man_location is None:
             # This indicates an invalid state representation or parsing issue
             return float('inf') # Cannot proceed without man's location

        # 2. Identify the set of loose goal nuts.
        # Filter the set of all goal nuts by those that are currently loose in the state.
        remaining_loose_goal_nuts = {n for n in self.goal_nuts if n in loose_nuts_in_state}

        # If all goal nuts are already tightened, heuristic is 0.
        if not remaining_loose_goal_nuts:
            return 0

        # 3. Identify the set of usable spanners (both carried and on the ground) present in the current state.
        all_usable_spanners_in_state = usable_spanners_in_state

        # 4. Check if the number of loose goal nuts exceeds the total number of usable spanners.
        if len(remaining_loose_goal_nuts) > len(all_usable_spanners_in_state):
            return float('inf') # Problem likely unsolvable

        # 5. Initialize the total estimated cost to 0.
        total_cost = 0

        # 6. Initialize the man's current location.
        current_location = man_location

        # 7. Initialize the set of usable spanners currently carried by the man.
        # These are spanners that are both carried AND usable in the current state.
        current_usable_carried = set(carried_spanners_in_state) & usable_spanners_in_state

        # 8. Initialize the set of usable spanners available on the ground.
        # These are spanners that are usable, not carried, and currently at a known location.
        available_ground_usable = {s for s in usable_spanners_in_state if s not in carried_spanners_in_state and s in obj_locations}

        # 9. Create a list of remaining loose goal nuts to process.
        # Use a list copy so we can remove elements during simulation.
        remaining_nuts_list = list(remaining_loose_goal_nuts)

        # 10. Enter a loop that continues as long as there are remaining loose goal nuts:
        while remaining_nuts_list:
            # a. If the man is not currently carrying any usable spanners:
            if not current_usable_carried:
                # i. Find the nearest usable spanner available on the ground.
                nearest_spanner = None
                min_dist = float('inf')

                # Iterate through available ground spanners to find the closest one
                # Ensure the spanner's location is known in the current state
                available_spanners_at_loc = [s for s in available_ground_usable if s in obj_locations]

                for s in available_spanners_at_loc:
                    d = self.dist(current_location, obj_locations[s])
                    if d < min_dist:
                        min_dist = d
                        nearest_spanner = s

                # ii. If no such spanner exists (either none on ground or none reachable)
                if nearest_spanner is None:
                    # This implies we need a spanner but cannot get one.
                    # Given the initial check (len(LGN) <= len(US)), this suggests
                    # either a disconnected graph preventing reachability, or an issue
                    # with state parsing/representation where spanners are usable but
                    # not at any location. For a heuristic, returning inf is safe.
                    return float('inf')

                # iii. Add the distance to this spanner's location plus 1 (for the pickup action) to the total cost.
                total_cost += min_dist + 1 # Walk to spanner + Pickup action

                # iv. Update the man's current location to the spanner's location.
                current_location = obj_locations[nearest_spanner]

                # v. Add the picked-up spanner to the set of conceptually carried usable spanners.
                current_usable_carried.add(nearest_spanner)

                # vi. Remove the picked-up spanner from the set of available ground spanners.
                available_ground_usable.remove(nearest_spanner)


            # b. The man is now conceptually carrying at least one usable spanner. Find the nearest remaining loose goal nut.
            nearest_nut = None
            min_dist = float('inf')

            # Iterate through remaining loose nuts to find the closest one
            # Ensure the nut's location is known (should always be true)
            remaining_nuts_at_loc = [n for n in remaining_nuts_list if n in obj_locations]

            for n in remaining_nuts_at_loc:
                d = self.dist(current_location, obj_locations[n])
                if d < min_dist:
                    min_dist = d
                    nearest_nut = n

            # c. If no such nut exists (should not happen if remaining_nuts_list is not empty)
            if nearest_nut is None:
                 break # Exit loop, all nuts processed

            # d. Add the distance to this nut's location plus 1 (for the tighten action) to the total cost.
            total_cost += min_dist + 1 # Walk to nut + Tighten action

            # e. Update the man's current location to the nut's location.
            current_location = obj_locations[nearest_nut]

            # f. Remove the tightened nut from the list of remaining loose goal nuts.
            remaining_nuts_list.remove(nearest_nut)

            # g. Remove an arbitrary spanner from the set of conceptually carried usable spanners (as one is consumed).
            # Since we only add specific spanners during pickup simulation, we remove one here.
            # The specific spanner doesn't matter for the heuristic count, just that one is used.
            if current_usable_carried:
                 # Use pop() to remove an arbitrary element from the set
                 current_usable_carried.pop()
            # else: This case should not be reached if logic is correct, as we only enter this block
            # when current_usable_carried is not empty (or becomes not empty after pickup).


        # 11. Return the total estimated cost.
        return total_cost
