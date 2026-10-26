from fnmatch import fnmatch
from collections import deque
import math # Import math for infinity

# Assume Heuristic base class is available as in the example
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and multiple spaces
    return fact.strip()[1:-1].split()

# Helper function to match PDDL facts with patterns
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

# Assume Heuristic base class exists and provides the structure
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#         # Assume task also has task.initial_state and task.objects_by_type

class spannerHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all goal nuts by summing:
    1. The number of untightened goal nuts (representing the minimum tighten actions).
    2. The cost to acquire the first usable spanner if the man is not already carrying one (travel to closest usable spanner + pickup action).
    3. The cost to travel from the man's current location (or the location where he acquires the first spanner) to the closest untightened nut location.

    # Assumptions
    - The location graph defined by 'link' predicates is connected for all relevant locations in solvable problems.
    - For solvable problems, there are always enough usable spanners available initially (either carried or on the ground) to tighten all goal nuts.
    - The man can carry at most one spanner at a time (based on domain definition).
    - Action costs are uniform (cost 1).

    # Heuristic Initialization
    - Identify the man object name by inspecting initial state facts.
    - Store the set of goal nuts from the task goals.
    - Build the location graph from 'link' static facts and locations mentioned in the initial state/goals.
    - Precompute shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of untightened goal nuts in the current state and count them (`num_untightened`).
    2. If `num_untightened` is 0, the state is a goal state, return 0.
    3. Initialize the heuristic value `h` with `num_untightened` (minimum tighten actions required).
    4. Find the man's current location (`man_location`).
    5. Check if the man is currently carrying a usable spanner (`man_has_usable_spanner`).
    6. Find all usable spanners currently on the ground and their locations (`usable_spanners_on_ground`).
    7. Calculate the cost and determine the man's effective location after addressing the immediate need (acquiring a spanner if necessary):
       - Initialize `spanner_acquisition_cost = 0`.
       - Initialize `effective_man_location = man_location`.
       - If the man is NOT carrying a usable spanner:
         - Find the closest usable spanner on the ground relative to `man_location`.
         - If a usable spanner on the ground is found:
           - `spanner_acquisition_cost` is the travel distance to this spanner's location plus 1 (for the pickup action).
           - `effective_man_location` becomes the location of this closest spanner.
         - If NO usable spanners are found on the ground (and man needs one):
           - This indicates a likely unsolvable state or a state very far from a solution. Add a large penalty (e.g., 100) to `spanner_acquisition_cost`. `effective_man_location` remains `man_location`.
    8. Add `spanner_acquisition_cost` to `h`.
    9. From the `effective_man_location`, find the closest untightened nut location. Add the travel distance to this closest nut location to `h`.
    10. Return the final heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and building the location graph."""
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        # Identify the man object name.
        # Try finding in 'carrying' facts first.
        self.man = None
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'carrying' and len(parts) == 3:
                self.man = parts[1]
                break

        # If not found in 'carrying', try finding in 'at' facts, excluding likely spanners/nuts by name convention.
        if self.man is None:
            spanner_like_names = set()
            nut_like_names = set()
            for fact in self.initial_state:
                parts = get_parts(fact)
                if parts[0] == 'usable' and len(parts) == 2:
                    spanner_like_names.add(parts[1])
                elif parts[0] == 'loose' and len(parts) == 2:
                    nut_like_names.add(parts[1])

            for fact in self.initial_state:
                parts = get_parts(fact)
                if parts[0] == 'at' and len(parts) == 3:
                    obj_name = parts[1]
                    if obj_name not in spanner_like_names and obj_name not in nut_like_names:
                         self.man = obj_name
                         break # Assume the first one found is the man

        # Store goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened" and len(args) > 0:
                self.goal_nuts.add(args[0])

        # Build location graph and compute distances
        self.location_graph = {}
        all_locations = set()

        # Add locations from link facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Links are bidirectional
                all_locations.add(l1)
                all_locations.add(l2)

        # Add locations from initial state 'at' facts
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3: # (at obj loc)
                 all_locations.add(parts[2])

        # Ensure all identified locations are nodes in the graph, even if isolated initially
        for loc in all_locations:
             self.location_graph.setdefault(loc, set())

        self.distances = {}
        for start_loc in self.location_graph:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """Perform BFS to find shortest distances from start_node to all reachable nodes."""
        distances = {node: math.inf for node in self.location_graph}
        if start_node in distances: # Ensure start_node is in the graph
            distances[start_node] = 0
            queue = deque([(start_node, 0)])

            while queue:
                (current_node, dist) = queue.popleft()

                # Ensure current_node is a valid key in the graph
                if current_node not in self.location_graph:
                     continue

                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state facts (frozenset of strings)

        # 1. Identify untightened goal nuts and their current locations
        untightened_goal_nuts = {} # {nut: location}
        for nut in self.goal_nuts:
            if '(tightened ' + nut + ')' not in state:
                # Find the current location of the nut
                nut_location = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'at' and len(parts) == 3 and parts[1] == nut:
                        nut_location = parts[2]
                        break
                if nut_location:
                    untightened_goal_nuts[nut] = nut_location
                # else: nut is not 'at' any location? Assume nuts are always at a location.

        num_untightened = len(untightened_goal_nuts)

        # 2. If num_untightened is 0, the state is a goal state, return 0.
        if num_untightened == 0:
            return 0

        # 3. Initialize the heuristic value
        h = num_untightened # Minimum tighten actions

        # 4. Find the man's current location
        man_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1] == self.man:
                man_location = parts[2]
                break
        # If man_location is None, something is wrong with the state representation or domain.
        # Assume man is always at a location present in the graph.
        if man_location is None or man_location not in self.distances:
             # This state is likely unreachable or invalid
             return math.inf # Or a large penalty

        # 5. Check if man is carrying a usable spanner.
        carrying_spanner = None
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'carrying' and len(parts) == 3 and parts[1] == self.man:
                 spanner = parts[2]
                 if '(usable ' + spanner + ')' in state:
                     carrying_spanner = spanner
                 break # Assuming man carries at most one spanner

        man_has_usable_spanner = carrying_spanner is not None

        # 6. Find usable spanners currently on the ground.
        usable_spanners_on_ground = {} # {spanner: location}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('spanner'): # Heuristic specific assumption
                spanner = parts[1]
                location = parts[2]
                if '(usable ' + spanner + ')' in state:
                     usable_spanners_on_ground[spanner] = location

        # 7. Calculate the cost and determine the man's effective location after addressing the immediate need (acquiring a spanner if necessary):
        spanner_acquisition_cost = 0
        effective_man_location = man_location

        if not man_has_usable_spanner:
            # Man needs to acquire a spanner. Target is the closest usable spanner location.
            min_acquire_cost = math.inf
            best_spanner_loc = None

            for spanner_loc in usable_spanners_on_ground.values():
                if spanner_loc in self.distances[man_location]:
                    # Cost is travel to spanner + pickup action (cost 1)
                    acquire_cost = self.distances[man_location][spanner_loc] + 1
                    if acquire_cost < min_acquire_cost:
                        min_acquire_cost = acquire_cost
                        best_spanner_loc = spanner_loc

            if best_spanner_loc is not None:
                spanner_acquisition_cost = min_acquire_cost
                effective_man_location = best_spanner_loc # Man is now at spanner location
            elif not usable_spanners_on_ground:
                 # No usable spanners on ground and man isn't carrying one.
                 # Add a large penalty as likely unsolvable or very far.
                 spanner_acquisition_cost = 100 # Penalty
                 effective_man_location = man_location # Man didn't move

            # else: best_spanner_loc is None but usable_spanners_on_ground is not empty?
            # This implies usable spanners are in locations unreachable from man_location.
            # The min_acquire_cost would remain inf, and the penalty would not be added
            # by the `elif not usable_spanners_on_ground` check.
            # Let's add the penalty if min_acquire_cost is still inf.
            if spanner_acquisition_cost == math.inf:
                 spanner_acquisition_cost = 100 # Penalty

        # Add the immediate cost (spanner acquisition + travel to spanner) to h
        h += spanner_acquisition_cost

        # 8. From the man's effective location, find the closest untightened nut location.
        min_dist_to_nut = math.inf
        closest_nut_loc = None

        if effective_man_location in self.distances:
            for nut_loc in untightened_goal_nuts.values():
                if nut_loc in self.distances[effective_man_location]:
                    dist = self.distances[effective_man_location][nut_loc]
                    if dist < min_dist_to_nut:
                        min_dist_to_nut = dist
                        closest_nut_loc = nut_loc

        if closest_nut_loc is not None:
            h += min_dist_to_nut # Travel to closest nut
        # else: untightened nuts exist, but no path from effective_man_location to any nut location.
        # This implies disconnected graph or problematic state. Assume connected graph for solvable problems.
        # If min_dist_to_nut remains inf, it means no path exists. The heuristic will be lower than it should be,
        # but the state is likely bad anyway. We don't add anything if min_dist_to_nut is inf.

        # 9. Return the final heuristic value h.
        return h

