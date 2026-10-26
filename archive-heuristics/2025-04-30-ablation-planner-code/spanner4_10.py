# Import necessary modules
from fnmatch import fnmatch
from collections import deque
import math # For infinity

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for BFS
def bfs(start_loc, location_links):
    """
    Performs Breadth-First Search to find shortest distances from start_loc
    to all reachable locations.

    Args:
        start_loc: The starting location.
        location_links: Adjacency dictionary mapping location to list of linked locations.

    Returns:
        A dictionary mapping reachable location to its distance from start_loc.
    """
    distances = {start_loc: 0}
    queue = deque([start_loc])
    visited = {start_loc}

    while queue:
        curr = queue.popleft()

        if curr in location_links: # Handle locations with no links
            for neighbor in location_links[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[curr] + 1
                    queue.append(neighbor)
    return distances

class spannerHeuristic: # Inherit from Heuristic in actual use, e.g., class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It calculates the minimum cost for each (nut, usable spanner) pair, assuming
    the man starts from his current location for each task, and then greedily
    assigns the cheapest spanners to nuts until all goal nuts are covered.

    # Assumptions:
    - There is only one man.
    - The man can carry multiple spanners (based on example state, contradicting
      action definition structure).
    - Spanners become unusable after one use for tightening a nut.
    - Nuts do not move.
    - Locations form a graph connected by links.

    # Heuristic Initialization
    - Builds the graph of locations based on `link` facts.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Identifies the set of nuts that need to be tightened based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify State Information**: Determine the man's current location,
        which nuts are currently loose, their locations, which spanners are
        usable and on the ground (and their locations), and which usable
        spanners the man is carrying.
    2.  **Identify Nuts to Tighten**: Filter the loose nuts to find those
        that are required to be tightened in the goal state. Let N be this count.
        If N is 0, the heuristic is 0.
    3.  **Identify Available Spanners**: Collect all usable spanners, including
        those the man is carrying and those on the ground. Let M be this count.
        If N > M, the goal is unreachable with available spanners, return infinity.
    4.  **Calculate Task Costs**: For each nut `n` that needs tightening (at location `l_n`)
        and each available usable spanner `s` (at location `l_s`, which is the man's
        location if carried, or its ground location), calculate the estimated cost
        to use spanner `s` to tighten nut `n`. This cost is estimated as:
        `dist(man_loc, l_s) + 1 (pickup, if on ground) + dist(l_s, l_n) + 1 (tighten)`.
        If `s` is carried, the `dist(man_loc, l_s)` and pickup cost are 0.
        Use precomputed shortest path distances. If any required travel is impossible,
        the cost for this pair is infinity.
    5.  **Create and Sort Tasks**: Store these (nut, spanner, cost) combinations
        as a list of tasks. Sort the list in ascending order of cost.
    6.  **Greedy Assignment**: Iterate through the sorted tasks. For each task
        `(n, s, cost)`, if nut `n` has not yet been assigned a spanner and spanner `s`
        has not yet been used, select this task. Add its cost to the total heuristic,
        mark `n` as assigned and `s` as used. Repeat until N nuts are assigned.
    7.  **Return Heuristic Value**: If N nuts were successfully assigned, return the
        total accumulated cost. If the greedy process finishes but fewer than N nuts
        were assigned (meaning remaining nuts are unreachable with remaining spanners),
        return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts,
        and precomputing shortest path distances.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                self.goal_nuts.add(args[0])

        # Build location graph from link facts
        self.location_links = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                self.location_links.setdefault(loc1, []).append(loc2)
                self.location_links.setdefault(loc2, []).append(loc1)
                locations.add(loc1)
                locations.add(loc2)

        self.locations = list(locations)

        # Precompute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = bfs(start_loc, self.location_links)

    def get_distance(self, loc1, loc2):
        """Looks up precomputed distance, returns infinity if unreachable."""
        if loc1 == loc2:
            return 0
        # Ensure loc1 is a valid start location in our precomputed distances
        if loc1 not in self.distances:
             # This location wasn't part of the linked graph, likely unreachable
             return float('inf')
        if loc2 in self.distances[loc1]:
            return self.distances[loc1][loc2]
        return float('inf')

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify State Information
        man_obj = None
        man_loc = None
        carried_spanners = [] # List of spanner objects carried by the man
        usable_spanners_on_ground_map = {} # spanner -> location
        nut_locations = {} # nut -> location
        loose_nuts = set()
        usable_spanners_set = set() # Set of all usable spanner objects in state

        # Pass 1: Identify man object and carried spanners
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                man_obj = get_parts(fact)[1]
                carried_spanners.append(get_parts(fact)[2])

        # Pass 2: Identify usable spanners, loose nuts, and locations of objects
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "usable":
                usable_spanners_set.add(parts[1])
            elif predicate == "loose":
                loose_nuts.add(parts[1])
            elif predicate == "at":
                obj, loc = parts[1:]
                # We need to determine if obj is man, nut, or spanner
                # Rely on predicates like 'carrying', 'loose', 'tightened', 'usable'
                is_nut = obj in loose_nuts or obj in self.goal_nuts # Check goal nuts too
                is_spanner = obj in usable_spanners_set or obj in carried_spanners

                if obj == man_obj:
                    man_loc = loc
                elif is_nut:
                     nut_locations[obj] = loc
                elif is_spanner:
                     # This spanner is on the ground at 'loc' if not carried
                     if obj not in carried_spanners:
                         usable_spanners_on_ground_map[obj] = loc

        # If man_obj wasn't found via 'carrying', try finding it among objects at locations
        # that are not known nuts or spanners. This is a fallback and might be fragile.
        if man_obj is None:
             known_nuts = set(loose_nuts).union(self.goal_nuts)
             known_spanners = set(usable_spanners_set).union(carried_spanners)
             for fact in state:
                 if match(fact, "at", "*", "*"):
                     obj, loc = get_parts(fact)[1:]
                     if obj not in known_nuts and obj not in known_spanners:
                          man_obj = obj
                          man_loc = loc # Assume this is the man's location
                          break # Found the man

        # If still no man_obj, cannot compute heuristic
        if man_obj is None or man_loc is None:
             # print("Warning: Could not identify the man object or location.")
             return float('inf') # Cannot proceed

        # 2. Identify Nuts to Tighten
        nuts_to_tighten = [nut for nut in loose_nuts if nut in self.goal_nuts]
        N = len(nuts_to_tighten)

        if N == 0:
            return 0 # Goal achieved

        # 3. Identify Available Usable Spanners
        available_usable_spanners = []
        # Add carried usable spanners
        for spanner in carried_spanners:
            if spanner in usable_spanners_set:
                available_usable_spanners.append(spanner)
        # Add usable spanners on the ground
        for spanner, loc in usable_spanners_on_ground_map.items():
             if spanner in usable_spanners_set:
                  available_usable_spanners.append(spanner)

        M = len(available_usable_spanners)

        if N > M:
            # Not enough usable spanners to tighten all required nuts
            return float('inf')

        # 4. Calculate Task Costs & 5. Create and Sort Tasks
        task_costs = []
        for nut in nuts_to_tighten:
            l_n = nut_locations.get(nut)
            if l_n is None:
                 # Nut location not found in state - should not happen in valid states
                 # print(f"Warning: Location for nut {nut} not found in state.")
                 continue # Skip this nut, effectively making goal unreachable if required

            for spanner in available_usable_spanners:
                if spanner in carried_spanners:
                    # Spanner is carried by the man
                    l_s = man_loc
                    # Cost: Walk from man_loc to nut_loc + tighten
                    walk_cost = self.get_distance(man_loc, l_n)
                    if walk_cost == float('inf'):
                         cost = float('inf')
                    else:
                         cost = walk_cost + 1 # +1 for tighten
                else:
                    # Spanner is on the ground
                    l_s = usable_spanners_on_ground_map.get(spanner)
                    if l_s is None:
                         # Spanner location not found - should not happen
                         # print(f"Warning: Location for spanner {spanner} not found on ground.")
                         continue # Skip this spanner

                    # Cost: Walk man_loc to spanner_loc + pickup + walk spanner_loc to nut_loc + tighten
                    walk1_cost = self.get_distance(man_loc, l_s)
                    walk2_cost = self.get_distance(l_s, l_n)

                    if walk1_cost == float('inf') or walk2_cost == float('inf'):
                         cost = float('inf')
                    else:
                         cost = walk1_cost + 1 + walk2_cost + 1 # +1 for pickup, +1 for tighten

                # Only add tasks with finite cost
                if cost != float('inf'):
                    task_costs.append((nut, spanner, cost))

        # If no finite cost tasks can be formed for any nut, goal is unreachable
        if not task_costs and N > 0:
             return float('inf')

        task_costs.sort(key=lambda x: x[2]) # Sort by cost

        # 6. Greedy Assignment
        total_heuristic_cost = 0
        assigned_nuts = set()
        used_spanners = set()
        nuts_assigned_count = 0

        for nut, spanner, cost in task_costs:
            if nut not in assigned_nuts and spanner not in used_spanners:
                total_heuristic_cost += cost
                assigned_nuts.add(nut)
                used_spanners.add(spanner)
                nuts_assigned_count += 1
                if nuts_assigned_count == N:
                    break # All required nuts have been assigned

        # 7. Return Heuristic Value
        if nuts_assigned_count == N:
            return total_heuristic_cost
        else:
            # Could not find a valid assignment for all N nuts
            return float('inf')
