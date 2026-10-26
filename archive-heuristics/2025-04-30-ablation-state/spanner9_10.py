from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_location_graph(static_facts):
    """Builds a graph of locations based on 'link' predicates."""
    graph = {}
    for fact in static_facts:
        if match(fact, "link", "*", "*"):
            parts = get_parts(fact)
            loc1 = parts[1]
            loc2 = parts[2]
            graph.setdefault(loc1, set()).add(loc2)
            graph.setdefault(loc2, set()).add(loc1)
    return graph

def compute_distances(graph, locations):
    """Computes all-pairs shortest path distances using BFS."""
    distances = {}
    for start_loc in locations:
        distances[(start_loc, start_loc)] = 0
        queue = deque([(start_loc, 0)])
        visited = {start_loc}
        while queue:
            current_loc, dist = queue.popleft()
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[(start_loc, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose goal nuts. It calculates the cost for each loose goal nut by
    considering the man's current location and whether he is carrying a
    usable spanner. If he needs a spanner, it finds the closest available
    usable spanner. It greedily processes the tasks (tightening nuts),
    prioritizing getting to the closest nut if already carrying a spanner,
    or getting the closest spanner and then going to the closest nut from there
    if not carrying a spanner.

    # Assumptions
    - The goal is always to tighten a specific set of nuts.
    - Each usable spanner can tighten exactly one nut.
    - The man can carry at most one spanner at a time.
    - The man is the only agent performing actions.
    - The graph of locations defined by 'link' predicates is connected for all relevant locations (man start, spanner locations, nut locations).
    - There are enough usable spanners initially for all goal nuts in solvable problems.

    # Heuristic Initialization
    - Extracts the set of goal nuts from the task goals.
    - Identifies the man, spanners, nuts, and locations from task facts.
    - Builds a graph of locations based on 'link' predicates.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man, his current location, and if he is carrying a usable spanner from the current state.
    2. Identify all nuts that are loose and are part of the goal from the current state.
    3. Identify all usable spanners that are currently at a location (not carried by the man) from the current state.
    4. If there are no loose goal nuts, the heuristic is 0.
    5. Initialize the total heuristic cost to 0.
    6. Initialize the man's current location for the heuristic calculation to his actual current location.
    7. Initialize a flag indicating if the man is carrying a spanner for the heuristic calculation based on the actual state.
    8. Initialize a list of remaining loose goal nuts (as (nut_name, location) tuples) and a list of available usable spanners (as (spanner_name, location) tuples, those not carried by the man).
    9. While there are remaining loose goal nuts:
       a. If the man is currently considered to be carrying a spanner:
          i. Find the loose goal nut closest to the man's current location using precomputed distances.
          ii. Calculate the cost for this step: distance to reach this nut's location + 1 for the tighten action.
          iii. Add this cost to the total heuristic.
          iv. Update the man's current location for the heuristic calculation to the nut's location.
          v. Set the man's carrying spanner flag to False (the spanner is used).
          vi. Remove the chosen nut from the list of remaining loose goal nuts.
       b. If the man is not currently considered to be carrying a spanner:
          i. Find the pair of (available usable spanner, remaining loose goal nut) that minimizes the cost: distance from man's current location to spanner's location + 1 (pickup) + distance from spanner's location to nut's location + 1 (tighten). Iterate through all available spanners and all remaining nuts to find the minimum cost pair.
          ii. Let the chosen spanner be S* at L_S* and the chosen nut be N* at L_N*.
          iii. Calculate the minimum cost for this step: Distance(current_man_loc, L_S*) + 1 + Distance(L_S*, L_N*) + 1.
          iv. Add this cost to the total heuristic.
          v. Update the man's current location for the heuristic calculation to L_N*.
          vi. Remove S* from the list of available usable spanners.
          vii. Remove N* from the list of remaining loose goal nuts.
          viii. Set the man's carrying spanner flag to False (spanner used).
    10. Return the total heuristic cost. If at any point a required location is unreachable, return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Precomputes location distances.
        """
        self.goals = task.goals
        static_facts = task.static
        all_facts = task.facts # Contains type information from problem file

        # Identify object names by type from all possible facts
        self.men = {get_parts(fact)[1] for fact in all_facts if match(fact, "man", "*")}
        self.spanners = {get_parts(fact)[1] for fact in all_facts if match(fact, "spanner", "*")}
        self.nuts = {get_parts(fact)[1] for fact in all_facts if match(fact, "nut", "*")}
        self.locations = {get_parts(fact)[1] for fact in all_facts if match(fact, "location", "*")}

        # Assume there is exactly one man and store his name
        self.man_name = None
        if len(self.men) == 1:
             self.man_name = next(iter(self.men))
        elif len(self.men) > 1:
             # print(f"Warning: Found {len(self.men)} men. Using the first one found: {next(iter(self.men))}")
             self.man_name = next(iter(self.men))
        # If len(self.men) == 0, self.man_name remains None, handled in __call__

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened" and args and args[0] in self.nuts:
                self.goal_nuts.add(args[0])

        # Build location graph and compute distances
        self.location_graph = build_location_graph(static_facts)
        # Compute distances for all locations identified in the problem file
        self.distances = compute_distances(self.location_graph, list(self.locations))

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returns infinity if no path."""
        if loc1 not in self.locations or loc2 not in self.locations:
             # This indicates an issue if a relevant location wasn't parsed
             # print(f"Warning: Distance requested for unknown location(s): {loc1}, {loc2}")
             return float('inf')
        return self.distances.get((loc1, loc2), float('inf'))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        if self.man_name is None:
             # Should not happen if problem is valid and has a man
             # print("Error: Man name not identified during initialization.")
             return float('inf')

        # 1. Identify man's current location and if carrying spanner
        man_location = None
        man_carrying_spanner = None # Store the spanner object name if carrying

        current_locations = {} # Map object to its location
        for fact in state:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 current_locations[parts[1]] = parts[2]
                 if parts[1] == self.man_name:
                     man_location = parts[2]
             elif match(fact, "carrying", self.man_name, "*"):
                 man_carrying_spanner = get_parts(fact)[2]

        if man_location is None:
             # Man must always be at a location in a valid state
             # print(f"Error: Man {self.man_name} has no location in state.")
             return float('inf')

        man_carrying_usable_spanner = False
        if man_carrying_spanner and f"(usable {man_carrying_spanner})" in state:
             man_carrying_usable_spanner = True

        # 2. Identify loose goal nuts and their locations
        loose_goal_nuts_info = [] # List of (nut_name, location)
        for nut_name in self.goal_nuts:
             if f"(loose {nut_name})" in state:
                 if nut_name in current_locations:
                     loose_goal_nuts_info.append((nut_name, current_locations[nut_name]))
                 else:
                     # A goal nut is loose but has no location? Problematic state.
                     # print(f"Error: Loose goal nut {nut_name} has no location in state.")
                     return float('inf')

        # If no loose goal nuts, we are done
        if not loose_goal_nuts_info:
            return 0

        # 3. Identify available usable spanners and their locations
        # Available means usable and currently at a location (not carried by the man)
        available_usable_spanners_info = [] # List of (spanner_name, location)
        for spanner_name in self.spanners:
             if f"(usable {spanner_name})" in state:
                 # Check if it's at a location and not carried by the man
                 if spanner_name in current_locations and spanner_name != man_carrying_spanner:
                      available_usable_spanners_info.append((spanner_name, current_locations[spanner_name]))

        # Check if enough spanners exist for the remaining nuts
        if len(available_usable_spanners_info) + (1 if man_carrying_usable_spanner else 0) < len(loose_goal_nuts_info):
             # Not enough spanners available to tighten all remaining goal nuts.
             # Problem is likely unsolvable from this state.
             # print("Warning: Not enough usable spanners for all loose goal nuts.")
             return float('inf')

        # 5-10. Calculate heuristic cost iteratively
        h = 0
        current_man_loc = man_location
        man_has_spanner = man_carrying_usable_spanner
        loose_nuts_tasks = list(loose_goal_nuts_info) # Copy the list
        available_spanners = list(available_usable_spanners_info) # Copy the list

        while loose_nuts_tasks:
            if man_has_spanner:
                # Man has a spanner, go to the closest remaining nut
                min_cost = float('inf')
                best_nut_task_index = -1

                for i, (nut_name, nut_loc) in enumerate(loose_nuts_tasks):
                    dist = self.get_distance(current_man_loc, nut_loc)
                    if dist == float('inf'): continue # Cannot reach this nut

                    cost = dist + 1 # walk + tighten
                    if cost < min_cost:
                        min_cost = cost
                        best_nut_task_index = i

                if best_nut_task_index == -1 or min_cost == float('inf'):
                     # Cannot reach any remaining nut from current location. Should not happen in solvable problems.
                     # print("Error: Cannot reach any remaining nut from current location.")
                     return float('inf')

                # Apply the best task
                h += min_cost
                chosen_nut_name, chosen_nut_loc = loose_nuts_tasks.pop(best_nut_task_index)
                current_man_loc = chosen_nut_loc
                man_has_spanner = False # Spanner used

            else: # Man does not have a spanner, need to get one and go to a nut
                # Find best (spanner, nut) pair that minimizes travel + pickup + travel + tighten
                min_cost = float('inf')
                best_spanner_index = -1
                best_nut_task_index = -1

                if not available_spanners:
                     # Need a spanner but none are available. Problem unsolvable.
                     # print("Error: Need spanner but none available.")
                     return float('inf')

                for i_s, (spanner_name, spanner_loc) in enumerate(available_spanners):
                    dist_to_spanner = self.get_distance(current_man_loc, spanner_loc)
                    if dist_to_spanner == float('inf'): continue # Cannot reach this spanner

                    for i_n, (nut_name, nut_loc) in enumerate(loose_nuts_tasks):
                        dist_spanner_to_nut = self.get_distance(spanner_loc, nut_loc)
                        if dist_spanner_to_nut == float('inf'): continue # Cannot reach nut from spanner loc

                        # Cost = travel to spanner + pickup + travel to nut + tighten
                        cost = dist_to_spanner + 1 + dist_spanner_to_nut + 1

                        if cost < min_cost:
                            min_cost = cost
                            best_spanner_index = i_s
                            best_nut_task_index = i_n

                if best_spanner_index == -1 or best_nut_task_index == -1 or min_cost == float('inf'):
                     # Cannot find a reachable spanner/nut pair. Should not happen in solvable problems.
                     # print("Error: Cannot find reachable spanner/nut pair.")
                     return float('inf')

                # Apply the best task sequence
                h += min_cost
                chosen_spanner_name, chosen_spanner_loc = available_spanners.pop(best_spanner_index)
                chosen_nut_name, chosen_nut_loc = loose_nuts_tasks.pop(best_nut_task_index)
                current_man_loc = chosen_nut_loc
                man_has_spanner = False # Spanner used

        return h
