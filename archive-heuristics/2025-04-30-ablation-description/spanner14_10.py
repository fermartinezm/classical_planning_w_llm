# Assume Heuristic base class is provided by the environment
# from heuristics.heuristic_base import Heuristic

import collections
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return [] # Not a valid fact string format we expect
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

def bfs(graph, start_node):
    """
    Perform Breadth-First Search to find shortest distances from start_node.

    Args:
        graph: Adjacency list representation of the graph {node: [neighbors]}.
        start_node: The starting node for the BFS.

    Returns:
        A dictionary mapping reachable nodes to their shortest distance from start_node.
        Returns an empty dictionary if start_node is not in the graph.
    """
    if start_node not in graph:
        return {} # Start node not in graph

    distances = {start_node: 0}
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Ensure current_node has neighbors in the graph (it should if it's a key)
        for neighbor in graph.get(current_node, []):
            if neighbor not in distances:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten the *first* remaining loose nut
    that is required by the goal. It calculates the cost as the sum of:
    1. Travel cost for the man to reach the nut's location.
    2. Cost to acquire a usable spanner (if not already carrying one). This involves
       traveling to the spanner's location and picking it up. The heuristic
       chooses the closest available usable spanner.
    3. Cost of the tighten action.

    # Assumptions
    - There is exactly one man.
    - Nuts and spanners are locatable objects.
    - The goal is to tighten one or more specific nuts.
    - Spanners become unusable after one use (tightening a nut).
    - The locations form a connected graph (or relevant locations are connected).
    - All necessary objects (man, nuts, usable spanners) exist and are initially
      placed in locations or carried.

    # Heuristic Initialization
    - Extracts the set of nuts that must be tightened according to the goal.
    - Identifies the name of the man.
    - Builds the graph of locations based on `link` facts from static information.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify which goal nuts are still in a `loose` state. These are the
       remaining goal nuts to be tightened.
    2. If no goal nuts are loose, the goal is effectively reached (for the nuts),
       and the heuristic value is 0.
    3. Select the first nut from the list of remaining loose goal nuts.
    4. Find the current location of the man.
    5. Find the current location of the selected nut.
    6. Check if the man is currently carrying a usable spanner.
       - Iterate through the state facts to find `(carrying <man> <spanner>)`
         and check if `(usable <spanner>)` is also in the state.
    7. If the man is carrying a usable spanner:
       - The cost is the shortest distance from the man's current location
         to the nut's location, plus 1 for the `tighten_nut` action.
    8. If the man is NOT carrying a usable spanner:
       - Find all usable spanners that are currently at some location (i.e.,
         not being carried).
       - If no such spanners exist, the state is likely unsolvable for this nut
         (as spanners are consumed), return infinity.
       - For each available usable spanner, calculate the total cost to use it
         for the current nut:
         - Distance from the man's location to the spanner's location.
         - Plus 1 for the `pickup_spanner` action.
         - Plus distance from the spanner's location (where man now is) to the nut's location.
         - Plus 1 for the `tighten_nut` action.
       - Choose the minimum cost among all available usable spanners.
       - Return this minimum cost.
    9. If any required location (man's, nut's, spanner's) is not found in the
       precomputed distance map (meaning it's unreachable in the location graph),
       return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing location distances.
        """
        self.task = task
        self.goals = task.goals

        # 1. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        # 2. Identify the man
        self.man_name = None
        # Try finding the man from 'carrying' facts in initial state
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'carrying' and len(parts) == 3:
                self.man_name = parts[1]
                break # Assuming only one man

        # If not found in 'carrying', try finding the first object in an 'at' fact
        if self.man_name is None:
             initial_at_facts = [fact for fact in task.initial_state if match(fact, "at", "*", "*")]
             if initial_at_facts:
                  self.man_name = get_parts(initial_at_facts[0])[1]
             # else: man_name remains None, __call__ will handle this.


        # 3. Build location graph and compute distances
        self.adj = collections.defaultdict(list)
        locations = set()

        # Get locations from link facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.adj[l1].append(l2)
                self.adj[l2].append(l1)
                locations.add(l1)
                locations.add(l2)

        # Get locations from initial 'at' facts (covers locations with objects but no links)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) == 3:
                 # The second argument is the location
                 locations.add(parts[2])

        # Ensure all identified locations are keys in adj even if they have no links
        for loc in locations:
             if loc not in self.adj:
                 self.adj[loc] = []

        # Compute distances only if we have locations
        self.distances = {}
        if locations:
            for loc in self.adj:
                self.distances[loc] = bfs(self.adj, loc)


    def __call__(self, node):
        """
        Estimate the minimum cost to tighten the first remaining goal nut.
        """
        state = node.state

        # Handle case where man_name wasn't found during init
        if self.man_name is None:
             # Cannot compute heuristic without knowing the man
             return float('inf')

        # 1. Identify remaining loose goal nuts
        nuts_remaining = [
            nut for nut in self.goal_nuts if f'(loose {nut})' in state
        ]

        # 2. If no goal nuts are loose, goal is reached (for nuts)
        if not nuts_remaining:
            return 0

        # 3. Select the first nut
        target_nut = nuts_remaining[0]

        # 4. Find man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_location = get_parts(fact)[2]
                break

        if man_location is None:
             # Man's location not found - inconsistent state?
             return float('inf') # Cannot proceed

        # 5. Find nut's current location
        nut_location = None
        for fact in state:
            if match(fact, "at", target_nut, "*"):
                nut_location = get_parts(fact)[2]
                break

        if nut_location is None:
             # Nut's location not found - inconsistent state?
             return float('inf') # Cannot proceed

        # Check if locations are in the precomputed distances map
        # If man_location is not in self.distances, it means it wasn't in the graph built from links/initial@.
        # If nut_location is not in self.distances[man_location], it means nut_location is unreachable from man_location.
        if man_location not in self.distances or nut_location not in self.distances.get(man_location, {}):
             # Man or nut is in an unreachable location from the main graph
             return float('inf')


        # 6. Check if man is carrying a usable spanner
        carrying_usable_spanner = False
        carried_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner = get_parts(fact)[2]
                # Check if the carried spanner is usable
                if f'(usable {carried_spanner})' in state:
                    carrying_usable_spanner = True
                break # Assuming man can carry at most one spanner based on domain actions

        # 7. Calculate cost
        if carrying_usable_spanner:
            # Man has a usable spanner, just needs to get to the nut
            travel_cost = self.distances[man_location][nut_location]

            # Cost = travel + tighten
            return travel_cost + 1
        else:
            # Man needs to find and pick up a usable spanner first
            available_usable_spanners = []
            for fact in state:
                # Find usable spanners
                if match(fact, "usable", "*"):
                    spanner_name = get_parts(fact)[1]
                    # Check if it's NOT being carried by the man
                    if not match(f'(carrying {self.man_name} {spanner_name})', "carrying", self.man_name, spanner_name):
                         # Check if it's at a location (not just usable but nowhere)
                         spanner_at_location = False
                         spanner_loc = None
                         for at_fact in state:
                             if match(at_fact, "at", spanner_name, "*"):
                                 spanner_at_location = True
                                 spanner_loc = get_parts(at_fact)[2]
                                 break
                         if spanner_at_location:
                             available_usable_spanners.append((spanner_name, spanner_loc))

            # 8. If no available usable spanners, unsolvable for this nut
            if not available_usable_spanners:
                return float('inf')

            # Find the best spanner to use (minimizing travel + pickup + travel + tighten)
            min_total_cost = float('inf')

            for spanner_name, spanner_loc in available_usable_spanners:
                # Check if spanner location is reachable from man's location
                dist_man_to_spanner = self.distances[man_location].get(spanner_loc, float('inf'))
                if dist_man_to_spanner == float('inf'):
                    continue # This spanner is in an unreachable location

                # Check if nut location is reachable from spanner's location (where man will be after pickup)
                # Need distance from spanner_loc to nut_location
                dist_spanner_to_nut = self.distances.get(spanner_loc, {}).get(nut_location, float('inf'))
                if dist_spanner_to_nut == float('inf'):
                     continue # Nut is unreachable from spanner location

                # Cost = travel_to_spanner + pickup + travel_to_nut + tighten
                current_spanner_cost = dist_man_to_spanner + 1 + dist_spanner_to_nut + 1
                min_total_cost = min(min_total_cost, current_spanner_cost)

            # If no spanner path was found (all available spanners are in unreachable parts of graph)
            if min_total_cost == float('inf'):
                 return float('inf')

            return min_total_cost
