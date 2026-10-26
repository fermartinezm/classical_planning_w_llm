from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import heapq # Included based on thought process, though list.sort is used

# Helper functions

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
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

def build_location_graph_and_locations(task):
    """
    Builds an adjacency list representation of the location graph
    and collects all relevant location names from links, initial state, and goals.
    """
    graph = {}
    all_locations = set()

    # Get locations from static links
    for fact in task.static:
        if match(fact, "link", "*", "*"):
            parts = get_parts(fact)
            if len(parts) == 3:
                _, loc1, loc2 = parts
                graph.setdefault(loc1, set()).add(loc2)
                graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional
                all_locations.add(loc1)
                all_locations.add(loc2)

    # Get locations from initial state 'at' facts
    for fact in task.initial_state:
         if match(fact, "at", "*", "*"):
             parts = get_parts(fact)
             if len(parts) == 3:
                 _, obj, loc = parts
                 all_locations.add(loc)

    # Get locations from goal 'at' facts (if any)
    # Goals can be conjunctions, need to iterate through them.
    goal_facts = task.goals
    if isinstance(task.goals, tuple) and task.goals[0] == 'and':
         goal_facts = frozenset(task.goals[1:])
    elif not isinstance(task.goals, frozenset):
         goal_facts = frozenset({task.goals})

    for fact in goal_facts:
         if match(fact, "at", "*", "*"):
             parts = get_parts(fact)
             if len(parts) == 3:
                 _, obj, loc = parts
                 all_locations.add(loc)

    # Ensure all locations in the graph are in the set (redundant but safe)
    all_locations.update(graph.keys())

    return graph, list(all_locations) # Return list for consistent ordering if needed

def bfs_shortest_path(graph, start_node, all_locations):
    """
    Computes shortest path distances from a start node to all other nodes
    within the connected components of the graph.
    """
    distances = {loc: float('inf') for loc in all_locations}
    if start_node not in all_locations:
        # Start node is not a known location, cannot start path from here
        return distances # All distances remain infinity

    # If the start node exists but has no links defined in the graph,
    # it's an isolated node. Distance to itself is 0, others are inf.
    if start_node not in graph and start_node in all_locations:
         distances[start_node] = 0
         return distances


    distances[start_node] = 0
    queue = deque([start_node])

    # Only traverse links from nodes that are actually keys in the graph structure
    nodes_with_links = set(graph.keys())

    while queue:
        current_loc = queue.popleft()
        current_dist = distances[current_loc]

        if current_loc in nodes_with_links: # Check if the location has defined links
            for neighbor in graph[current_loc]:
                # Ensure neighbor is a known location before updating distance
                if neighbor in all_locations and distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(task):
    """Computes shortest path distances between all pairs of relevant locations."""
    graph, all_locations = build_location_graph_and_locations(task)

    distances = {}
    # Compute distances from all relevant locations
    for start_node in all_locations:
         distances[start_node] = bfs_shortest_path(graph, start_node, all_locations)

    # Return a function to easily get distance
    def get_distance(loc1, loc2):
        if loc1 not in distances or loc2 not in distances[loc1]:
             # This case should ideally not happen if all_locations is complete
             # but serves as a safeguard.
             return float('inf')
        return distances[loc1][loc2]

    return get_distance, all_locations # Also return the list of locations


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost based on the number of loose goal nuts,
    the number of spanners needed, and the travel required to
    pick up spanners and visit nut locations.

    The heuristic calculates:
    1. The number of 'tighten_nut' actions needed (equal to the number of loose goal nuts).
    2. The number of 'pickup_spanner' actions needed (number of loose goal nuts minus carried spanners, minimum 0).
    3. The estimated travel cost for the man to first visit locations of needed spanners
       and then visit locations of loose goal nuts. Travel is estimated using a greedy
       closest-first strategy for each stage.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and identifying
        key objects and goals.
        """
        self.task = task
        self.goals = task.goals

        # Identify objects and their types based on predicates in initial state and goals
        potential_men = set()
        potential_spanners = set()
        potential_nuts = set()

        # Process initial state
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'carrying' and len(parts) == 3:
                     potential_men.add(parts[1])
                     potential_spanners.add(parts[2])
                elif parts[0] == 'usable' and len(parts) == 2:
                     potential_spanners.add(parts[1])
                elif parts[0] == 'loose' and len(parts) == 2:
                     potential_nuts.add(parts[1])
                # 'at' facts are processed during location identification and __call__

        # Process goals
        goal_facts = task.goals
        if isinstance(task.goals, tuple) and task.goals[0] == 'and':
             goal_facts = frozenset(task.goals[1:])
        elif not isinstance(task.goals, frozenset):
             goal_facts = frozenset({task.goals})

        for fact in goal_facts:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'tightened' and len(parts) == 2:
                    potential_nuts.add(parts[1])
                # 'at' goals are processed during location identification

        # Infer man name: object in potential_men, likely only one
        self.man_name = None
        if potential_men:
            self.man_name = list(potential_men)[0]
        else:
             # Fallback: Try to find an object in 'at' facts in initial state
             # that is not a potential spanner or nut.
             inferred_men = set()
             for fact in task.initial_state:
                 if match(fact, "at", "*", "*"):
                     obj = get_parts(fact)[1]
                     if obj not in potential_spanners and obj not in potential_nuts:
                         inferred_men.add(obj)
             if inferred_men:
                 self.man_name = list(inferred_men)[0]
             # If still not found, man_name remains None. Heuristic will return inf if man_loc isn't found.


        # Store identified spanners and nuts for use in __call__
        self.all_spanners = potential_spanners
        self.all_nuts = potential_nuts

        # Identify goal nuts (those that need to be tightened)
        self.goal_nuts = set()
        for goal in goal_facts:
            if match(goal, "tightened", "*"):
                parts = get_parts(goal)
                if len(parts) == 2:
                    _, nut = parts
                    self.goal_nuts.add(nut)

        # Precompute all-pairs shortest paths between locations
        self.get_distance, self.all_locations = compute_all_pairs_shortest_paths(task)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location
        man_loc = None
        if self.man_name: # Only look for man's location if man was identified
            for fact in state:
                if match(fact, "at", self.man_name, "*"):
                    parts = get_parts(fact)
                    if len(parts) == 3:
                        man_loc = parts[2]
                        break

        if man_loc is None or man_loc not in self.all_locations:
             # Man's location not found or is at an unknown/unreachable location
             return float('inf')

        # 2. Identify loose goal nuts and their locations
        loose_goal_nuts_info = [] # List of (nut_name, nut_location)
        current_object_locations = {} # Map object_name to location
        for fact in state:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                    current_object_locations[parts[1]] = parts[2]

        for nut in self.goal_nuts:
            if f"(loose {nut})" in state:
                if nut in current_object_locations:
                    nut_loc = current_object_locations[nut]
                    if nut_loc not in self.all_locations: # Check if location is known
                         return float('inf') # Nut at unreachable location
                    loose_goal_nuts_info.append((nut, nut_loc))
                else:
                    # Loose goal nut not found at any location in state? Unreachable.
                    # This shouldn't happen in valid states if initial state was valid.
                    return float('inf')


        N_nuts = len(loose_goal_nuts_info)
        if N_nuts == 0:
            return 0 # Goal reached

        nut_locations = [loc for _, loc in loose_goal_nuts_info]

        # 3. Identify spanners carried by the man
        carried_spanners = [get_parts(fact)[2] for fact in state if match(fact, "carrying", self.man_name, "*")]
        N_carried = len(carried_spanners)

        # 4. Identify usable spanners on the ground and their locations
        usable_spanners_on_ground_info = [] # List of (spanner_name, spanner_location)
        for spanner_name in self.all_spanners:
             if f"(usable {spanner_name})" in state:
                  if spanner_name in current_object_locations:
                       spanner_loc = current_object_locations[spanner_name]
                       if spanner_loc not in self.all_locations: # Check if location is known
                            continue # Ignore spanners at unknown locations
                       usable_spanners_on_ground_info.append((spanner_name, spanner_loc))
                  # else: usable spanner not at any location? Ignore.

        # 5. Calculate number of spanners to pick up
        N_pickup = max(0, N_nuts - N_carried)

        # 6. Check if enough usable spanners exist (carried + on ground)
        if N_pickup > len(usable_spanners_on_ground_info):
            return float('inf') # Not enough usable spanners in the world

        # 7. Calculate cost components
        # Base cost: tighten actions + pickup actions
        base_cost = N_nuts + N_pickup

        # Travel cost:
        travel_cost = 0
        current_travel_loc = man_loc

        # Step 7a: Travel to pick up needed spanners
        if N_pickup > 0:
            # Select the N_pickup closest usable spanners on the ground from current_travel_loc
            # Sort usable spanners on ground by distance from current_travel_loc
            usable_spanners_on_ground_info.sort(key=lambda item: self.get_distance(current_travel_loc, item[1]))
            pickup_spanner_locs = [loc for _, loc in usable_spanners_on_ground_info[:N_pickup]]

            # Travel to these pickup locations in greedy closest-first order
            remaining_pickup_locs = list(pickup_spanner_locs)
            while remaining_pickup_locs:
                 closest_loc = None
                 min_dist = float('inf')
                 closest_idx = -1

                 for i, loc in enumerate(remaining_pickup_locs):
                     dist = self.get_distance(current_travel_loc, loc)
                     if dist < min_dist:
                         min_dist = dist
                         closest_loc = loc
                         closest_idx = i

                 if closest_loc is None or min_dist == float('inf'):
                      return float('inf') # Cannot reach a pickup location

                 travel_cost += min_dist
                 current_travel_loc = closest_loc
                 remaining_pickup_locs.pop(closest_idx)

        # Step 7b: Travel to visit nut locations
        # Travel to nut locations in greedy closest-first order from the current location (after pickups)
        remaining_nut_locs = list(nut_locations)
        while remaining_nut_locs:
             closest_loc = None
             min_dist = float('inf')
             closest_idx = -1

             for i, loc in enumerate(remaining_nut_locs):
                 dist = self.get_distance(current_travel_loc, loc)
                 if dist < min_dist:
                     min_dist = dist
                     closest_loc = loc
                     closest_idx = i

             if closest_loc is None or min_dist == float('inf'):
                  return float('inf') # Cannot reach a nut location

             travel_cost += min_dist
             current_travel_loc = closest_loc
             remaining_nut_locs.pop(closest_idx)


        # Total heuristic cost
        total_cost = base_cost + travel_cost

        return total_cost
