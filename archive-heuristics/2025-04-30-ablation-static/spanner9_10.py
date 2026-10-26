from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

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
    # Ensure we don't try to match more args than parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_graph(static_facts):
    """Build a graph of locations based on 'link' facts."""
    graph = {}
    locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'link':
            loc1, loc2 = parts[1], parts[2]
            locations.add(loc1)
            locations.add(loc2)
            graph.setdefault(loc1, set()).add(loc2)
            graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional
    return graph, list(locations)

def bfs(graph, start_node):
    """Compute shortest distances from a start node using BFS."""
    distances = {node: float('inf') for node in graph}
    if start_node in graph: # Ensure start_node is in the graph
        distances[start_node] = 0
        queue = deque([start_node])
        while queue:
            current = queue.popleft()
            if current in graph: # Handle nodes that might be in locations list but not linked to anything
                for neighbor in graph[current]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph, locations):
    """Compute shortest distances between all pairs of locations."""
    all_distances = {}
    # Ensure all locations from the problem are considered, even if isolated in the graph
    all_nodes_in_graph = set(graph.keys())
    all_relevant_locations = set(locations) | all_nodes_in_graph

    for start_loc in all_relevant_locations:
        all_distances[start_loc] = bfs(graph, start_loc)
    return all_distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It simulates a greedy strategy: repeatedly finds the closest usable spanner
    (if the man is not carrying enough), travels to it, picks it up, then travels
    to the closest untightened goal nut, and tightens it. The estimated cost is
    the sum of travel distances (shortest path) and action costs (pickup, tighten).

    # Assumptions
    - Nuts and spanners have fixed locations (except when carried). Their initial
      locations are their permanent ground locations.
    - Links between locations are bidirectional.
    - Action costs are 1 for walk, pickup_spanner, and tighten_nut.
    - The man can carry multiple spanners.
    - The problem is solvable, implying enough usable spanners exist initially.
    - Object names follow a convention (e.g., 'bob' for man, 'nut' for nuts, 'spanner' for spanners).

    # Heuristic Initialization
    - Identify all goal nuts from the task's goal conditions.
    - Build a graph of locations based on `link` facts from static information.
    - Compute all-pairs shortest paths between all known locations.
    - Store the fixed ground locations of all nuts and spanners from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the set of loose nuts that are also goal conditions. If this set is empty,
       the heuristic value is 0 (goal reached).
    2. Find the man's current location.
    3. Identify the set of usable spanners currently available: those carried by the man
       and those on the ground marked as usable.
    4. Check if the total number of usable spanners available is less than the number
       of untightened goal nuts. If so, the problem is unsolvable from this state,
       return infinity (or a large number).
    5. Initialize the estimated cost to 0.
    6. Initialize the current location for the greedy simulation to the man's location.
    7. Initialize the set of untightened nuts to the set of loose goal nuts found in step 1.
    8. Initialize the set of available usable spanners on the ground.
    9. Initialize the count of usable spanners currently carried by the man.
    10. While there are still untightened nuts:
        a. Determine how many *more* spanners are needed from the ground. This is the
           number of remaining nuts minus the number of spanners currently carried,
           but not less than zero.
        b. If more spanners are needed (`spanners_to_pick_up_count > 0`) and there
           are usable spanners available on the ground:
            i. Find the usable spanner on the ground that is closest to the man's
               current location (in the simulation).
            ii. Add the shortest path distance to this spanner's location to the cost.
            iii. Add 1 to the cost for the `pickup_spanner` action.
            iv. Update the man's current location in the simulation to the spanner's location.
            v. Remove the picked-up spanner from the set of available ground spanners.
            vi. Increment the count of spanners carried by the man.
        c. If the man is not carrying any usable spanners at this point (which implies
           step 10b could not satisfy the need, likely because no usable spanners
           were on the ground, contradicting step 4), return infinity.
        d. Find the untightened nut that is closest to the man's current location
           (in the simulation).
        e. Add the shortest path distance to this nut's location to the cost.
        f. Add 1 to the cost for the `tighten_nut` action.
        g. Update the man's current location in the simulation to the nut's location.
        h. Remove the tightened nut from the set of untightened nuts.
        i. Decrement the count of spanners carried by the man (one spanner is used).
    11. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find static object locations

        # 1. Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'tightened'}

        # 2. Build location graph and compute distances
        graph, locations = build_graph(static_facts)
        self.distances = compute_all_pairs_shortest_paths(graph, locations)
        # Store all locations encountered in graph or initial state for distance lookup
        self.all_locations = set(locations) | set(graph.keys())
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                  self.all_locations.add(parts[2])


        # 3. Store initial locations of nuts and spanners (they are static)
        self.nut_locations = {}
        self.spanner_locations = {}
        # Assuming object types can be inferred from names (domain-dependent)
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('nut'):
                    self.nut_locations[obj] = loc
                elif obj.startswith('spanner'):
                    self.spanner_locations[obj] = loc

        # Ensure all goal nuts have known locations
        for nut in self.goal_nuts:
            if nut not in self.nut_locations:
                 # This indicates a potentially malformed problem instance
                 print(f"Warning: Location for goal nut {nut} not found in initial state.")
                 # Heuristic might return inf later if location is unreachable


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose goal nuts in the current state
        current_loose_goal_nuts = {
            nut for nut in self.goal_nuts
            if f'(loose {nut})' in state # Check if it's still loose
        }

        # If all goal nuts are tightened, heuristic is 0
        if not current_loose_goal_nuts:
            return 0

        # 2. Find man's current location
        man_location = None
        carried_spanners = set() # Usable spanners carried by man
        usable_spanners_on_ground = set() # Usable spanners on the ground

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('bob'): # Assuming man is named 'bob'
                 man_location = parts[2]
            elif parts[0] == 'carrying' and parts[1].startswith('bob'):
                 spanner = parts[2]
                 # Check if the carried spanner is usable
                 if f'(usable {spanner})' in state:
                     carried_spanners.add(spanner)
            elif parts[0] == 'at' and parts[1].startswith('spanner'):
                 spanner, loc = parts[1], parts[2]
                 # Check if the spanner on the ground is usable
                 if f'(usable {spanner})' in state:
                     usable_spanners_on_ground.add(spanner)

        # Combine all usable spanners available in the current state
        current_usable_spanners = carried_spanners | usable_spanners_on_ground

        # 4. Check solvability based on spanners
        if len(current_loose_goal_nuts) > len(current_usable_spanners):
             # Not enough usable spanners to tighten all remaining nuts
             return float('inf') # Indicate unsolvable from here

        # Need a way to get distance between locations
        def get_dist(loc1, loc2):
            if loc1 not in self.all_locations or loc2 not in self.all_locations:
                 # Locations not known from initial state or links - should not happen in valid PDDL
                 print(f"Error: Unknown location(s) encountered: {loc1}, {loc2}")
                 return float('inf') # Cannot reach

            # BFS might not find paths if locations are isolated.
            # The compute_all_pairs_shortest_paths should handle this by including all locations.
            # If a location is unreachable from another, its distance will be inf.
            if loc1 not in self.distances or loc2 not in self.distances[loc1]:
                 # This means loc2 is unreachable from loc1
                 return float('inf')

            return self.distances[loc1][loc2]


        # 5-9. Initialize cost and state for greedy simulation
        cost = 0
        current_loc = man_location
        nuts_to_tighten = set(current_loose_goal_nuts)
        available_spanners_ground_sim = set(usable_spanners_on_ground) # Usable spanners on ground for simulation
        num_spanners_carried_sim = len(carried_spanners) # Count of usable spanners carried for simulation

        # 10. Greedy simulation
        while nuts_to_tighten:
            # a. Determine how many more spanners are needed from the ground
            spanners_to_pick_up_count = max(0, len(nuts_to_tighten) - num_spanners_carried_sim)

            # b. If we need to pick up spanners and there are usable spanners on the ground:
            if spanners_to_pick_up_count > 0 and available_spanners_ground_sim:
                 # Find the closest available spanner on the ground
                 closest_spanner = None
                 min_dist_spanner = float('inf')
                 spanner_loc_map = {s: self.spanner_locations[s] for s in available_spanners_ground_sim}

                 for spanner, loc in spanner_loc_map.items():
                      d = get_dist(current_loc, loc)
                      if d < min_dist_spanner:
                          min_dist_spanner = d
                          closest_spanner = spanner

                 if closest_spanner is None or min_dist_spanner == float('inf'):
                      # Cannot reach any available spanner on the ground
                      return float('inf')

                 cost += min_dist_spanner # Travel to spanner (walk actions)
                 cost += 1 # Pickup spanner (pickup_spanner action)
                 current_loc = spanner_loc_map[closest_spanner]
                 available_spanners_ground_sim.remove(closest_spanner)
                 num_spanners_carried_sim += 1 # Now carrying one more

            # c. If after potentially picking up, we still don't have a spanner to use for the next nut
            # (i.e., num_spanners_carried_sim is 0, but nuts_to_tighten > 0)
            if num_spanners_carried_sim == 0:
                 # This implies we needed a spanner but couldn't get one (either none on ground or unreachable)
                 # This state should have been caught by the initial check, but defensive programming.
                 return float('inf')

            # d. Find the untightened nut that is closest to the man's current location
            closest_nut = None
            min_dist_nut = float('inf')
            nut_loc_map = {n: self.nut_locations[n] for n in nuts_to_tighten}

            for nut, loc in nut_loc_map.items():
                 d = get_dist(current_loc, loc)
                 if d < min_dist_nut:
                     min_dist_nut = d
                     closest_nut = nut

            if closest_nut is None or min_dist_nut == float('inf'):
                 # Cannot reach any remaining nut
                 return float('inf')

            cost += min_dist_nut # Travel to nut (walk actions)
            cost += 1 # Tighten nut (tighten_nut action)
            current_loc = nut_loc_map[closest_nut]
            nuts_to_tighten.remove(closest_nut)
            num_spanners_carried_sim -= 1 # Use one spanner

        # 11. Return the total estimated cost.
        return cost

