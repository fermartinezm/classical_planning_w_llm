from fnmatch import fnmatch

# Assuming a Heuristic base class is provided by the planning framework
# from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start):
    """
    Performs BFS on a graph to find shortest distances from a start node.
    graph: dict where keys are nodes and values are lists of neighbors.
    start: the starting node.
    Returns: dict mapping nodes to their shortest distance from start.
    """
    # Collect all unique nodes from the graph definition
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    distances = {node: float('inf') for node in all_nodes}

    if start not in all_nodes:
         # Start node is not in the graph at all. This might indicate an issue,
         # but we can treat it as an isolated node with distance 0 to itself.
         # Distances to all other nodes remain inf.
         if start in distances: # Should be in distances if it was in all_nodes
             distances[start] = 0
         else: # Start node wasn't even in the graph definition
             # Add it to distances if it's a valid node name
             distances[start] = 0
         return distances

    distances[start] = 0
    queue = [start]
    while queue:
        current = queue.pop(0)
        # Check if current node has neighbors defined
        if current in graph:
            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances


# class spannerHeuristic(Heuristic): # Inherit if Heuristic base is provided
class spannerHeuristic: # Standalone if Heuristic base is not strictly required or provided
    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal nuts (those that need to be tightened).
        - Static facts (link relationships).
        """
        self.goals = task.goals
        static_facts = task.static

        # Identify nuts that need to be tightened in the goal.
        self.goal_nuts = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "tightened", "*")
        }

        # Build the location graph from link facts.
        self.location_graph = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # Note: Locations mentioned only in 'at' facts in initial/goal state
        # but not in 'link' facts will not be in the graph keys/values initially.
        # BFS handles this by calculating distances only among nodes it knows about.
        # If man_loc or a nut_loc/spanner_loc is isolated, BFS from that node
        # will only find itself (distance 0), and distances to others will be inf.
        # This is correct.

    def __call__(self, node):
        """Estimate the minimum cost to tighten all goal nuts."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Identify current state information
        man_loc = None
        carried_spanners = set() # Store spanner objects carried
        usable_spanners_in_state = set() # Usable spanners anywhere
        spanner_locations = {} # Map spanner object to its location (if on ground)
        nut_locations = {} # Map nut object to its location
        loose_nuts_in_state = set() # Nuts that are currently loose

        # Collect all object locations and states
        all_spanners = set()
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1:]
                if obj.startswith("bob"): # Assuming 'bob' is the man
                    man_loc = loc
                elif obj.startswith("spanner"):
                    all_spanners.add(obj)
                    spanner_locations[obj] = loc # Store location if on ground
                elif obj.startswith("nut"):
                    nut_locations[obj] = loc
            elif match(fact, "carrying", "*", "*"):
                 carrier, spanner = parts[1:]
                 if carrier.startswith("bob"): # Assuming 'bob' is the man
                     all_spanners.add(spanner)
                     carried_spanners.add(spanner) # Store the carried spanner object
            elif match(fact, "loose", "*"):
                 nut = parts[1]
                 loose_nuts_in_state.add(nut)
            # Usable facts are processed below after identifying all spanners

        for fact in state:
             if match(fact, "usable", "*"):
                 spanner = get_parts(fact)[1]
                 if spanner in all_spanners:
                     usable_spanners_in_state.add(spanner)

        # Determine if the man is carrying *any* usable spanner
        man_carrying_usable_spanner = any(s in usable_spanners_in_state for s in carried_spanners)

        # Identify loose nuts that are also goal nuts
        loose_goal_nuts = self.goal_nuts.intersection(loose_nuts_in_state)

        # If no loose goal nuts, goal is reached (checked above)
        if not loose_goal_nuts:
             return 0

        # Calculate distances from the man's current location
        man_distances = bfs(self.location_graph, man_loc)

        # Calculate distances from all usable spanner locations on the ground
        usable_spanner_ground_locations_map = {
            s: spanner_locations[s] for s in usable_spanners_in_state if s in spanner_locations
        }
        spanner_ground_distances = {
             loc: bfs(self.location_graph, loc) for loc in usable_spanner_ground_locations_map.values()
        }

        # Heuristic calculation: Sum of minimum costs for each loose goal nut independently.
        # Cost for nut n at l_n:
        # 1 (tighten)
        # + Cost to get man and a usable spanner to l_n.
        #   - If man carrying usable: dist(man_loc, l_n)
        #   - If not carrying usable: min_{s at l_s usable} ( dist(man_loc, l_s) + 1 + dist(l_s, l_n) )

        h = 0
        for nut in loose_goal_nuts:
            nut_loc = nut_locations.get(nut)
            if nut_loc is None:
                 # Should not happen in valid states where goal nuts have locations
                 # If a goal nut's location is unknown, it's likely unsolvable.
                 return float('inf') # Indicate unsolvable/invalid state

            cost_for_this_nut = 0

            # Cost for tighten action
            cost_for_this_nut += 1

            # Cost to get man and a usable spanner to the nut location
            cost_get_man_and_spanner_to_nut_loc = float('inf')

            # Option 1: Man is already carrying a usable spanner
            if man_carrying_usable_spanner:
                 # Cost is just travel for the man
                 dist_man_to_nut = man_distances.get(nut_loc, float('inf'))
                 cost_get_man_and_spanner_to_nut_loc = dist_man_to_nut
            else:
                 # Option 2: Man is not carrying a usable spanner. Needs to pick one up and bring it.
                 # Find the minimum cost sequence: travel man to spanner, pickup, travel man with spanner to nut.
                 min_cost_pickup_and_travel = float('inf')
                 # Iterate through usable spanners on the ground
                 for spanner, l_s in usable_spanner_ground_locations_map.items():
                      dist_man_to_spanner = man_distances.get(l_s, float('inf'))
                      # Use precomputed spanner distances
                      dist_spanner_to_nut = spanner_ground_distances.get(l_s, {}).get(nut_loc, float('inf'))
                      cost_sequence = dist_man_to_spanner + 1 + dist_spanner_to_nut # Travel to spanner + pickup + travel to nut
                      min_cost_pickup_and_travel = min(min_cost_pickup_and_travel, cost_sequence)

                 cost_get_man_and_spanner_to_nut_loc = min_cost_pickup_and_travel

            # If cost_get_man_and_spanner_to_nut_loc is still inf, it means this nut is unreachable or no usable spanners are available on the ground.
            # If man was carrying a usable spanner, this branch is skipped.
            # If man was NOT carrying a usable spanner, and there are no usable spanners on the ground, then min_cost_pickup_and_travel remains inf.
            # This correctly makes the heuristic inf if no usable spanners can be acquired for this nut.

            # Add the cost to get resources to the nut location
            # Only add if the cost is finite. If it's inf, the total h will become inf.
            if cost_get_man_and_spanner_to_nut_loc == float('inf'):
                 return float('inf') # Indicate unsolvable state

            h += cost_for_this_nut + cost_get_man_and_spanner_to_nut_loc

        return h
