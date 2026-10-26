import heapq
from collections import deque
import logging

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
    This heuristic estimates the cost to reach the goal by summing three components:
    1. The number of loose goal nuts (representing the minimum number of tighten_nut actions).
    2. The number of additional spanners the man needs to pick up (representing pickup_spanner actions).
    3. An estimate of the walk actions required to reach all necessary locations (nut locations and spanner pickup locations), calculated using a Minimum Spanning Tree (MST) on the relevant locations plus the distance from the man's current location to the closest relevant location.

    Assumptions:
    - There is a single man object in the domain.
    - Spanners are consumed after one use (as per domain effects).
    - Links between locations are bidirectional.
    - The state representation allows reliable identification of the man, nuts, spanners, and locations based on predicate arguments and goal facts. Specifically, it assumes:
        - Objects in goal facts like `(tightened ?n)` are goal nuts.
        - Objects in state facts like `(loose ?n)` are nuts.
        - Objects in state facts like `(usable ?s)` or as the second argument of `(carrying ?m ?s)` are spanners.
        - Objects in static facts like `(link ?l1 ?l2)` or as the second argument of `(at ?o ?l)` are locations.
        - The man is the unique locatable object that is not a nut or a spanner.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes the following:
    - Identifies goal nut names from the task goals.
    - Builds the location graph based on static `link` facts.
    - Computes all-pairs shortest paths between locations using BFS, storing distances.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Parse the state to identify:
       - The man's current location (`l_m`).
       - Whether the man is currently carrying a usable spanner (`has_usable_spanner`).
       - The set of goal nuts that are currently loose (`LooseGoalNuts`).
       - The location of each loose goal nut (`nut_locations`).
       - The set of usable spanners currently at locations (not carried by the man) and their locations (`usable_spanners_at_locs`).
       - The total count of usable spanners available (carried or at locations).
    2. Calculate `N_loose_nuts`, the number of loose goal nuts. If 0, the goal is reached, return 0.
    3. Check for unsolvability: If the total number of usable spanners is less than `N_loose_nuts`, return `float('inf')`.
    4. Calculate `N_needed_spanners`, the number of additional spanners the man needs to pick up. This is `N_loose_nuts` if the man doesn't have a usable spanner, or `N_loose_nuts - 1` if he does (minimum 0).
    5. Select `N_needed_spanners` usable spanner locations from `usable_spanners_at_locs` that are closest to the man's current location (`l_m`). Let this set be `selected_spanner_locs`.
    6. Define the set of target locations for the man's travel: `nodes_for_mst` consists of the locations of all loose goal nuts plus the `selected_spanner_locs`.
    7. Calculate `min_dist_to_targets`, the minimum distance from the man's current location (`l_m`) to any location in `nodes_for_mst`. If any distance is infinite, return `float('inf')`.
    8. Calculate `mst_internal_cost`, the cost of the Minimum Spanning Tree connecting all locations within `nodes_for_mst`. If the locations are disconnected, return `float('inf')`.
    9. The estimated `walk_cost` is `mst_internal_cost + min_dist_to_targets`.
    10. The final heuristic value is the sum of `N_loose_nuts` (tighten actions), `N_needed_spanners` (pickup actions), and `walk_cost`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goal_nut_names = set()

        # Parse task.goals to get goal nut names
        for goal_fact_str in task.goals:
            pred, args = self.parse_fact(goal_fact_str)
            if pred == 'tightened' and len(args) == 1:
                self.goal_nut_names.add(args[0])

        # Build location graph from static facts and infer location names
        self.location_names = set()
        link_facts = set()
        for static_fact_str in task.static:
            pred, args = self.parse_fact(static_fact_str)
            if pred == 'link' and len(args) == 2:
                loc1, loc2 = args
                self.location_names.add(loc1)
                self.location_names.add(loc2)
                link_facts.add((loc1, loc2))

        self.location_graph = {loc: [] for loc in self.location_names}
        for loc1, loc2 in link_facts:
            self.location_graph[loc1].append(loc2)
            self.location_graph[loc2].append(loc1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.location_names:
            self.distances[start_node] = {}
            q = deque([start_node])
            self.distances[start_node][start_node] = 0
            visited = {start_node}
            while q:
                curr = q.popleft()
                dist_curr = self.distances[start_node][curr]
                for neighbor in self.location_graph.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_node][neighbor] = dist_curr + 1
                        q.append(neighbor)

    def parse_fact(self, fact_str):
        """Helper to parse a PDDL fact string."""
        # Example: '(at bob shed)' -> ('at', ['bob', 'shed'])
        # Example: '(tightened nut1)' -> ('tightened', ['nut1'])
        parts = fact_str[1:-1].split()
        return parts[0], parts[1:]

    def unparse_fact(self, predicate, args):
         """Helper to unparse predicate and args into a fact string."""
         # Example: ('at', ['bob', 'shed']) -> '(at bob shed)'
         return f"({predicate} {' '.join(args)})"

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance between two locations."""
        if loc1 not in self.location_names or loc2 not in self.location_names:
             # Should not happen if parsing is correct, but safety check
             return float('inf')
        return self.distances.get(loc1, {}).get(loc2, float('inf'))

    def compute_mst_cost(self, nodes):
        """Helper to compute MST cost for a set of locations."""
        if not nodes:
            return 0
        if len(nodes) == 1:
            return 0

        cost = 0
        in_tree = set()
        # Start Prim's from an arbitrary node in the set
        start_node = next(iter(nodes))
        in_tree.add(start_node)
        edges = [] # Priority queue (weight, neighbor)

        for neighbor in nodes - in_tree:
            dist = self.get_distance(start_node, neighbor)
            if dist != float('inf'):
                heapq.heappush(edges, (dist, neighbor))

        while len(in_tree) < len(nodes) and edges:
            w, v = heapq.heappop(edges)
            if v not in in_tree:
                in_tree.add(v)
                cost += w
                # Add edges from the newly added node v to nodes not yet in tree
                for next_neighbor in nodes - in_tree:
                     dist = self.get_distance(v, next_neighbor)
                     if dist != float('inf'):
                         heapq.heappush(edges, (dist, next_neighbor))

        # If not all nodes could be included in the tree, they are disconnected
        if len(in_tree) < len(nodes):
            return float('inf')
        return cost


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state
        state_facts_parsed = {self.parse_fact(f) for f in state}

        # --- 1. Parse state and infer object roles ---
        man_name = None
        l_m = None
        carried_spanner = None
        has_usable_spanner = False
        loose_goal_nuts = set()
        nut_locations = {} # nut_name -> location
        usable_spanners_at_locs = set() # (spanner_name, location)
        current_usable_spanners = set() # Names of usable spanners currently existing

        # Infer object roles (man, nuts, spanners) from state facts
        inferred_nut_names = self.goal_nut_names | {args[0] for pred, args in state_facts_parsed if pred == 'loose' and len(args) == 1}
        inferred_spanner_names = {args[1] for pred, args in state_facts_parsed if pred == 'carrying' and len(args) == 2} | {args[0] for pred, args in state_facts_parsed if pred == 'usable' and len(args) == 1}
        inferred_locatable_names = {args[0] for pred, args in state_facts_parsed if pred == 'at' and len(args) == 2}

        # Man is the unique locatable object that is not an inferred nut or spanner
        man_candidates = inferred_locatable_names - inferred_nut_names - inferred_spanner_names
        if len(man_candidates) == 1:
            man_name = man_candidates.pop()
        else:
             # Fallback: Assume the first argument of 'carrying' is the man if someone is carrying.
             carrying_agents = {args[0] for pred, args in state_facts_parsed if pred == 'carrying' and len(args) == 2}
             if len(carrying_agents) == 1:
                 man_name = carrying_agents.pop()
             else:
                 # Cannot reliably identify man. This state might be malformed or unsolvable.
                 logging.warning("Could not reliably identify the man object in state.")
                 return float('inf') # Safety fallback

        # Find man's current location and carried spanner
        for pred, args in state_facts_parsed:
            if pred == 'at' and len(args) == 2 and args[0] == man_name:
                l_m = args[1]
            elif pred == 'carrying' and len(args) == 2 and args[0] == man_name:
                carried_spanner = args[1]

        # Check if carried spanner is usable
        if carried_spanner and ('usable', [carried_spanner]) in state_facts_parsed:
            has_usable_spanner = True
            current_usable_spanners.add(carried_spanner)

        # Find loose goal nuts and their locations
        for nut_name in self.goal_nut_names:
            if ('loose', [nut_name]) in state_facts_parsed:
                loose_goal_nuts.add(nut_name)
                # Find location of this loose nut
                for pred, args in state_facts_parsed:
                    if pred == 'at' and len(args) == 2 and args[0] == nut_name:
                        nut_locations[nut_name] = args[1]
                        break # Found location, move to next nut

        # If any loose goal nut location wasn't found, something is wrong.
        if len(nut_locations) != len(loose_goal_nuts):
             logging.warning(f"Could not find location for all loose goal nuts. Loose goal nuts: {loose_goal_nuts}, Found locations: {nut_locations.keys()}")
             return float('inf') # Should not happen in valid states

        # Find usable spanners at locations (not carried)
        for spanner_name in inferred_spanner_names: # Iterate over inferred spanners
            if spanner_name != carried_spanner and ('usable', [spanner_name]) in state_facts_parsed:
                 current_usable_spanners.add(spanner_name)
                 # Find location of this usable spanner
                 for pred, args in state_facts_parsed:
                     if pred == 'at' and len(args) == 2 and args[0] == spanner_name:
                         usable_spanners_at_locs.add((spanner_name, args[1]))
                         break # Found location, move to next spanner

        # --- 2. Calculate heuristic components ---
        N_loose_nuts = len(loose_goal_nuts)

        # Goal reached
        if N_loose_nuts == 0:
            return 0

        # Check total usable spanners vs loose nuts
        if len(current_usable_spanners) < N_loose_nuts:
             return float('inf') # Unsolvable

        N_needed_spanners = max(0, N_loose_nuts - (1 if has_usable_spanner else 0))

        # Select N_needed_spanners closest usable spanner locations to l_m
        available_spanner_locs_list = list(usable_spanners_at_locs) # List of (spanner_name, loc)
        # Sort by distance from l_m
        available_spanner_locs_list.sort(key=lambda item: self.get_distance(l_m, item[1]))
        selected_spanner_locs = {loc for spanner, loc in available_spanner_locs_list[:N_needed_spanners]}

        # Nodes for MST: locations of loose goal nuts + selected spanner locations
        nodes_for_mst = set(nut_locations.values()) | selected_spanner_locs

        # Calculate min distance from man's location to any node in MST set
        min_dist_to_targets = float('inf')
        for loc in nodes_for_mst:
            dist = self.get_distance(l_m, loc)
            if dist == float('inf'):
                 # Man cannot reach a required location
                 return float('inf')
            min_dist_to_targets = min(min_dist_to_targets, dist)

        # Calculate MST cost for the target locations
        mst_internal_cost = self.compute_mst_cost(nodes_for_mst)

        if mst_internal_cost == float('inf'):
            # Target locations are disconnected from each other
            return float('inf')

        walk_cost = mst_internal_cost + min_dist_to_targets

        # Heuristic = tighten actions + pickup actions + walk actions
        h_value = N_loose_nuts + N_needed_spanners + walk_cost

        return h_value

