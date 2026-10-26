from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

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

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It calculates the minimum cost for each loose nut independently, considering the
    actions needed to bring a man and a usable spanner to the nut's location and
    perform the tightening. The total heuristic is the sum of these minimum costs
    for all loose nuts.

    # Assumptions
    - Each loose nut requires one unique usable spanner to be tightened.
    - The cost of moving between linked locations is 1.
    - Picking up a spanner costs 1.
    - Tightening a nut costs 1.
    - The shortest path between locations is used for travel cost.
    - The heuristic sums the minimum costs for each nut independently, which may
      overestimate the true cost by not fully accounting for shared travel paths
      or resource contention (beyond spanner uniqueness).

    # Heuristic Initialization
    - Extracts all location objects and the links between them from static facts.
    - Builds a graph of locations and precomputes the shortest path distances
      between all pairs of locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all loose nuts and their current locations.
    2. If there are no loose nuts, the heuristic is 0 (goal state).
    3. Identify all usable spanners and their current status (at a location or carried by a man).
    4. Identify all men and their current locations.
    5. If there are loose nuts but no usable spanners or no men, the problem is likely unsolvable; return a large value (infinity).
    6. For each loose nut `n` at location `l_n`:
       a. Calculate the minimum cost to get *any* man `m` with *any* usable spanner `s` to location `l_n`. This minimum cost, `MinApproachCost(l_n)`, is determined as follows:
          - Consider usable spanners `s` currently carried by men `m` at location `l_m`. The cost for this man `m` to reach `l_n` is `dist(l_m, l_n)`.
          - Consider usable spanners `s` currently at location `l_s`. A man `m` at location `l_m` must travel to `l_s`, pick up the spanner (cost 1), and then travel to `l_n`. The cost for this man `m` and spanner `s` is `dist(l_m, l_s) + 1 + dist(l_s, l_n)`.
          - `MinApproachCost(l_n)` is the minimum of all these costs over all available men and usable spanners.
       b. The estimated cost to tighten this specific nut is `MinApproachCost(l_n) + 1` (for the `tighten_nut` action itself).
    7. The total heuristic value is the sum of the estimated costs for each loose nut.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and computing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        self.locations = set()
        self.graph = {} # Adjacency list {loc: [neighbor1, neighbor2, ...]}

        # Build the location graph from static link facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # Compute all-pairs shortest paths (distances)
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find distances to all reachable nodes.
        Returns a dictionary {node: distance}.
        """
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in self.graph: # Ensure node exists in graph keys
                for neighbor in self.graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def dist(self, loc1, loc2):
        """
        Returns the precomputed shortest distance between two locations.
        Returns float('inf') if locations are not connected or not found.
        """
        if loc1 in self.distances and loc2 in self.distances[loc1]:
             return self.distances[loc1][loc2]
        return float('inf') # Should not happen in connected graphs for solvable problems

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to tighten all loose nuts.
        """
        state = node.state

        # 1. Identify loose nuts and their locations
        loose_nuts_locs = {} # {nut_name: location_name}
        # Find all nuts first to distinguish them from men/spanners later
        all_nuts = set()
        for fact in state:
             parts = get_parts(fact)
             if parts[0] in ["loose", "tightened"] and len(parts) == 2:
                  all_nuts.add(parts[1])

        for nut_name in all_nuts:
             # Check if it's loose
             if f"(loose {nut_name})" in state:
                 # Find location of this loose nut
                 for loc_fact in state:
                     loc_parts = get_parts(loc_fact)
                     if loc_parts[0] == "at" and len(loc_parts) == 3 and loc_parts[1] == nut_name:
                         loose_nuts_locs[nut_name] = loc_parts[2]
                         break # Found location for this nut


        # If there are no loose nuts, the goal is met (assuming goal is only tightening nuts)
        if not loose_nuts_locs:
             return 0

        # 2. Identify usable spanners and their status
        usable_spanners_at_locs = {} # {spanner_name: location_name}
        usable_spanners_carried = {} # {spanner_name: man_name}
        usable_spanners = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "usable":
                spanner_name = parts[1]
                usable_spanners.add(spanner_name)

        # Find locations/carriers for usable spanners
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3 and parts[1] in usable_spanners:
                  usable_spanners_at_locs[parts[1]] = parts[2]
             elif parts[0] == "carrying" and len(parts) == 3 and parts[2] in usable_spanners:
                  usable_spanners_carried[parts[2]] = parts[1] # spanner_name: man_name


        # 3. Identify men and their locations
        man_locs = {} # {man_name: location_name}
        all_locatable_at_facts = {} # {obj_name: loc_name} for all 'at' facts
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3:
                  all_locatable_at_facts[parts[1]] = parts[2]

        # Men are objects in 'carrying' facts OR objects in 'at' facts that are not nuts or spanners.
        men_carrying = set(usable_spanners_carried.values()) # Men carrying usable spanners

        for obj_name, loc_name in all_locatable_at_facts.items():
             if obj_name not in all_nuts and obj_name not in usable_spanners: # Check against usable spanners
                  man_locs[obj_name] = loc_name
             elif obj_name in men_carrying:
                  man_locs[obj_name] = loc_name # Add men who are carrying

        # Prepare usable spanners carried with their locations
        usable_spanners_carried_by_man_locs = [] # List of (man_name, spanner_name, man_location) tuples
        for s_name, m_name in usable_spanners_carried.items():
             if m_name in man_locs:
                  usable_spanners_carried_by_man_locs.append((m_name, s_name, man_locs[m_name]))


        # If there are loose nuts but no usable spanners or no men, return infinity
        if loose_nuts_locs and (not usable_spanners or not man_locs): # Check if man_locs is empty
             return float('inf') # Problem likely unsolvable

        total_heuristic = 0

        # 6. For each loose nut n at location l_n:
        for nut_name, l_n in loose_nuts_locs.items():
            min_approach_cost = float('inf')

            # Option 1: Use a usable spanner already carried by a man
            # Iterate through (man_name, spanner_name, man_location) for carried usable spanners
            for (m_name, s_name, l_m) in usable_spanners_carried_by_man_locs:
                 travel_cost = self.dist(l_m, l_n)
                 if travel_cost != float('inf'):
                     cost = travel_cost # Cost to get man+spanner to nut location (walks)
                     min_approach_cost = min(min_approach_cost, cost)

            # Option 2: Use a usable spanner at a location, picked up by a man
            # Iterate through (spanner_name, spanner_location) for usable spanners at locations
            for s_name, l_s in usable_spanners_at_locs.items():
                 # Iterate through (man_name, man_location) for all men
                 for m_name, l_m in man_locs.items():
                     travel_cost_m_to_s = self.dist(l_m, l_s)
                     travel_cost_s_to_n = self.dist(l_s, l_n) # This is distance for man carrying spanner
                     if travel_cost_m_to_s != float('inf') and travel_cost_s_to_n != float('inf'):
                         cost = travel_cost_m_to_s + 1 + travel_cost_s_to_n # Walk to spanner + pickup + walk to nut
                         min_approach_cost = min(min_approach_cost, cost)

            # If no way to get a man with a usable spanner to this nut, problem is unsolvable
            if min_approach_cost == float('inf'):
                 return float('inf') # One nut cannot be serviced

            # Add cost for this nut: min approach cost + tighten action
            total_heuristic += min_approach_cost + 1

        return total_heuristic
