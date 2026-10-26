import heapq
from collections import deque

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a list of strings."""
    # Remove surrounding brackets and split by spaces
    # Handles potential quotes around object names if they existed, though not in example
    parts = fact_string.strip("()'").split()
    return parts


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
    Estimates the cost to reach the goal by summing:
    1. The number of loose goal nuts (representing the tighten actions needed).
    2. The shortest path distance from the man's current location to the
       nearest location containing a loose goal nut (representing initial travel).
    3. The shortest path distance from the man's current location to the
       nearest location containing a usable spanner, plus the pickup cost (1),
       if the man is not currently carrying a usable spanner.

    Assumptions:
    - There is exactly one man object.
    - Nut locations are static and are present in the initial state 'at' facts.
    - The location graph defined by 'link' facts is undirected.
    - All locations, nuts, spanners, and the man are reachable from each other
      if a solution exists. The heuristic returns infinity if required items/locations are unreachable.
    - The goal is always to tighten a specific set of nuts.
    - Object types (man, spanner, nut, location) can be inferred from predicate usage.

    Heuristic Initialization:
    - Parses static facts to build the location graph.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the man object name, spanner names, nut names, and location names
      by inspecting facts in the initial state and static information.
    - Stores the static location of each nut.
    - Stores the set of goal nut names.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state and the set of goal nuts.
    2. Identify all goal nuts that are currently loose in the state. If none are loose, the heuristic is 0 (goal state).
    3. Initialize the heuristic value `h` with the number of loose goal nuts. This accounts for the 'tighten_nut' action needed for each.
    4. Find the man's current location in the state.
    5. Find the locations of all loose goal nuts using the precomputed static locations.
    6. Calculate the shortest distance from the man's current location to the nearest loose goal nut location using the precomputed distances. Add this distance to `h`. If no loose goal nuts are reachable, return infinity.
    7. Check if the man is currently carrying a usable spanner by inspecting the state facts.
    8. If the man is NOT carrying a usable spanner:
        a. Find all usable spanners that are currently at some location (not carried by the man) by inspecting the state facts.
        b. If there are no such spanners available anywhere, return infinity (problem unsolvable).
        c. Find the shortest distance from the man's current location to the nearest location containing an available usable spanner using the precomputed distances.
        d. If no available spanners are reachable, return infinity.
        e. Add this distance plus 1 (for the pickup action) to `h`.
    9. Return the final heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        # Extract goal nut names
        self.goal_nuts = {self.parse_fact(g)[1] for g in task.goals}

        # Data structures to store object names by type and nut locations
        self.man_name = None
        self.spanner_names = set()
        self.nut_names = set()
        self.location_names = set()
        self.nut_locations = {} # Maps nut name to its static location

        # Build location graph
        self.location_graph = {} # Adjacency list {loc: [neighbor1, neighbor2]}

        # Collect all facts that might mention objects or links
        all_relevant_facts = set(task.initial_state) | set(task.static) | set(task.goals)

        # First pass to infer object types and collect locations/links
        for fact_string in all_relevant_facts:
            parts = self.parse_fact(fact_string)
            predicate = parts[0]

            if predicate == 'link':
                l1, l2 = parts[1], parts[2]
                self.location_names.add(l1)
                self.location_names.add(l2)
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1) # Links are bidirectional

            elif predicate == 'carrying':
                 man, spanner = parts[1], parts[2]
                 self.man_name = man # Assume the carrier is the man
                 self.spanner_names.add(spanner)

            elif predicate == 'usable':
                 spanner = parts[1]
                 self.spanner_names.add(spanner)

            elif predicate == 'loose' or predicate == 'tightened':
                 nut = parts[1]
                 self.nut_names.add(nut)

            elif predicate == 'at':
                 obj, loc = parts[1], parts[2]
                 self.location_names.add(loc)
                 # Object type inference for 'at' facts is less direct, rely on other predicates first

        # Second pass to get static nut locations from initial state 'at' facts
        for fact_string in task.initial_state:
             parts = self.parse_fact(fact_string)
             if parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 if obj in self.nut_names:
                     self.nut_locations[obj] = loc
                 # Man and spanner initial locations are dynamic, found in __call__

        # Ensure all locations mentioned are in the graph keys, even if they have no links
        for loc in self.location_names:
             self.location_graph.setdefault(loc, [])

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.location_names:
            self.distances[start_loc] = self._bfs(start_loc)

    def parse_fact(self, fact_string):
        """Parses a PDDL fact string into a list of strings."""
        # Remove surrounding brackets and split by spaces
        # Handles potential quotes around object names if they existed, though not in example
        parts = fact_string.strip("()'").split()
        return parts

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.location_names}
        if start_node not in self.location_names:
             # Start node is not a known location, cannot compute distances
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Ensure current_node is a valid key in the graph (should be if in location_names)
            if current_node not in self.location_graph:
                 continue

            for neighbor in self.location_graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # This implies loc1 or loc2 is not a known location, or unreachable
             return float('inf')
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for a given state.
        """
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nuts = {n for n in self.goal_nuts if f"'(loose {n})'" in state}

        # If all goal nuts are tightened, goal reached
        if not loose_goal_nuts:
            return 0

        h = 0

        # Add cost for tighten actions (1 action per loose goal nut)
        h += len(loose_goal_nuts)

        # Get man's current location
        man_loc = None
        for fact in state:
            if fact.startswith(f"'(at {self.man_name} "):
                man_loc = self.parse_fact(fact)[2]
                break

        if man_loc is None:
             # Man's location is unknown, indicates an invalid state representation
             return float('inf')

        # Add cost to reach the nearest loose nut location
        target_nut_locations = {self.nut_locations[n] for n in loose_goal_nuts if n in self.nut_locations}
        if not target_nut_locations:
             # Should not happen if loose_goal_nuts is not empty and nut_locations are correctly initialized
             # If a goal nut's location isn't known statically, it's an issue with problem setup.
             # Returning inf seems appropriate as we can't estimate cost to reach it.
             return float('inf')

        min_dist_to_nut = float('inf')
        for nut_loc in target_nut_locations:
            dist = self.get_distance(man_loc, nut_loc)
            if dist != float('inf'):
                min_dist_to_nut = min(min_dist_to_nut, dist)

        if min_dist_to_nut == float('inf'):
             # Nearest nut location is unreachable from man's current location
             return float('inf')
        h += min_dist_to_nut

        # Check if man is carrying a usable spanner
        man_has_usable_spanner = False
        # Assuming man carries at most one spanner
        for fact in state:
             if fact.startswith(f"'(carrying {self.man_name} "):
                 carried_spanner = self.parse_fact(fact)[2]
                 if f"'(usable {carried_spanner})'" in state:
                     man_has_usable_spanner = True
                 break

        # Add cost to acquire a spanner if needed
        if not man_has_usable_spanner:
            available_usable_spanners_locs = []
            for spanner_name in self.spanner_names:
                # Check if usable and not carried by the man
                is_usable = f"'(usable {spanner_name})'" in state
                is_carried = f"'(carrying {self.man_name} {spanner_name})'" in state
                if is_usable and not is_carried:
                    # Find its location in the current state
                    spanner_loc = None
                    for fact in state:
                        if fact.startswith(f"'(at {spanner_name} "):
                            spanner_loc = self.parse_fact(fact)[2]
                            break
                    if spanner_loc:
                        available_usable_spanners_locs.append(spanner_loc)

            if not available_usable_spanners_locs:
                # No usable spanners available anywhere in the state
                return float('inf')

            # Find nearest usable spanner location
            min_dist_to_spanner = float('inf')
            for spanner_loc in available_usable_spanners_locs:
                dist = self.get_distance(man_loc, spanner_loc)
                if dist != float('inf'):
                    min_dist_to_spanner = min(min_dist_to_spanner, dist)

            if min_dist_to_spanner == float('inf'):
                 # Nearest spanner location is unreachable from man's current location
                 return float('inf')

            h += min_dist_to_spanner + 1 # Walk to nearest spanner + pickup action

        return h
