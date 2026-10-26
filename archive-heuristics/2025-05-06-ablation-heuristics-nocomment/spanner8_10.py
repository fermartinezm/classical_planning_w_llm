from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    return fact[1:-1].split()

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-calculating static information.

        Summary:
            This heuristic estimates the cost to reach a goal state by summing
            the estimated number of tighten actions, the estimated travel cost
            to the first required location (closest loose nut), and the
            estimated number of spanner pickup actions needed.

        Assumptions:
            - There is exactly one man object in the domain.
            - Nuts do not move from their initial locations.
            - Spanners do not move unless carried by the man.
            - Link predicates define a bidirectional graph.
            - Solvable instances provide enough usable spanners in total to tighten all goal nuts.
            - Object names follow a convention (e.g., 'nut' for nuts, 'spanner' for spanners, a distinct name for the man).

        Heuristic Initialization:
            - Parses static facts to build the location graph based on 'link' predicates.
            - Identifies all locations mentioned in static links, initial state, and goals.
            - Computes all-pairs shortest path distances between all identified locations using BFS.
            - Identifies the man's name, all spanner names, and all nut names from the initial state.
            - Stores the static location for each nut from the initial state.
            - Identifies the set of goal nuts from the task goals.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # 1. Build location graph and compute distances
        self.location_graph = {}
        all_locations = set()

        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                l1, l2 = parts[1], parts[2]
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1)
                all_locations.add(l1)
                all_locations.add(l2)

        # Add locations from initial state and goals
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == "at":
                all_locations.add(parts[2])

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                 all_locations.add(parts[2])
            # Nut locations from 'tightened' goals are static and added from initial state

        self.distances = {}
        # Ensure all_locations are in the graph dictionary keys, even if isolated
        for loc in all_locations:
             self.location_graph.setdefault(loc, set())

        for start_loc in all_locations:
            self.distances[start_loc] = self._bfs(start_loc, all_locations)

        # 2. Identify objects and their static properties
        self.man_name = None
        self.all_spanners = set()
        self.all_nuts = set()
        self.nut_locations = {} # Nut locations are static

        # Find man name (assuming the first object in an 'at' fact that is not a nut/spanner)
        # This relies on domain structure and naming conventions from examples.
        # A more robust way would involve parsing types from the domain file.
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj = parts[1]
                # Assuming nuts start with 'nut', spanners with 'spanner'
                if not obj.startswith('nut') and not obj.startswith('spanner'):
                    self.man_name = obj # Found the man (assuming only one)
                # Collect all objects by assumed type prefix
                if obj.startswith('nut'):
                    self.all_nuts.add(obj)
                    self.nut_locations[obj] = parts[2] # Store static nut location
                elif obj.startswith('spanner'):
                    self.all_spanners.add(obj)

        # Collect goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened":
                self.goal_nuts.add(parts[1])

    def _bfs(self, start_node, all_nodes):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in all_nodes}
        distances[start_node] = 0
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()

            # Check if current_node is in the graph keys (might be an isolated location)
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Step-By-Step Thinking for Computing Heuristic:
            1. Identify the set of goal nuts that are currently 'loose'. If this set is empty, the goal is reached, and the heuristic is 0.
            2. Calculate the base heuristic value as the number of loose goal nuts. This represents the minimum number of 'tighten_nut' actions required.
            3. Find the man's current location and check if he is carrying a usable spanner.
            4. Estimate the travel cost: Find the minimum distance from the man's current location to any location containing a loose goal nut. Add this distance to the heuristic. If no loose goal nuts exist (already handled), or if nuts are unreachable, this term is 0 or results in a large value.
            5. Estimate the spanner acquisition cost: Determine how many spanner pickup actions are needed. This is equal to the number of loose goal nuts, minus one if the man is already carrying a usable spanner (since he has the first spanner needed). Add this number to the heuristic.
            6. Check for unsolvability: If there are loose goal nuts but the total number of usable spanners available in the state (carried by man + at locations) is less than the number of loose goal nuts, the problem is likely unsolvable from this state. Return a large heuristic value in this case. Also, if the closest nut location is unreachable, return a large value.
            7. The final heuristic value is the sum of the base cost (tighten actions), estimated travel, and estimated pickup actions.
        """
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nuts = {nut for nut in self.goal_nuts if f'(loose {nut})' in state}
        K = len(loose_goal_nuts)

        if K == 0:
            return 0 # Goal reached

        h = K # Base cost: number of tighten actions

        # Find man's current location and spanner status
        man_loc = None
        man_carrying_spanner = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] == self.man_name:
                man_loc = parts[2]
            elif parts[0] == "carrying" and parts[1] == self.man_name:
                man_carrying_spanner = parts[2]

        # Ensure man_loc was found (should always be the case in a valid state)
        if man_loc is None:
             # This indicates an invalid state representation or task setup
             return 1000000 # Large value indicating unsolvability/invalid state

        # 6. Check for unsolvability: Not enough usable spanners available in total.
        total_usable_spanners_in_state = sum(1 for s in self.all_spanners if f'(usable {s})' in state)
        if man_carrying_spanner is not None and f'(usable {man_carrying_spanner})' in state:
             # If man is carrying a usable spanner, it's already counted in all_spanners if (usable S) is true.
             # No need to add separately.
             pass
        # Check if total usable spanners (at locations + carried) is less than nuts to tighten
        # Note: The sum above counts all spanners with (usable S) true, regardless of location or if carried.
        # This is the correct total count of usable spanner "units" available.
        if K > total_usable_spanners_in_state:
             # Not enough spanners available to tighten all nuts
             return 1000000 # Large value indicating unsolvability

        # 4. Estimate travel cost: Distance to the closest loose nut location.
        min_dist_to_nut_loc = float('inf')
        for nut in loose_goal_nuts:
            nut_loc = self.nut_locations[nut]
            # Check if man_loc and nut_loc are in the calculated distances
            if man_loc in self.distances and nut_loc in self.distances[man_loc]:
                min_dist_to_nut_loc = min(min_dist_to_nut_loc, self.distances[man_loc][nut_loc])
            # else: nut_loc is unreachable from man_loc, distance remains inf

        if min_dist_to_nut_loc == float('inf'):
            # Loose nuts exist but are unreachable
            return 1000000 # Large value indicating unsolvability

        h += min_dist_to_nut_loc

        # 5. Estimate spanner acquisition cost: Number of pickup actions needed.
        man_has_usable_spanner = man_carrying_spanner is not None and f'(usable {man_carrying_spanner})' in state

        pickups_needed = K # Need K spanners in total
        if man_has_usable_spanner:
            pickups_needed -= 1 # Already have the first one

        h += max(0, pickups_needed) # Add the number of pickup actions

        # 7. Return the total heuristic value
        return h
