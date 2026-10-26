from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def bfs(graph, start_node):
    """
    Performs a Breadth-First Search to find shortest distances from a start node
    in an unweighted graph.

    Args:
        graph: Adjacency list representation of the graph (dict: node -> set of neighbors).
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its distance from the start_node.
        Unreachable nodes will have a distance of float('inf').
    """
    # Initialize distances for all nodes present as keys in the graph
    distances = {node: float('inf') for node in graph}
    # Add start_node if it's not already a key (e.g., an isolated node mentioned in 'at')
    if start_node not in distances:
        distances[start_node] = float('inf')

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph dict
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Use .get for safety if neighbor is not a key (shouldn't happen if graph is built correctly)
                if distances.get(neighbor, float('inf')) == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It assumes a greedy strategy where the man repeatedly acquires the closest usable spanner,
    travels to the closest loose goal nut, and tightens it.

    # Assumptions
    - Nuts do not move from their initial locations.
    - Spanners become unusable after one tightening action.
    - The man can only carry one spanner at a time.
    - There is exactly one man.
    - Objects involved are the man, nuts, and spanners, and their types can be inferred from initial state predicates.
    - All locations relevant to the problem (initial locations of objects, goal locations of nuts, linked locations) are considered in the graph.
    - The graph of locations is undirected (links are bidirectional).

    # Heuristic Initialization
    - Builds a graph of locations based on `link` predicates and object initial/goal locations.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies goal nuts from the task goals.
    - Identifies the man's name, initial spanner locations, and initial usable spanners from the initial state.
    - Stores the static locations of nuts (assumed to be their initial locations).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic simulates a greedy process to tighten each required nut:

    1.  Identify the set of loose goal nuts in the current state. If this set is empty, the heuristic value is 0 (goal reached).
    2.  Identify the man's current location. If the man's location is unknown, return infinity.
    3.  Determine if the man is currently carrying a usable spanner.
    4.  Identify the set of usable spanners currently on the ground and their locations.
    5.  Calculate the total number of usable spanners available (on the ground plus the one potentially carried by the man).
    6.  If the number of loose goal nuts is greater than the total number of usable spanners, the problem is unsolvable from this state, and the heuristic returns infinity.
    7.  Initialize the heuristic cost to 0.
    8.  Create working copies of the set of loose goal nuts and the dictionary of usable spanners on the ground.
    9.  Loop while there are still loose goal nuts to tighten:
        a.  **Acquire Spanner:** If the man is not currently carrying a usable spanner:
            -   If there are no usable spanners left on the ground, return infinity (should be caught earlier, but as a safeguard).
            -   Find the usable spanner on the ground that is closest to the man's current location using the precomputed distances. If no reachable spanner exists, return infinity.
            -   Add the distance from the man's current location to the spanner's location to the heuristic cost (representing `walk` actions).
            -   Update the man's current location to the location of the chosen spanner.
            -   Remove the chosen spanner from the working set of available spanners on the ground.
            -   Add 1 to the heuristic cost for the `pickup_spanner` action.
            -   Mark the man as now carrying a usable spanner.
        b.  **Go to Nut and Tighten:** The man is now carrying a usable spanner.
            -   Find the loose goal nut that is closest to the man's current location using the precomputed distances. If no reachable nut exists, return infinity.
            -   Add the distance from the man's current location to the nut's location to the heuristic cost (representing `walk` actions).
            -   Update the man's current location to the location of the chosen nut.
            -   Remove the chosen nut from the working set of loose goal nuts.
            -   Add 1 to the heuristic cost for the `tighten_nut` action.
            -   Mark the man as *not* carrying a usable spanner (as the spanner becomes unusable after tightening).
    10. Return the total accumulated heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.initial_state = task.initial_state
        static_facts = task.static

        self.locations = set()
        self.graph = {}

        # 1. Build location graph from link facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional
                self.locations.add(loc1)
                self.locations.add(loc2)

        # 2. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "tightened":
                if len(parts) > 1:
                    self.goal_nuts.add(parts[1])

        # 3. Identify initial spanners and their usability/locations
        initial_spanners_set = set()
        self.initial_usable_spanners = set()

        # Find spanners mentioned as usable or carried in initial state
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == "usable":
                if len(parts) > 1:
                    spanner_name = parts[1]
                    initial_spanners_set.add(spanner_name)
                    self.initial_usable_spanners.add(spanner_name)
            elif parts and parts[0] == "carrying":
                 if len(parts) > 2:
                     spanner_name = parts[2]
                     initial_spanners_set.add(spanner_name)

        self.initial_spanner_locations = {}
        # Find initial locations for spanners and nuts from initial state
        self.nut_locations = {}
        self.man_name = None

        locatable_objects_at_start = set()
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == "at":
                 obj, loc = parts[1], parts[2]
                 locatable_objects_at_start.add(obj)
                 self.locations.add(loc) # Add all initial 'at' locations to graph nodes
                 self.graph.setdefault(loc, set()) # Ensure location exists in graph dict

                 if obj in self.goal_nuts: # Assume objects in goal_nuts are nuts
                     self.nut_locations[obj] = loc
                 elif obj in initial_spanners_set: # Assume objects in initial_spanners_set are spanners
                     self.initial_spanner_locations[obj] = loc

        # 4. Identify the man's name
        # The man is the locatable object that is not a nut or a spanner
        potential_men = locatable_objects_at_start - self.goal_nuts - initial_spanners_set

        if len(potential_men) == 1:
            self.man_name = potential_men.pop()
        # else: self.man_name remains None, handled in __call__


        # 5. Compute all-pairs shortest paths
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = bfs(self.graph, loc)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if man_name was successfully identified during initialization
        if self.man_name is None:
             # Cannot compute heuristic without knowing the man
             return float('inf')

        # 1. Identify current loose goal nuts
        current_loose_goals = {
            n for n in self.goal_nuts if f"(loose {n})" in state
        }

        if not current_loose_goals:
            return 0 # Goal reached for all nuts

        # 2. Identify man's current location
        man_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) > 1 and parts[1] == self.man_name:
                if len(parts) > 2:
                    man_location = parts[2]
                break
        if man_location is None:
             # Man's location must be known in a valid state
             return float('inf')

        # 3. Identify usable spanners available (on ground or carried)
        current_available_spanners_on_ground = {} # {spanner: location}
        man_carrying_usable = False

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at":
                if len(parts) > 2:
                    obj, loc = parts[1], parts[2]
                    # Check if this object is a spanner based on initial state info
                    # We assume any object initially identified as a spanner remains a spanner type
                    if obj in self.initial_spanner_locations or obj in self.initial_usable_spanners: # Check against known spanners
                         current_available_spanners_on_ground[obj] = loc
            elif parts and parts[0] == "carrying" and len(parts) > 1 and parts[1] == self.man_name:
                 if len(parts) > 2:
                     carried_spanner_name = parts[2]
                     # Check if the carried spanner is usable in the current state
                     if f"(usable {carried_spanner_name})" in state:
                         man_carrying_usable = True

        # Filter available spanners on ground to only include usable ones
        current_usable_spanners_on_ground = {s:l for s,l in current_available_spanners_on_ground.items() if f"(usable {s})" in state}

        # Total usable spanners available (on ground + carried)
        total_usable_spanners = len(current_usable_spanners_on_ground) + (1 if man_carrying_usable else 0)

        # 4. Check solvability based on spanners
        num_loose_goals = len(current_loose_goals)
        if total_usable_spanners < num_loose_goals:
            return float('inf') # Not enough usable spanners to tighten all nuts

        # 5. Compute greedy cost
        cost = 0
        current_lgn = set(current_loose_goals) # Copy to modify
        current_aus_on_ground = dict(current_usable_spanners_on_ground) # Copy to modify

        # Simulate the process
        while current_lgn:
            # a. Acquire Spanner (if needed)
            if not man_carrying_usable:
                if not current_aus_on_ground:
                     # This case should be caught by the total_usable_spanners check earlier,
                     # but this is a safeguard if state is inconsistent or logic flawed.
                     return float('inf')

                # Find closest usable spanner on ground
                closest_spanner = None
                min_dist_spanner = float('inf')
                spanner_loc = None

                # Check if man_location is a valid key in distances
                if man_location not in self.distances:
                     return float('inf') # Man is in an unmapped location?

                for spanner, loc in current_aus_on_ground.items():
                    # Check if spanner location is a valid key in distances from man_location
                    if loc in self.distances[man_location]:
                        dist = self.distances[man_location][loc]
                        if dist < min_dist_spanner:
                            min_dist_spanner = dist
                            closest_spanner = spanner
                            spanner_loc = loc

                if closest_spanner is None or min_dist_spanner == float('inf'):
                     # Cannot reach any usable spanner on the ground from man's current location
                     return float('inf')

                cost += min_dist_spanner # Walk to spanner
                man_location = spanner_loc # Man is now at spanner location
                del current_aus_on_ground[closest_spanner] # Spanner is picked up
                cost += 1 # Pickup action
                man_carrying_usable = True # Man is now carrying a usable spanner

            # b. Go to Nut and Tighten
            # Find closest loose goal nut
            closest_nut = None
            min_dist_nut = float('inf')
            nut_loc = None

            # Check if man_location is a valid key in distances
            if man_location not in self.distances:
                 return float('inf') # Man is in an unmapped location?

            for nut in current_lgn:
                loc = self.nut_locations.get(nut)
                # Check if nut location is known and reachable from man_location
                if loc is not None and man_location in self.distances and loc in self.distances[man_location]:
                    dist = self.distances[man_location][loc]
                    if dist < min_dist_nut:
                        min_dist_nut = dist
                        closest_nut = nut
                        nut_loc = loc

            if closest_nut is None or min_dist_nut == float('inf'):
                 # Cannot reach any loose goal nut from man's current location
                 return float('inf')

            cost += min_dist_nut # Walk to nut
            man_location = nut_loc # Man is now at nut location
            current_lgn.remove(closest_nut) # Nut is tightened
            cost += 1 # Tighten action
            man_carrying_usable = False # Spanner becomes unusable

        return cost
