from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # Uncomment this line in the actual environment

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError
        def get_name(self):
            return self.__class__.__name__


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
    # Ensure we don't try to match more args than parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts specified in the goal. It considers the costs of movement,
    spanner acquisition, and the tightening actions themselves.

    # Assumptions:
    - The man can only carry one spanner at a time.
    - Each tightening action consumes the usability of one spanner.
    - The graph of locations connected by 'link' predicates is static.
    - All relevant locations (man, spanners, nuts) are part of the connected graph.
    - The problem is solvable (enough usable spanners exist and locations are reachable).

    # Heuristic Initialization
    - Precomputes all-pairs shortest paths between locations based on 'link' facts.
    - Identifies all objects (men, nuts, spanners) present in the initial state.
    - Stores goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the total cost as the sum of:
    1.  The number of 'tighten_nut' actions required. This is equal to the number
        of loose nuts that are specified as 'tightened' in the goal.
    2.  The movement cost for the man to reach the location of the *first*
        loose goal nut. This is estimated as the minimum shortest path distance
        from the man's current location to any location containing a loose goal nut.
    3.  The cost to acquire the *first* usable spanner and bring it to a nut location.
        -   If the man is already carrying a usable spanner, this cost is 0.
        -   If not, the cost is estimated as the minimum shortest path distance
            from the man's current location to any location with a usable spanner
            on the ground, plus 1 for the 'pickup_spanner' action, plus the minimum
            shortest path distance from any usable spanner location on the ground
            to any loose goal nut location.
    4.  A simplified cost for tightening the *subsequent* nuts (all but the first).
        For each additional loose goal nut (after the first one has been tightened),
        the man needs to acquire another usable spanner and perform the tighten action.
        This is estimated as a fixed cost of 3 actions per subsequent nut (simplified
        estimate for pickup + travel + tighten).

    The total heuristic value is the sum of these components. If there are no
    loose nuts in the goal, the heuristic is 0. If any required distance is
    infinite (locations are disconnected or resources unreachable), the heuristic
    is infinite.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Identify all objects and locations from initial state and static facts
        all_locatables = set()
        all_locations = set()

        # Process initial state facts to find objects and their initial locations
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                all_locatables.add(obj)
                all_locations.add(loc)
            # Also capture objects mentioned in other initial state facts like loose/usable
            elif len(parts) == 2: # e.g., (loose nut1), (usable spanner1)
                 all_locatables.add(parts[1])


        # Process static facts to find all locations and build the link graph
        self.links = {} # Adjacency list {loc: {neighbor1, neighbor2}}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                all_locations.add(l1)
                all_locations.add(l2)
                self.links.setdefault(l1, set()).add(l2)
                self.links.setdefault(l2, set()).add(l1)

        self.locations = all_locations

        # Infer types (assuming standard naming conventions or based on initial state predicates)
        self.all_nuts = {get_parts(fact)[1] for fact in task.initial_state if get_parts(fact)[0] == 'loose'}
        self.all_spanners = {get_parts(fact)[1] for fact in task.initial_state if get_parts(fact)[0] == 'usable'}
        # Assume anything else locatable is a man
        self.all_men = {obj for obj in all_locatables if obj not in self.all_nuts and obj not in self.all_spanners}


        # Precompute all-pairs shortest paths
        self.distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        distances = {}
        for start_node in self.locations:
            distances[start_node] = self._bfs(start_node)
        return distances

    def _bfs(self, start_node):
        """Performs BFS starting from a node to find distances to all reachable nodes."""
        dist = {node: float('inf') for node in self.locations}
        dist[start_node] = 0
        queue = [start_node]
        while queue:
            u = queue.pop(0)
            # Handle locations that might be isolated (no links)
            if u in self.links:
                for v in self.links[u]:
                    if dist[v] == float('inf'):
                        dist[v] = dist[u] + 1
                        queue.append(v)
        return dist

    def get_distance(self, loc1, loc2):
        """Returns the precomputed shortest distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # This handles cases where locations might not be in the graph or are disconnected.
             # In a solvable problem, relevant locations should be connected.
             return 0 if loc1 == loc2 else float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify man's current location (assuming only one man)
        man_loc = None
        # Find the single man object
        man_obj = next(iter(self.all_men), None) # Assumes there is at least one man
        if man_obj:
             man_loc = next((get_parts(fact)[2] for fact in state if match(fact, "at", man_obj, "*")), None)

        if man_loc is None:
             # This state is likely unreachable or invalid, return infinity
             return float('inf')

        # Identify loose nuts that are goals
        loose_goal_nuts = {
            nut for nut in self.all_nuts
            if f"(tightened {nut})" in self.goals and f"(loose {nut})" in state
        }
        N_loose_goal = len(loose_goal_nuts)

        # If all goal nuts are tightened, heuristic is 0
        if N_loose_goal == 0:
            return 0

        # Identify locations of loose goal nuts
        loose_nut_locations = set()
        for nut in loose_goal_nuts:
             nut_loc = next((get_parts(fact)[2] for fact in state if match(fact, "at", nut, "*")), None)
             if nut_loc:
                 loose_nut_locations.add(nut_loc)

        # Identify usable spanners on the ground and if man is carrying a usable one
        usable_spanners_on_ground_locs = {
            get_parts(fact)[2] for fact in state
            if match(fact, "at", "*", "*") and f"(usable {get_parts(fact)[1]})" in state
        }

        man_carrying_spanner = next((get_parts(fact)[2] for fact in state if match(fact, "carrying", man_obj, "*")), None)
        man_carrying_usable = (man_carrying_spanner is not None) and (f"(usable {man_carrying_spanner})" in state)


        # --- Heuristic Calculation ---

        # Base cost: one tighten action for each loose goal nut
        h = N_loose_goal

        # Cost 1: Movement to the first nut location
        min_dist_to_any_nut = float('inf')
        for nut_loc in loose_nut_locations:
             dist = self.get_distance(man_loc, nut_loc)
             if dist != float('inf'):
                 min_dist_to_any_nut = min(min_dist_to_any_nut, dist)

        # If any nut location is unreachable, the problem is unsolvable from here
        if min_dist_to_any_nut == float('inf'):
             return float('inf')
        h += min_dist_to_any_nut


        # Cost 2: Cost to get the first usable spanner if not already carrying one
        cost_first_spanner_acquisition = 0
        if not man_carrying_usable:
            min_dist_to_spanner_and_pickup = float('inf')
            # Need at least one usable spanner on the ground to pick up
            if usable_spanners_on_ground_locs:
                for ls in usable_spanners_on_ground_locs:
                    dist = self.get_distance(man_loc, ls)
                    if dist != float('inf'):
                        min_dist_to_spanner_and_pickup = min(min_dist_to_spanner_and_pickup, dist + 1) # +1 for pickup

            # If no usable spanners on ground and not carrying one, problem is unsolvable
            if min_dist_to_spanner_and_pickup == float('inf'):
                 return float('inf')

            min_dist_spanner_loc_to_nut_loc = float('inf')
            # Need paths from usable spanners on ground to loose goal nuts
            if usable_spanners_on_ground_locs and loose_nut_locations:
                for sloc in usable_spanners_on_ground_locs:
                    for nloc in loose_nut_locations:
                        dist = self.get_distance(sloc, nloc)
                        if dist != float('inf'):
                             min_dist_spanner_loc_to_nut_loc = min(min_dist_spanner_loc_to_nut_loc, dist)

            # If no path from any usable spanner on ground to any nut, problem is unsolvable
            if min_dist_spanner_loc_to_nut_loc == float('inf'):
                 return float('inf')

            cost_first_spanner_acquisition = min_dist_to_spanner_and_pickup + min_dist_spanner_loc_to_nut_loc

        h += cost_first_spanner_acquisition

        # Cost 3: Simplified cost for subsequent nuts (pickup + travel + tighten)
        # Each subsequent nut requires getting a new spanner and tightening.
        # We need N_loose_goal spanners in total.
        # If man starts carrying usable, 1 is covered. Needs N_loose_goal - 1 more.
        # If man does not start carrying usable, 0 is covered. Needs N_loose_goal more.
        # The first spanner acquisition cost is handled above.
        # The remaining spanners needed = N_loose_goal - (1 if man_carrying_usable else 0)
        # For each of these, we need at least a pickup (1) and a tighten (1).
        # Add a minimal travel cost (1) between locations. Total 3 per subsequent nut.
        subsequent_nuts_count = max(0, N_loose_goal - 1)
        h += subsequent_nuts_count * 3

        # If any component resulted in infinity, the total is infinity
        if h == float('inf'):
             return float('inf')

        return h
