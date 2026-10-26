from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It considers the number of nuts to tighten, the number of spanners to pick up,
    and the estimated travel cost for the man to perform these actions.

    # Assumptions:
    - There is exactly one man, assumed to be named 'bob' based on examples.
    - Spanners relevant to the problem are those marked as 'usable' in the initial state.
    - Nuts relevant to the problem are those specified in the goal as needing to be 'tightened'.
    - Links between locations are bidirectional.
    - Action costs are 1.

    # Heuristic Initialization
    - Build a graph of locations based on 'link' facts and compute all-pairs shortest paths (distances) using BFS.
    - Identify the names of goal nuts and initial usable spanners.
    - Identify the man's name (hardcoded as 'bob').

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify loose goal nuts in the current state. If none, the heuristic is 0.
    2. Identify the man's current location and if he is carrying a usable spanner.
    3. Identify usable spanners available at locations (not carried by the man).
    4. Calculate the number of 'tighten_nut' actions needed (equal to the number of loose goal nuts).
    5. Calculate the number of 'pickup_spanner' actions needed (equal to the number of loose goal nuts, minus 1 if the man starts carrying a usable spanner, minimum 0).
    6. Check for basic unsolvability: If the total number of usable spanners (carried + at locations) is less than the number of loose goal nuts, return infinity.
    7. Estimate travel cost:
       - The man needs to perform a sequence of actions: (potentially pickup) -> tighten -> (potentially pickup) -> tighten ...
       - This involves moving from the current location to the first action location, and then moving between subsequent action locations.
       - The first action location is either a spanner location (if a pickup is needed) or a nut location (if only one nut is left and the man is carrying a spanner).
       - Subsequent actions alternate between spanner locations (for pickup) and nut locations (for tighten).
       - Estimate travel as:
         - Distance from man's current location to the first action location.
         - Distance for each subsequent pickup-tighten cycle (travel from nut location to spanner location, then from spanner location to nut location), repeated for the number of pickups needed after the first. Use the closest available spanner location from the nut location for this cycle estimate.
    8. Sum the counts of tighten actions, pickup actions, and estimated travel cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and object names."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Identify goal nuts
        self.goal_nut_names = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'tightened'}

        # Identify initial usable spanners (relevant spanners for the problem)
        self.initial_usable_spanner_names = {get_parts(fact)[1] for fact in initial_state if get_parts(fact)[0] == 'usable'}

        # Identify the man (assuming 'bob' based on examples)
        # A more robust approach would parse object types from the domain file.
        self.man_name = 'bob' # Hardcoded based on examples

        # Build location graph and compute distances
        self.location_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Links are bidirectional
                locations.add(l1)
                locations.add(l2)

        self.distances = {}
        for start_loc in locations:
            self.distances[start_loc] = self._bfs_distances(start_loc, locations)

    def _bfs_distances(self, start_node, all_nodes):
        """Compute shortest path distances from start_node to all other nodes using BFS."""
        distances = {node: float('inf') for node in all_nodes}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        man_location = None
        carrying_usable_spanner = False
        loose_goal_nuts = []
        nut_locations = {} # Map nut name to location
        available_spanners = {} # Map spanner name to location if usable and not carried

        # Extract relevant information from the current state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj == self.man_name:
                    man_location = loc
                elif obj in self.goal_nut_names:
                     nut_locations[obj] = loc
                # Check if it's an initial usable spanner and still usable and not carried
                elif obj in self.initial_usable_spanner_names and '(usable ' + obj + ')' in state and '(carrying ' + self.man_name + ' ' + obj + ')' not in state:
                     available_spanners[obj] = loc
            elif parts[0] == 'carrying' and parts[1] == self.man_name:
                carried_spanner = parts[2]
                # Check if the carried spanner is one of the initial usable ones and is still usable
                if carried_spanner in self.initial_usable_spanner_names and '(usable ' + carried_spanner + ')' in state:
                    carrying_usable_spanner = True
            elif parts[0] == 'loose' and parts[1] in self.goal_nut_names:
                loose_goal_nuts.append(parts[1])

        # If all goal nuts are tightened, the heuristic is 0.
        if not loose_goal_nuts:
            return 0

        tightens_needed = len(loose_goal_nuts)
        # Number of pickups needed: one for each nut, unless already carrying one for the first nut.
        pickups_needed = max(0, tightens_needed - (1 if carrying_usable_spanner else 0))

        # Basic unsolvability check: Not enough usable spanners exist in the state.
        num_usable_spanners_in_state = len(available_spanners) + (1 if carrying_usable_spanner else 0)
        if num_usable_spanners_in_state < tightens_needed:
             return float('inf') # Problem likely unsolvable

        # Heuristic cost starts with the required actions (tighten + pickup)
        total_cost = tightens_needed + pickups_needed
        travel_cost = 0

        # Estimate travel cost
        if tightens_needed > 0:
            # Use the location of the first loose goal nut as a representative nut location
            # for estimating travel cycles. The specific nut doesn't matter as all nuts
            # need the same process.
            first_nut_loc = nut_locations.get(loose_goal_nuts[0])
            if first_nut_loc is None:
                 # This shouldn't happen if loose_goal_nuts is not empty, but handle defensively
                 return float('inf')

            if pickups_needed > 0:
                # The first action is likely a pickup. Find the closest available spanner location from the man.
                closest_spanner_loc_from_man = None
                min_dist_to_spanner_from_man = float('inf')
                for s_loc in available_spanners.values():
                    if man_location in self.distances and s_loc in self.distances[man_location]:
                        dist = self.distances[man_location][s_loc]
                        if dist < min_dist_to_spanner_from_man:
                            min_dist_to_spanner_from_man = dist
                            closest_spanner_loc_from_man = s_loc

                if closest_spanner_loc_from_man is None or min_dist_to_spanner_from_man == float('inf'):
                     return float('inf') # Cannot reach any spanner

                # Travel cost: Man's location -> Closest Spanner location (for first pickup)
                travel_cost += min_dist_to_spanner_from_man

                # Travel cost: Closest Spanner location -> First Nut location (after first pickup)
                if closest_spanner_loc_from_man in self.distances and first_nut_loc in self.distances[closest_spanner_loc_from_man]:
                    travel_cost += self.distances[closest_spanner_loc_from_man][first_nut_loc]
                else:
                     return float('inf') # Cannot reach the first nut location from the first spanner location

                # Travel cost for subsequent pickup-tighten cycles (if more than one pickup is needed)
                # After the first tighten, the man is at first_nut_loc. He needs (pickups_needed - 1) more cycles.
                # Each cycle involves: Travel Nut -> Spanner -> Nut.
                # Find the closest available spanner location from the nut location.
                closest_spanner_loc_from_nut = None
                min_dist_nut_spanner = float('inf')
                for s_loc in available_spanners.values():
                     if first_nut_loc in self.distances and s_loc in self.distances[first_nut_loc]:
                         dist = self.distances[first_nut_loc][s_loc]
                         if dist < min_dist_nut_spanner:
                             min_dist_nut_spanner = dist
                             closest_spanner_loc_from_nut = s_loc

                if closest_spanner_loc_from_nut is None or min_dist_nut_spanner == float('inf'):
                     return float('inf') # Cannot reach any spanner from the nut location

                # Travel cost: Nut location -> Closest Spanner location (for subsequent pickups)
                travel_cost += max(0, pickups_needed - 1) * min_dist_nut_spanner

                # Travel cost: Closest Spanner location -> Nut location (for subsequent tightens)
                if closest_spanner_loc_from_nut in self.distances and first_nut_loc in self.distances[closest_spanner_loc_from_nut]:
                     dist_spanner_nut = self.distances[closest_spanner_loc_from_nut][first_nut_loc]
                     travel_cost += max(0, pickups_needed - 1) * dist_spanner_nut
                else:
                     return float('inf') # Cannot reach the nut location from the spanner location

            elif tightens_needed > 0: # pickups_needed is 0, implies tightens_needed is 1 and carrying spanner
                # Go directly to the nut location to tighten the single nut.
                if man_location in self.distances and first_nut_loc in self.distances[man_location]:
                     travel_cost += self.distances[man_location][first_nut_loc]
                else:
                     return float('inf') # Cannot reach the nut location

        # Total heuristic cost = action costs + estimated travel cost
        return total_cost + travel_cost

