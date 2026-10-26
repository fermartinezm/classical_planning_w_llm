import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, and if each part matches the corresponding arg pattern
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Estimates the cost to reach the goal state (all goal nuts tightened)
    by considering the actions required for the man to collect usable spanners
    and visit the locations of all loose goal nuts.

    Summary:
    The heuristic estimates the remaining cost by summing the minimum required
    tighten actions, pickup actions for spanners needed from locations,
    the travel cost to reach the first necessary location (either a loose nut
    location or a location with an available usable spanner if spanners are needed),
    and a proxy for subsequent travel between the remaining necessary unique locations.

    Assumptions:
    - Links between locations are bidirectional.
    - There is exactly one object of type 'man'. The heuristic attempts to find
      its name by looking for the first argument of any '(carrying ?m ?s)' fact
      in the initial state; if none is found, it defaults to the name 'bob'.
    - All goal nuts have an initial location specified by an '(at ?n ?l)' fact
      in the initial state. Nuts do not move.
    - The set of locations derived from 'link' facts and initial 'at' facts
      for locatable objects covers all locations relevant to the problem.
    - The graph of locations is connected such that all necessary locations
      (loose nut locations and available spanner locations if needed) are
      reachable from the man's initial location in solvable problems.

    Heuristic Initialization:
    During initialization, the heuristic performs the following steps:
    1. Identifies all unique locations mentioned in 'link' facts and initial 'at' facts.
    2. Builds an adjacency list representation of the location graph based on 'link' facts.
    3. Computes all-pairs shortest path distances between all identified locations using Breadth-First Search (BFS).
    4. Identifies all goal nuts from the task's goal state.
    5. Finds the initial location for each goal nut by searching the initial state facts.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the man's current location by searching for the '(at man_name ?l)' fact. If the man's location is not found and the goal is not reached, return infinity.
    2. Identify all spanners that are currently marked as '(usable ?s)'.
    3. Identify all spanners the man is currently '(carrying man_name ?s)'.
    4. Determine which usable spanners the man is carrying. Count them (`num_carried_usable`).
    5. Determine which usable spanners are at specific locations (not carried). Store their locations (`available_usable_spanners_locs`). Count them (`num_available_usable`).
    6. Identify which goal nuts are currently '(loose ?n)'. These are the `loose_goal_nuts`.
    7. If there are no `loose_goal_nuts`, the goal is reached, return 0.
    8. Calculate the total number of usable spanners available (`total_usable_spanners = num_carried_usable + num_available_usable`).
    9. Calculate the number of loose goal nuts (`num_loose_goal_nuts`).
    10. If `num_loose_goal_nuts` is greater than `total_usable_spanners`, the problem is unsolvable from this state, return infinity.
    11. Calculate the number of additional usable spanners the man needs to pick up from locations: `needed_spanners_from_locations = max(0, num_loose_goal_nuts - num_carried_usable)`.
    12. Initialize the heuristic value: `h = num_loose_goal_nuts` (cost for tighten actions) + `needed_spanners_from_locations` (cost for pickup actions).
    13. Identify the set of unique locations that the man must visit (`locations_to_visit`). This set includes the initial locations of all `loose_goal_nuts`. If `needed_spanners_from_locations > 0`, it also includes the locations where available usable spanners were found.
    14. Calculate the shortest path distance from the man's current location (`man_loc`) to the nearest location in `locations_to_visit`. Add this distance to `h`. If no locations are in `locations_to_visit` (should not happen if `loose_goal_nuts` is not empty) or if they are unreachable, return infinity.
    15. Add a proxy cost for the travel required to visit the remaining unique locations in `locations_to_visit`. This is estimated as 1 unit of cost for each unique location in `locations_to_visit` after the first one visited: `max(0, len(locations_to_visit) - 1)`. Add this to `h`.
    16. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Args:
            task: The planning task object.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all locations and nut locations

        # Find the man's name
        self.man_name = 'bob' # Default assumption based on examples
        # Look for the man in the initial state by checking 'carrying' facts
        for fact in initial_state:
             if match(fact, "carrying", "*", "*"):
                  self.man_name = get_parts(fact)[1]
                  break # Found the man's name

        # Extract all locations
        self.locations = set()
        # Locations can appear in 'at' facts (initial state) or 'link' facts (static)
        all_relevant_facts = set(initial_state) | set(static_facts)
        for fact in all_relevant_facts:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                self.locations.add(parts[1])
                self.locations.add(parts[2])
            elif parts[0] == 'at' and len(parts) == 3:
                 # The second argument of 'at' is a location
                 self.locations.add(parts[2])

        # Build adjacency list for locations based on 'link' facts
        self.adj = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.adj[l1].append(l2)
                self.adj[l2].append(l1) # Links are bidirectional

        # Compute shortest path distances between all pairs of locations using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Identify goal nuts and their initial locations (nuts don't move)
        self.goal_nuts = set()
        self.nut_locations = {} # Map nut -> initial location
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                self.goal_nuts.add(nut)

        # Find initial locations of all goal nuts from the initial state
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.goal_nuts:
                     self.nut_locations[obj] = loc

        # Note: If a goal nut's initial location is not found, the heuristic might fail later.
        # Assuming valid PDDL instances where goal nuts have initial locations.


    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not a known location, cannot compute distances
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()
            # Use .get(current_node, []) to handle nodes with no defined links safely
            for neighbor in self.adj.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Returns the shortest path distance between two locations."""
        # Check if both locations are in our distance map and loc2 is reachable from loc1
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the state.

        Returns:
            An integer estimating the remaining cost to the goal, or float('inf') if unsolvable.
        """
        state = node.state

        # 1. Identify state elements
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                 man_loc = get_parts(fact)[2]
                 break

        # If man is not located anywhere, check if goal is reached. Otherwise, unsolvable.
        if man_loc is None:
             return 0 if self.goals <= state else float('inf')

        carried_spanners = set()
        for fact in state:
             if match(fact, "carrying", self.man_name, "*"):
                  spanner = get_parts(fact)[2]
                  carried_spanners.add(spanner)

        all_usable_spanners = set()
        for fact in state:
             if match(fact, "usable", "*"):
                 spanner = get_parts(fact)[1]
                 all_usable_spanners.add(spanner)

        carried_usable_spanners = carried_spanners.intersection(all_usable_spanners)
        num_carried_usable = len(carried_usable_spanners)

        available_usable_spanners_locs = collections.defaultdict(list)
        for spanner in all_usable_spanners - carried_usable_spanners:
             for fact in state:
                 if match(fact, "at", spanner, "*"):
                     loc = get_parts(fact)[2]
                     available_usable_spanners_locs[loc].append(spanner)
                     break # Found location for this spanner

        loose_goal_nuts = set()
        for nut in self.goal_nuts:
            if f'(loose {nut})' in state:
                 loose_goal_nuts.add(nut)

        # 7. If no loose goal nuts, goal is reached.
        if not loose_goal_nuts:
            return 0

        # 8. Check solvability based on usable spanners
        num_available_usable = sum(len(spanners) for spanners in available_usable_spanners_locs.values())
        total_usable_spanners = num_carried_usable + num_available_usable
        num_loose_goal_nuts = len(loose_goal_nuts)

        if num_loose_goal_nuts > total_usable_spanners:
             return float('inf')

        # 11. Calculate needed spanners from locations
        needed_spanners_from_locations = max(0, num_loose_goal_nuts - num_carried_usable)

        # 12. Calculate base cost (tighten + pickup)
        heuristic_value = num_loose_goal_nuts + needed_spanners_from_locations

        # 13. Identify locations to visit
        loose_nut_locations = {self.nut_locations[nut] for nut in loose_goal_nuts}
        locations_to_visit = set(loose_nut_locations)
        if needed_spanners_from_locations > 0:
             locations_to_visit.update(available_usable_spanners_locs.keys())

        # 14. Calculate travel cost to first necessary location
        min_dist_to_first_visit_loc = float('inf')
        if locations_to_visit:
             # Ensure man_loc is a valid start node in our distance map
             if man_loc in self.distances:
                 min_dist_to_first_visit_loc = min(self.get_distance(man_loc, loc) for loc in locations_to_visit)
             # else: man_loc is not a known location, min_dist remains inf

        if min_dist_to_first_visit_loc == float('inf'):
             # This implies necessary locations are unreachable from man_loc.
             # Given the total_usable_spanners check passed, this means nuts/spanners are unreachable.
             return float('inf')

        heuristic_value += min_dist_to_first_visit_loc

        # 15. Add proxy for subsequent travel
        heuristic_value += max(0, len(locations_to_visit) - 1)

        # 16. Return final heuristic value
        return heuristic_value
