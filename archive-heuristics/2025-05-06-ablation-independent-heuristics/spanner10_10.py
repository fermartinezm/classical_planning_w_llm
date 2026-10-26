import math
import collections
from heuristics.heuristic_base import Heuristic

class spannerHeuristic(Heuristic):
    """
    Summary:
    This heuristic estimates the cost to reach the goal state in the spanner domain.
    The goal is to tighten all specified nuts. The heuristic is based on the observation
    that tightening each loose goal nut requires a 'tighten_nut' action, which in turn
    requires the man to be at the nut's location and to be carrying a usable spanner.
    The heuristic is a sum of three main cost components:
    1. The number of loose goal nuts (representing the 'tighten_nut' actions).
    2. The cost to get the man to the location of the closest loose goal nut (representing initial travel towards objectives).
    3. The cost to acquire the first usable spanner, if the man is not already carrying one (representing the initial spanner requirement).

    Assumptions:
    - Nuts remain at their initial locations throughout the plan (static nuts).
    - Links between locations are bidirectional.
    - All relevant objects (man, spanners, nuts, locations) and links are defined in the task description and initial state.
    - The total number of spanner objects in the domain is fixed and can be determined from the task definition.
    - A state is represented as a frozenset of PDDL fact strings.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes the shortest path distances between all pairs of locations defined by the '(link ?l1 ?l2)' static facts using a Breadth-First Search (BFS). It also identifies all location objects, all spanner objects, all nut objects, the man object(s), and the initial locations of all nuts (assuming they are static). The goal facts are stored for quick access.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to identify:
        -   The man's current location.
        -   Which spanners the man is carrying.
        -   Which spanners are usable.
        -   Which spanners are at which locations.
        -   Which nuts are loose.
    2.  Identify the set of loose nuts that are also specified as goal facts (loose goal nuts). Get their locations using the precomputed initial nut locations.
    3.  If there are no loose goal nuts, the goal is achieved, so the heuristic value is 0.
    4.  Check for unsolvability: If the total number of spanner objects in the domain is less than the number of loose goal nuts, the goal is impossible to achieve, so return infinity.
    5.  Initialize the heuristic value `h` with the number of loose goal nuts. This accounts for the 'tighten_nut' action cost for each nut.
    6.  Calculate the walk cost:
        -   Determine the set of locations where loose goal nuts are located.
        -   If the man is not currently at any of these locations, find the shortest distance from the man's current location to the closest location containing a loose goal nut using the precomputed distance map. Add this distance to `h`. If the man is already at a loose goal nut location, the walk cost is 0. If any required nut location is unreachable, return infinity.
    7.  Calculate the spanner acquisition cost:
        -   Check if the man is currently carrying any usable spanner.
        -   If the man is *not* carrying any usable spanner, he needs to acquire at least one. Find the closest usable spanner that is currently at a location. Calculate the cost to acquire it (shortest distance from the man's location to the spanner's location + 1 for the pickup action). Add this cost to `h`. If no usable spanners are available at any location when one is needed, return infinity. If the man is already carrying a usable spanner, the spanner acquisition cost is 0 for the purpose of getting the *first* spanner capability. If the closest usable spanner location is unreachable, return infinity.
    8.  Return the final calculated value of `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goal_facts = task.goals
        self.static_facts = task.static
        self.task_objects = task.objects # List of (obj_name, type_name)

        # Checklist item 5: Information from static facts extracted into suitable data structures.
        self.locations = set()
        self.links = set()
        for fact in self.static_facts:
            if fact.startswith('(link '):
                parts = fact.strip('()').split()
                loc1 = parts[1]
                loc2 = parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.links.add((loc1, loc2))
                self.links.add((loc2, loc1)) # Links are bidirectional

        # Precompute shortest path distances between all locations
        self.distance_map = self._precompute_distances()

        # Parse objects by type
        self.total_man_objects = {obj for obj, obj_type in self.task_objects if obj_type == 'man'}
        self.total_spanner_objects = {obj for obj, obj_type in self.task_objects if obj_type == 'spanner'}
        self.total_nut_objects = {obj for obj, obj_type in self.task_objects if obj_type == 'nut'}

        # Parse initial locations of nuts (assuming nuts are static)
        self.nut_initial_locations = {}
        for fact in task.initial_state:
            if fact.startswith('(at '):
                parts = fact.strip('()').split()
                obj = parts[1]
                loc = parts[2]
                if obj in self.total_nut_objects:
                    self.nut_initial_locations[obj] = loc

    def _precompute_distances(self):
        """
        Precomputes shortest path distances between all pairs of locations
        using BFS.
        """
        dist = {loc: {other_loc: math.inf for other_loc in self.locations} for loc in self.locations}

        for start_loc in self.locations:
            dist[start_loc][start_loc] = 0
            q = collections.deque([(start_loc, 0)])
            visited = {start_loc}

            while q:
                current_loc, d = q.popleft() # BFS queue

                # Find neighbors from links
                neighbors = []
                for l1, l2 in self.links:
                    if l1 == current_loc:
                        neighbors.append(l2)

                for neighbor_loc in neighbors:
                    if neighbor_loc not in visited:
                        visited.add(neighbor_loc)
                        dist[start_loc][neighbor_loc] = d + 1
                        q.append((neighbor_loc, d + 1))
        return dist

    def get_distance(self, loc1, loc2):
        """
        Returns the precomputed shortest distance between two locations.
        Returns infinity if locations are not in the map or unreachable.
        """
        if loc1 not in self.distance_map or loc2 not in self.distance_map.get(loc1, {}):
             # This indicates an issue with location parsing or unreachable locations
             # If locations are unreachable, distance is infinity.
             # If loc1 or loc2 wasn't parsed, it's an error, but returning inf is safe.
             return math.inf
        return self.distance_map[loc1][loc2]

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.

        Keyword arguments:
        node -- the current state node
        """
        state = node.state

        # 1. Parse the current state
        man_loc = None
        held_spanners = set()
        usable_spanners_in_state = set() # All usable spanners in the state
        spanners_at_locs = {} # map spanner -> location
        loose_nuts_in_state = set()

        for fact in state:
            parts = fact.strip('()').split()
            if not parts: continue # Skip empty facts if any

            predicate = parts[0]
            if predicate == 'at':
                obj = parts[1]
                loc = parts[2]
                if obj in self.total_man_objects:
                    man_loc = loc
                elif obj in self.total_spanner_objects:
                    spanners_at_locs[obj] = loc
                # Nut locations are assumed static, retrieved from self.nut_initial_locations
            elif predicate == 'carrying':
                m = parts[1]
                s = parts[2]
                if m in self.total_man_objects and s in self.total_spanner_objects:
                    held_spanners.add(s)
            elif predicate == 'usable':
                s = parts[1]
                if s in self.total_spanner_objects:
                    usable_spanners_in_state.add(s)
            elif predicate == 'loose':
                n = parts[1]
                if n in self.total_nut_objects:
                    loose_nuts_in_state.add(n)
            # We don't need 'tightened' facts directly for the heuristic calculation
            # as we focus on the 'loose' goal nuts.

        # Ensure man_loc was found (should always be the case in a valid state)
        if man_loc is None:
             # This state is likely invalid or represents an unsolvable scenario
             return math.inf

        # 2. Identify loose goal nuts and their locations
        loose_goal_nuts_info = [] # List of (nut_name, nut_location)
        goal_nuts = set()
        for goal_fact in self.goal_facts:
            if goal_fact.startswith('(tightened '):
                nut = goal_fact.strip('()').split()[1]
                goal_nuts.add(nut)
                if nut in loose_nuts_in_state:
                    # Get nut location from initial state (assuming static)
                    if nut in self.nut_initial_locations:
                        loose_goal_nuts_info.append((nut, self.nut_initial_locations[nut]))
                    else:
                        # Goal nut exists but its location is unknown? Unsolvable.
                        return math.inf

        # 3. If no loose goal nuts, goal is reached
        N_nuts = len(loose_goal_nuts_info)
        if N_nuts == 0:
            # Checklist item 2: The heuristic is 0 only for goal states.
            # If loose_goal_nuts_info is empty, all goal nuts are tightened.
            return 0

        # 4. Check for unsolvability based on total spanners
        # Checklist item 3: The heuristic value is finite for solvable states.
        # If there are fewer total spanners than nuts to tighten, it's unsolvable.
        if len(self.total_spanner_objects) < N_nuts:
             return math.inf

        # 5. Initialize heuristic with tighten action cost
        h = N_nuts

        # 6. Calculate walk cost to the closest loose nut location
        loose_nut_locations = {l_n for _, l_n in loose_goal_nuts_info}
        walk_cost = 0
        if man_loc not in loose_nut_locations:
            min_dist_to_closest_nut_loc = math.inf
            for l_n in loose_nut_locations:
                dist = self.get_distance(man_loc, l_n)
                if dist < min_dist_to_closest_nut_loc:
                    min_dist_to_closest_nut_loc = dist
            # If min_dist is still inf, it means a nut location is unreachable
            if min_dist_to_closest_nut_loc == math.inf:
                 return math.inf
            walk_cost = min_dist_to_closest_nut_loc
        h += walk_cost

        # 7. Calculate spanner acquisition cost (for the first spanner if needed)
        held_usable_spanners = held_spanners.intersection(usable_spanners_in_state)
        N_held_usable = len(held_usable_spanners)

        needs_spanner_pickup = (N_held_usable == 0)
        spanner_cost = 0
        if needs_spanner_pickup:
            # Find usable spanners that are currently at locations
            usable_spanners_at_locs = [(s, loc) for s, loc in spanners_at_locs.items() if s in usable_spanners_in_state]

            if len(usable_spanners_at_locs) == 0:
                # Man needs a spanner but none are available at locations.
                # If he doesn't hold any usable ones, it's unsolvable from here.
                return math.inf

            min_spanner_dist_cost = math.inf
            for s, l in usable_spanners_at_locs:
                dist = self.get_distance(man_loc, l)
                # If the spanner location is unreachable, this path is not viable
                if dist == math.inf:
                    continue
                cost = dist + 1 # walk + pickup action
                if cost < min_spanner_dist_cost:
                    min_spanner_dist_cost = cost

            # If min_spanner_dist_cost is still inf, it means all usable spanners
            # at locations are unreachable.
            if min_spanner_dist_cost == math.inf:
                 return math.inf

            spanner_cost = min_spanner_dist_cost
        h += spanner_cost

        # Checklist item 3: The heuristic value is finite for solvable states.
        # Infinity is returned explicitly in unsolvable cases. Otherwise, it's a sum of finite values.

        # 8. Return the final heuristic value
        return h
