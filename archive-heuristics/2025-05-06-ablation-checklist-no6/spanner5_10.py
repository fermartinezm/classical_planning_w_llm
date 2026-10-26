# Assuming the Heuristic base class is provided externally
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the number of actions (walk, pickup, tighten) required to tighten
    all goal nuts. It uses a greedy simulation approach: the man repeatedly
    goes to the closest available usable spanner (if needed), picks it up,
    goes to the closest loose goal nut, and tightens it.

    # Heuristic Initialization
    - Identifies all objects by type (man, spanner, nut, location) present in the task facts.
    - Stores goal nut names and their initial locations.
    - Builds the location graph from 'link' facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic (__call__)
    1. Identify the man's current location and if he's carrying a usable spanner.
    2. Identify all loose goal nuts and their locations.
    3. Identify all usable spanners currently on the ground and their locations.
    4. If no loose goal nuts, return 0.
    5. Initialize heuristic cost `h = 0`.
    6. Initialize current man location, set of remaining loose nuts, and set of remaining usable spanners on the ground for the simulation.
    7. While there are remaining loose nuts:
        a. If the man is not carrying a usable spanner:
            i. Find the closest usable spanner on the ground that is reachable.
            ii. If none exists, the state is likely unsolvable; return a large value.
            iii. Add walk cost to spanner location + 1 (pickup) to `h`.
            iv. Update man's current location to spanner location.
            v. Mark man as carrying a usable spanner.
            vi. Remove the picked-up spanner from available ground spanners.
        b. Find the closest remaining loose goal nut that is reachable.
        c. If none exists, the state is likely unsolvable; return a large value.
        d. Add walk cost to nut location + 1 (tighten) to `h`.
        e. Update man's current location to nut location.
        f. Mark man as *not* carrying a usable spanner (it's consumed).
        g. Remove the tightened nut from remaining loose nuts.
    8. Return `h`.

    # Assumptions:
    - There is exactly one man.
    - Nut locations are static (true in PDDL).
    - Spanner locations are static unless carried (true in PDDL).
    - A usable spanner is consumed after one tighten action (true in PDDL).
    - The location graph is undirected (links are bidirectional).
    - Solvable instances have enough usable spanners initially and all required locations are reachable.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and goal details.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Identify objects by type present in the task facts
        self.men = set()
        self.spanners = set()
        self.nuts = set()
        self.locations = set()

        # Collect all unique objects and infer types from predicates they appear in
        all_relevant_facts = set(initial_state) | set(static_facts) | set(self.goals)

        for fact in all_relevant_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at':
                if len(args) == 2:
                    obj, loc = args
                    # 'at' predicate arguments are locatable and location
                    # We'll refine types based on other predicates
                    self.locations.add(loc)
            elif predicate == 'carrying':
                 if len(args) == 2:
                    m, s = args
                    self.men.add(m)
                    self.spanners.add(s)
            elif predicate == 'usable':
                 if len(args) == 1:
                    s = args[0]
                    self.spanners.add(s)
            elif predicate == 'link':
                 if len(args) == 2:
                    l1, l2 = args
                    self.locations.add(l1)
                    self.locations.add(l2)
            elif predicate == 'tightened' or predicate == 'loose':
                 if len(args) == 1:
                    n = args[0]
                    self.nuts.add(n)

        # Ensure we found exactly one man (assumption based on typical instances)
        # If there can be multiple men, the heuristic needs significant changes.
        if len(self.men) != 1:
             # This heuristic is designed for a single man.
             # Store the man if found, otherwise handle in __call__.
             self.the_man = list(self.men)[0] if self.men else None
        else:
             self.the_man = list(self.men)[0]


        # 2. Store goal nut names and their initial locations
        self.goal_nut_locations = {}
        goal_nuts_set = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                goal_nuts_set.add(nut)

        # Find initial locations of all nuts (assuming they are static and on the ground initially)
        nut_initial_locations = {}
        for fact in initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in self.nuts:
                    nut_initial_locations[obj] = loc

        # Store locations only for goal nuts that were found at a location initially
        for nut in goal_nuts_set:
             if nut in nut_initial_locations:
                 self.goal_nut_locations[nut] = nut_initial_locations[nut]
             else:
                 # This case implies a goal nut is not on the ground initially.
                 # The domain doesn't allow carrying nuts, so this shouldn't happen
                 # in valid instances according to the domain definition.
                 # If it did, the heuristic would need to account for how the nut gets to a location.
                 # Assuming valid instances where goal nuts start at a location.
                 # If a goal nut's location isn't found, it won't be in self.goal_nut_locations,
                 # and the heuristic will treat it as if it doesn't need tightening (which is wrong).
                 # A more robust heuristic might check if all goal nuts have known locations.
                 pass


        # 3. Build the location graph from 'link' facts
        self.location_graph = {loc: set() for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                # Ensure locations from links are in our identified locations set
                if l1 in self.locations and l2 in self.locations:
                    self.location_graph[l1].add(l2)
                    self.location_graph[l2].add(l1) # Links are bidirectional

        # 4. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_loc, dist = queue.popleft()
                self.distances[start_node][current_loc] = dist

                # Check if current_loc exists as a key in the graph (it should if from self.locations)
                if current_loc in self.location_graph:
                    for neighbor in self.location_graph.get(current_loc, set()): # Use .get for safety
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

            # Mark unreachable locations with infinity
            for loc in self.locations:
                 if loc not in self.distances[start_node]:
                      self.distances[start_node][loc] = float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Handle case with no man or multiple men (heuristic assumption violated)
        if self.the_man is None:
             # No man object found in task facts. Cannot perform actions.
             # If there are loose goal nuts, this is unsolvable.
             # Check if any goal nut is loose.
             needs_tightening = any(f'(loose {nut})' in state for nut in self.goal_nut_locations)
             return 1000000 if needs_tightening else 0


        # 1. Identify man's current location and if he's carrying a usable spanner.
        man_location = None
        carrying_spanner = None
        carrying_usable_spanner = False

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == self.the_man:
                man_location = parts[2]
            elif parts[0] == 'carrying' and parts[1] == self.the_man:
                carrying_spanner = parts[2]

        # If man_location is None, the man is not 'at' any location. This shouldn't happen in valid states.
        if man_location is None:
             # Man is nowhere? Unsolvable if actions are needed.
             needs_tightening = any(f'(loose {nut})' in state for nut in self.goal_nut_locations)
             return 1000000 if needs_tightening else 0


        if carrying_spanner and f'(usable {carrying_spanner})' in state:
             carrying_usable_spanner = True

        # 2. Identify all loose goal nuts and their locations.
        # Filter the pre-computed goal_nut_locations by checking if the nut is still loose in the current state
        remaining_loose_goal_nuts = {
            nut for nut in self.goal_nut_locations
            if f'(loose {nut})' in state
        }

        if not remaining_loose_goal_nuts:
            return 0 # Goal reached

        # 3. Identify all usable spanners currently on the ground and their locations.
        remaining_usable_spanners_on_ground = {} # {spanner: location}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in self.spanners and f'(usable {obj})' in state:
                    remaining_usable_spanners_on_ground[obj] = loc

        # 4. Greedy simulation to estimate cost
        h = 0
        current_man_loc = man_location
        current_carrying_usable = carrying_usable_spanner
        current_remaining_loose_nuts = set(remaining_loose_goal_nuts)
        current_remaining_usable_spanners_on_ground = dict(remaining_usable_spanners_on_ground)

        # Simulation loop: While there are nuts left to tighten
        while current_remaining_loose_nuts:
            # Step 1: Ensure man is carrying a usable spanner
            if not current_carrying_usable:
                # Man needs a spanner. Find the closest usable one on the ground.
                if not current_remaining_usable_spanners_on_ground:
                    # No usable spanners left on the ground, and man isn't carrying one.
                    # If there are still loose nuts, this state is likely unsolvable.
                    return 1000000 # Return a large value

                min_dist_to_spanner = float('inf')
                closest_spanner = None
                closest_spanner_loc = None

                for s, s_loc in current_remaining_usable_spanners_on_ground.items():
                    # Check if spanner location is reachable from current man location
                    if current_man_loc in self.distances and s_loc in self.distances[current_man_loc]:
                        dist = self.distances[current_man_loc][s_loc]
                        if dist < min_dist_to_spanner:
                            min_dist_to_spanner = dist
                            closest_spanner = s
                            closest_spanner_loc = s_loc

                if closest_spanner_loc is None or min_dist_to_spanner == float('inf'):
                     # No reachable usable spanners on the ground. Unsolvable.
                     return 1000000

                # Add cost to walk to spanner and pick it up
                h += min_dist_to_spanner # Walk action(s)
                current_man_loc = closest_spanner_loc # Update man's location
                h += 1 # Pickup action
                current_carrying_usable = True # Man is now carrying a usable spanner
                del current_remaining_usable_spanners_on_ground[closest_spanner] # Spanner is no longer on the ground

            # Step 2: Go to a nut and tighten it
            # Man is now carrying a usable spanner. Find the closest loose goal nut.
            min_dist_to_nut = float('inf')
            closest_nut = None
            closest_nut_loc = None

            for nut in current_remaining_loose_nuts:
                # Ensure the goal nut location was found in __init__
                if nut not in self.goal_nut_locations:
                     # This nut is a goal but its location is unknown. Unsolvable.
                     return 1000000

                nut_loc = self.goal_nut_locations[nut]
                 # Check if nut location is reachable from current man location
                if current_man_loc in self.distances and nut_loc in self.distances[current_man_loc]:
                    dist = self.distances[current_man_loc][nut_loc]
                    if dist < min_dist_to_nut:
                        min_dist_to_nut = dist
                        closest_nut = nut
                        closest_nut_loc = nut_loc

            if closest_nut_loc is None or min_dist_to_nut == float('inf'):
                 # No reachable loose goal nuts left. Unsolvable.
                 return 1000000

            # Add cost to walk to nut and tighten it
            h += min_dist_to_nut # Walk action(s)
            current_man_loc = closest_nut_loc # Update man's location
            h += 1 # Tighten action
            current_carrying_usable = False # Spanner is consumed, man is no longer carrying a usable one
            current_remaining_loose_nuts.remove(closest_nut) # Nut is now tightened

        return h
