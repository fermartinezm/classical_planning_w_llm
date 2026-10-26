from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to extract components of a PDDL fact
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all loose nuts by summing:
    1. The number of loose nuts (for tighten actions).
    2. The number of spanner pickups needed.
    3. An estimate of the movement cost.

    Movement cost estimate:
    - Distance from the man's current location to the closest required location (spanner or nut).
    - Plus a fixed cost (2 moves) for each subsequent nut task (representing travel to get a spanner and go to the next nut).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances between locations
        and extracting goal nuts.
        """
        static_facts = task.static

        # Extract goal nuts (nuts that need to be tightened)
        self.goal_nuts = {get_parts(g)[1] for g in task.goals if g.startswith("(tightened ")}

        # Assuming the man object is named 'bob' based on examples.
        # A more general approach would parse object types from the domain or task.
        self.man_obj = 'bob'

        # Build adjacency list for locations from link facts
        self.links = {}
        all_locations = set()
        for fact in static_facts:
            if fact.startswith("(link "):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                self.links.setdefault(l1, set()).add(l2)
                self.links.setdefault(l2, set()).add(l1) # Assuming links are bidirectional
                all_locations.add(l1)
                all_locations.add(l2)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in all_locations:
            self.distances[(start_loc, start_loc)] = 0
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            while queue:
                current_loc, dist = queue.popleft()
                if current_loc in self.links:
                    for neighbor in self.links[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_loc, neighbor)] = dist + 1
                            queue.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations, or infinity if unreachable."""
        # If locations are the same, distance is 0.
        if loc1 == loc2:
            return 0
        # Look up precomputed distance. Return infinity if not found (unreachable).
        return self.distances.get((loc1, loc2), float('inf'))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        loose_nut_locations = set()
        usable_spanners_on_ground_locations = set()
        man_location = None
        man_carrying_spanner = False

        # Extract relevant information from the current state
        for fact in state:
            if fact.startswith("(at "):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                if obj == self.man_obj:
                    man_location = loc
                # Check if it's a nut that is loose and is a goal nut
                elif obj.startswith("nut") and obj in self.goal_nuts and f"(loose {obj})" in state:
                     loose_nut_locations.add(loc)
                # Check if it's a usable spanner on the ground
                elif obj.startswith("spanner") and f"(usable {obj})" in state:
                    usable_spanners_on_ground_locations.add(loc)
            # Check if the man is carrying a usable spanner
            elif fact.startswith("(carrying "):
                parts = get_parts(fact)
                carrier, item = parts[1], parts[2]
                if carrier == self.man_obj and item.startswith("spanner") and f"(usable {item})" in state:
                     man_carrying_spanner = True # Assuming the carried spanner is usable *before* the tighten action


        N_loose = len(loose_nut_locations)

        # If all goal nuts are tightened, heuristic is 0
        if N_loose == 0:
            return 0

        # Base cost: tighten actions (one per loose goal nut)
        h = N_loose

        # Cost: pickup actions
        # Need one spanner per nut. If carrying one initially, need N_loose - 1 more pickups.
        pickups_needed = max(0, N_loose - (1 if man_carrying_spanner else 0))
        h += pickups_needed

        # Check if enough usable spanners exist (carried + on ground)
        usable_spanners_available = (1 if man_carrying_spanner else 0) + len(usable_spanners_on_ground_locations)
        if usable_spanners_available < N_loose:
             # Not enough spanners to tighten all nuts
             return float('inf') # Problem is unsolvable from this state

        # Movement cost
        movement_cost = 0

        # Identify locations the man needs to visit for the first task
        first_move_targets = set(loose_nut_locations)
        if pickups_needed > 0:
            first_move_targets.update(usable_spanners_on_ground_locations)

        # Cost to reach the first required location from man's current location
        if not first_move_targets:
             # This case should not happen if N_loose > 0 and spanners are available.
             # If N_loose > 0, loose_nut_locations should be non-empty if solvable.
             # If pickups_needed > 0, usable_spanners_on_ground_locations should be non-empty if solvable.
             # If N_loose > 0 but loose_nut_locations is empty, nuts are not 'at' locations? Unsolvable.
             return float('inf') # Should not reach here in solvable states with N_loose > 0

        min_dist_man_to_first_target = float('inf')
        for target_loc in first_move_targets:
            dist = self.get_distance(man_location, target_loc)
            if dist == float('inf'): # If any required location is unreachable
                return float('inf')
            min_dist_man_to_first_target = min(min_dist_man_to_first_target, dist)
        movement_cost += min_dist_man_to_first_target

        # Cost for subsequent movements between tasks
        # After the first move and action, the man is at a required location.
        # He needs to perform N_loose - 1 more tightening tasks.
        # Each task requires getting a spanner (if not already carrying one from a previous pickup)
        # and moving to the next nut location.
        # A simplified estimate: 2 moves per remaining nut task (move to spanner, move to nut).
        if N_loose > 1:
            movement_cost += (N_loose - 1) * 2 # 2 moves per remaining nut task

        h += movement_cost

        return h
