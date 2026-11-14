from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract parts of a fact without parentheses."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches a pattern with optional wildcards.

    Returns True if all parts match the pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal in the `secret_domain`
    by reasoning about the positions of du66z5od objects (type of ?b3qr2) relative to their goal locations (x6ibpitl objects),
    and the positions of hctyyy8k objects (carriers) and kyvrixb5 objects (carried entities).

    Assumptions
    -----------
    - Goals are conjunctions of (sjuejzxf ?du66z5od ?x6ibpitl) facts.
    - Objects of type du66z5od ("packages") move between locations (x6ibpitl) by carriers of type hctyyy8k ("vehicles").
    - Vehicles carry kyvrixb5 objects ("cargo"), and each vehicle carries at most one cargo at a time.
      Carriage status is tracked by (do9fmoyo ?hctyyy8k ?kyvrixb5), and cargo relationships by (hab0vli4 ?kyvrixb5 ?kyvrixb5).
    - Movement of a vehicle between locations changes (sjuejzxf ?hctyyy8k ?x6ibpitl).
    - Du66z5od objects change locations by getting picked up and dropped off via szsevy61 and tmoxe3x5 actions.
    - The static predicate (miud1izi ?x6ibpitl ?x6ibpitl) represent bidirectional adjacency between locations.
    - Transitions involve a small fixed number of actions per location movement or pickup/drop.

    Heuristic Initialization
    ------------------------
    - Extract goal facts for all du66z5od objects and their target x6ibpitl locations.
    - Build adjacency graph of locations from static (miud1izi) facts.
    - Precompute shortest-path distances between all pairs of locations using BFS.
    - Store vehicle-to-cargo admissible carriage pairs from (hab0vli4) static facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state is a goal state, return heuristic = 0.

    2. For each du66z5od object, get its current location (the location where (sjuejzxf ?du66z5od ?x6ibpitl) is true)
       and its goal location.

    3. Calculate the minimal travel distance between current and goal location of each du66z5od object,
       using precomputed shortest paths.

    4. For each du66z5od object not at its goal location:
       - Add a cost to pick it up and drop it off.
         Picking/dropping involves changing (sjuejzxf) facts and the status of (un606moq) and (do9fmoyo).
         Approximate this by 3 actions per package move (pickup, move vehicle, drop).

    5. For vehicles (hctyyy8k), if they carry any cargo (do9fmoyo true), and that cargo is not at the goal location
       of the du66z5od they are carrying, count an estimated cost to move toward that goal location.

    6. Sum all the above costs. The heuristic is the sum of movement costs over all packages plus the minimal
       estimated vehicle repositioning costs.

    7. Return the total sum as the heuristic estimate.

    Notes:
    - This heuristic is not admissible but aims to closely estimate expected number of required actions.
    - The heuristic is efficiently computable using BFS for distances and direct lookups for predicates.
    - The heuristic avoids expansive search or relaxed planning graphs.

    """

    def __init__(self, task):
        # Extract goals as mapping from du66z5od objects to goal locations (x6ibpitl)
        self.goal_map = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "sjuejzxf" and len(parts) == 3:
                du_obj, goal_loc = parts[1], parts[2]
                self.goal_map[du_obj] = goal_loc

        # Build adjacency graph for x6ibpitl locations from static (miud1izi)
        self.adjacency = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "miud1izi":
                loc1, loc2 = parts[1], parts[2]
                self.adjacency.setdefault(loc1, set()).add(loc2)
                self.adjacency.setdefault(loc2, set()).add(loc1)

        # Precompute shortest path distances between locations using BFS
        self.dist_cache = {}

        def bfs(start):
            dist = {start: 0}
            frontier = [start]
            while frontier:
                current = frontier.pop(0)
                for neighbor in self.adjacency.get(current, []):
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        frontier.append(neighbor)
            return dist

        # Precompute distances for all locations
        for loc in self.adjacency:
            self.dist_cache[loc] = bfs(loc)

        # Extract valid cargo to vehicle carrying relationships from static (hab0vli4)
        # Map vehicle to set of cargos it can carry
        self.vehicle_cargo_map = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "hab0vli4" and len(parts) == 3:
                cargo, vehicle = parts[1], parts[2]
                self.vehicle_cargo_map.setdefault(vehicle, set()).add(cargo)

    def __call__(self, node):
        state = node.state

        # Quickly return 0 for goal recognition
        if node.task.goal_reached(state):
            return 0

        # Extract current locations of each du66z5od object based on (sjuejzxf ?du66z5od ?x6ibpitl)
        du_obj_locations = {}
        # Also track locations of vehicles (hctyyy8k) (sjuejzxf ?hctyyy8k ?x6ibpitl)
        vehicle_locations = {}
        # Track vehicle cargo status (do9fmoyo ?hctyyy8k ?kyvrixb5)
        vehicle_cargo = {}

        # Cache for quick membership test of (sjuejzxf ?obj ?loc)
        sjuejzxf_facts = [fact for fact in state if fact.startswith("(sjuejzxf ")]

        for fact in sjuejzxf_facts:
            parts = get_parts(fact)
            predicate, obj, loc = parts[0], parts[1], parts[2]
            # Objects of type du66z5od appear in goal_map keys and do not appear in static facts?
            # Heuristic assumption: du66z5od objects are keys in goal_map; others are vehicles if in vehicle_cargo_map
            if obj in self.goal_map:
                # This is a du66z5od object location
                du_obj_locations[obj] = loc
            elif obj in self.vehicle_cargo_map:
                # This is a vehicle location
                vehicle_locations[obj] = loc
            # else: ignore other cases

        # Track current vehicle cargo (do9fmoyo ?hctyyy8k ?kyvrixb5)
        for fact in state:
            if fact.startswith("(do9fmoyo "):
                parts = get_parts(fact)
                _, vehicle, cargo = parts
                vehicle_cargo[vehicle] = cargo

        total_cost = 0

        # Cost constants (approximate number of actions)
        pickup_drop_cost = 3  # approximate cost for picking up and dropping a package (one cycle)
        vehicle_move_cost_per_step = 1  # every adjacency traversal considered 1 action cost

        # For each du66z5od object (package)
        for du_obj, goal_loc in self.goal_map.items():
            current_loc = du_obj_locations.get(du_obj, None)
            if current_loc is None:
                # Package location unknown - heuristic penalize highly (arbitrary large cost)
                total_cost += 10
                continue
            if current_loc == goal_loc:
                # Already at goal location - no cost for movement or pickup/drop needed
                continue

            # Distance between current and goal location for the package
            dist = self.dist_cache.get(current_loc, {}).get(goal_loc, None)
            if dist is None:
                # No path found, penalize highly
                dist = 10

            # Heuristic cost for moving this package:
            # distance * vehicle moves + pickup/drop cycles
            # Assume: vehicle must move "dist" steps, and do one pickup + one drop
            cost_for_package = dist * vehicle_move_cost_per_step + pickup_drop_cost
            total_cost += cost_for_package

        # Additionally, consider vehicle repositioning costs for vehicles carrying cargo
        for vehicle, cargo in vehicle_cargo.items():
            # Try to find related du66z5od object that is the cargo
            # Because cargo are kyvrixb5 type, and du66z5od objects differ,
            # We estimate cost based on whether cargo relates to du66z5od objects at goal or not.
            # Approximate by checking if any du66z5od's goal relates to the cargo

            # Find du66z5od objects that can be associated with this cargo via hab0vli4
            # We'll estimate cost if vehicle location differs from relevant goal location

            # Find du66z5od objects whose goal location we know
            impacted_packages = []
            for d_obj in self.goal_map:
                # Check if d_obj's goal cargo matches this vehicle's cargo or is unknown, to estimate influence
                # This is a loose heuristic: we add cost if vehicle carrying cargo but is not at its goal location
                goal_loc = self.goal_map[d_obj]
                du_loc = du_obj_locations.get(d_obj, None)
                if du_loc is None:
                    continue
                if cargo in self.vehicle_cargo_map.get(vehicle, set()):
                    # heuristic: if vehicle location not equal to d_obj goal location, add cost
                    veh_loc = vehicle_locations.get(vehicle, None)
                    if veh_loc is not None and veh_loc != goal_loc:
                        dist = self.dist_cache.get(veh_loc, {}).get(goal_loc, 0)
                        total_cost += dist * vehicle_move_cost_per_step  # vehicle travel to goal location

        return max(total_cost, 1)  # heuristic returns at least 1 for non-goal states to break ties
