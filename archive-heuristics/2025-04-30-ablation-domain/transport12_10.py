# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of necessary actions (load, unload, and transport)
    to move each package from its current location to its goal location, summing the
    minimum required actions for each package independently. It ignores vehicle capacity
    and the road network structure, assuming any location is reachable from any other
    in a single drive action for a vehicle.

    # Assumptions
    - Actions (load, unload, drive) have a cost of 1.
    - A vehicle can move between any two locations in a single 'drive' action (ignoring
      the road network structure and distance).
    - Vehicle capacity and the possibility of transporting multiple packages in a single
      drive action are ignored, leading to a potential overestimation of drive actions.
    - The heuristic counts the minimum actions required *per package* to reach its goal state.
    - All vehicles are listed with a `capacity` predicate in the static facts.
    - Packages relevant to the goal are those mentioned in `(at package location)` goal facts.

    # Heuristic Initialization
    - Extracts the goal location for each package mentioned in the task's goal conditions.
    - Identifies vehicles based on the `capacity` predicate in the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the total heuristic cost to 0.
    2. Iterate through each package that has a specified goal location according to the task definition (`self.goal_locations`).
    3. For the current package `p` with goal location `loc_goal`:
       a. Check if the fact `(at p loc_goal)` is present in the current state (`node.state`). If it is, this package has reached its goal, and we add 0 cost for this package. Continue to the next package.
       b. If the package is not at its goal location, determine its current status by examining the state facts:
          - Look for a fact `(at p loc_current)`. If found, the package is on the ground at `loc_current`. Since we already know `(at p loc_goal)` is false, `loc_current` must be different from `loc_goal`. To move it, it needs to be loaded into a vehicle, the vehicle needs to drive from `loc_current` to `loc_goal`, and the package needs to be unloaded. This requires a minimum of 3 actions (load, drive, unload). Add 3 to the total cost.
          - If no `(at p ...)` fact is found for this package, look for a fact `(in p v)`. If found, the package is inside vehicle `v`.
          - If the package is inside vehicle `v`, find the current location `loc_v` of vehicle `v` by looking for a fact `(at v loc_v)` in the state.
          - If `loc_v` is the same as the goal location `loc_goal`, the package only needs to be unloaded from the vehicle. This requires a minimum of 1 action (unload). Add 1 to the total cost.
          - If `loc_v` is different from the goal location `loc_goal`, the vehicle `v` needs to drive from `loc_v` to `loc_goal`, and then the package needs to be unloaded. This requires a minimum of 2 actions (drive, unload). Add 2 to the total cost.
          - (Handle cases where package status or vehicle location is unexpectedly missing in the state, although this indicates a potentially malformed state representation. The current implementation adds a default cost estimate in such cases).
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages
        and identifying vehicles from static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package mentioned in the goals.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goal facts are like (at package location)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Identify vehicles from static facts (those with capacity).
        self.vehicles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            # Assuming capacity facts are like (capacity vehicle size)
            if parts and parts[0] == "capacity" and len(parts) == 3:
                 self.vehicles.add(parts[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current locations of packages and vehicles.
        # package_status: {package_name: ('at', location) or ('in', vehicle_name)}
        # vehicle_locations: {vehicle_name: location}
        package_status = {}
        vehicle_locations = {}

        # Populate vehicle_locations and package_status from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "at":
                # Fact is (at obj loc)
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    if obj in self.vehicles:
                        vehicle_locations[obj] = loc
                    # Assume anything else in an 'at' fact is a package relevant to goals
                    # if it's one of the packages we care about.
                    elif obj in self.goal_locations:
                         package_status[obj] = ('at', loc)

            elif predicate == "in":
                # Fact is (in package vehicle)
                if len(parts) == 3:
                    p, v = parts[1], parts[2]
                    # Only track packages relevant to goals
                    if p in self.goal_locations:
                        package_status[p] = ('in', v)

            # Ignore other predicates like 'capacity', 'road', 'capacity-predecessor' in the state loop

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location in the current state.
            goal_fact_string = f"(at {package} {goal_location})"
            if goal_fact_string in state:
                # Package is already at its goal, no cost needed for this package.
                continue

            # If not at goal, calculate cost for this package based on its current status.
            current_status = package_status.get(package)

            # Handle cases where package status is unexpectedly missing.
            # This shouldn't happen in a valid state, but we add a fallback.
            if current_status is None:
                 # Assume it needs full transport if its status isn't explicitly known.
                 # This adds a cost of 3 (load, drive, unload).
                 total_cost += 3
                 # print(f"Warning: Status for package {package} not found in state. Assuming needs full transport.")

            elif current_status[0] == 'at':
                # Package is on the ground at current_location.
                # Since the goal fact (at package goal_location) is not in state,
                # the current_location must be different from goal_location.
                # Needs load, drive, unload.
                total_cost += 3

            elif current_status[0] == 'in':
                # Package is inside a vehicle.
                vehicle = current_status[1]
                vehicle_location = vehicle_locations.get(vehicle)

                # Handle cases where vehicle location is unexpectedly missing.
                # This shouldn't happen in a valid state, but we add a fallback.
                if vehicle_location is None:
                    # Assume vehicle is somewhere not at the goal, needs drive and unload.
                    # This adds a cost of 2 (drive, unload).
                    total_cost += 2
                    # print(f"Warning: Location for vehicle {vehicle} (carrying {package}) not found in state. Assuming needs drive+unload.")

                elif vehicle_location != goal_location:
                    # Vehicle is not at the goal location. Needs vehicle drive and unload.
                    total_cost += 2

                else: # vehicle_location == goal_location
                    # Vehicle is at the goal location. Needs only unload.
                    total_cost += 1

        return total_cost
