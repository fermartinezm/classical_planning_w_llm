from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match facts (optional, but useful)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't try to match more args than parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It counts the necessary pick-up and drop actions
    for each package not at its goal, and adds a simplified estimate for drive actions.
    Specifically, for each package not at its goal:
    - If it's on the ground, it needs an estimated 3 actions (pick-up, drive, drop).
    - If it's in a vehicle, it needs an estimated 2 actions (drive, drop).
    The total heuristic is the sum of these estimates for all packages not at their goal.

    # Assumptions
    - Each package not at its goal needs at least one pick-up (if on the ground)
      and one drop action.
    - A drive action is needed whenever a package is transported between locations.
      This heuristic simplifies drive costs by associating a drive cost (of 1) with each
      package movement requirement, potentially overcounting drives but providing
      a non-admissible estimate suitable for greedy search.
    - Vehicle capacity and exact vehicle locations (beyond whether a package is 'in' one)
      are not explicitly modeled in the action count. The existence of a vehicle
      with sufficient capacity and the ability to reach locations via roads is assumed
      implicitly for the actions to be possible.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.
    - Static facts like `road` and `capacity-predecessor` are noted but not
      directly used in this simplified heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the total heuristic cost to 0.
    2. Identify the goal location for each package from the `self.goal_locations` dictionary, which was populated during initialization.
    3. Create a temporary mapping `package_status` to quickly find the current state of each package (whether it's `at` a location or `in` a vehicle). Iterate through the facts in the current state to populate this map, focusing only on facts involving packages that have a goal.
    4. Iterate through each package (`p`) and its goal location (`l_goal`) stored in `self.goal_locations`.
    5. For the current package `p`:
       a. Retrieve its current status from the `package_status` map. If a package with a goal is not found in the state facts (which shouldn't happen in valid states but is handled defensively), assume it's on the ground at a non-goal location.
       b. Check if the package is currently `(at p l_goal)`.
       c. If `p` is currently `(at p l_goal)`, it has reached its goal. Add 0 to the total cost for this package.
       d. If `p` is currently `(at p l_current)` where `l_current != l_goal`:
          - This package is on the ground and not at its goal. It needs to be picked up, transported, and dropped.
          - Add 3 to the total cost (representing pick-up, drive, and drop actions).
       e. If `p` is currently `(in p v)` for some vehicle `v`:
          - This package is in a vehicle and not yet at its goal location (since if it were at the goal, it would likely be dropped). It needs to be transported and dropped.
          - Add 2 to the total cost (representing drive and drop actions).
    6. The final `total_cost` is the heuristic value for the given state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages.
        """
        self.goals = task.goals  # Goal conditions.
        # static_facts = task.static # Static facts like road/capacity are not used in this simple heuristic

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Note: This heuristic assumes package goals are always (at package location).
            # If '(in p v)' goals were possible, the logic would need adjustment.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts).

        # Track where packages are currently located or if they are in a vehicle.
        # We only care about packages that have a goal location defined in self.goal_locations.
        package_status = {} # Maps package name to its status: ("location", l) or ("in_vehicle", v)

        # Populate package_status from the current state facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Check if the object is one of the packages we care about (i.e., has a goal)
                if obj in self.goal_locations:
                     package_status[obj] = ("location", loc)
            elif parts[0] == "in" and len(parts) == 3:
                 pkg, veh = parts[1], parts[2]
                 # Check if the object is one of the packages we care about
                 if pkg in self.goal_locations:
                     package_status[pkg] = ("in_vehicle", veh)

        total_cost = 0  # Initialize action cost counter.

        # Iterate through all packages that have a goal defined
        for package, goal_location in self.goal_locations.items():
            # Get the current status of the package.
            # If a package with a goal is somehow not found in the state facts (e.g., due to
            # an invalid state representation), we treat it as needing full transport.
            # In a valid STRIPS state, a locatable object is either 'at' a location or 'in' another object.
            # Assuming valid states, package_status will contain the package if it's in self.goal_locations.
            current_status = package_status.get(package)

            # If the package is not found in the state facts, it's an unexpected state.
            # We'll treat it as needing full transport from an unknown location.
            if current_status is None:
                 # print(f"Warning: Package {package} with goal {goal_location} not found in state facts.")
                 # Treat as on ground, not at goal
                 status_type = "location"
                 current_loc_or_veh = "unknown" # Placeholder, not used in logic below

            else:
                 status_type, current_loc_or_veh = current_status


            # Check if the package is already at its goal location
            if status_type == "location" and current_loc_or_veh == goal_location:
                # Package is on the ground at the goal. No cost for this package.
                pass # Cost is 0 for this package
            else:
                # Package is not at its goal location. It needs moving.
                # It needs at least one drop action at the goal.
                total_cost += 1 # Cost for the drop action

                # If the package is on the ground (not in a vehicle), it also needs a pick-up action.
                if status_type == "location":
                    total_cost += 1 # Cost for the pick-up action

                # It also needs transport (drive action).
                # This simple heuristic adds 1 drive action cost per package needing transport.
                # This overcounts drives if multiple packages travel together, but is simple and non-admissible.
                total_cost += 1 # Cost for the drive action

        return total_cost
