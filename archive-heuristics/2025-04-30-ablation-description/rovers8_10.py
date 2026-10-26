from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This case should ideally not happen with valid PDDL state representations
        # print(f"Warning: Unexpected fact format in get_parts: {fact}")
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the remaining effort to achieve the goal conditions
    by counting the number of unsatisfied goal predicates and assigning a cost
    based on the type of data communication required and whether the necessary
    data (soil sample, rock sample, or image) has already been collected.

    # Assumptions
    - The primary goal is to communicate data (soil, rock, or image).
    - Collecting data (sampling or imaging) is a prerequisite for communication.
    - Imaging requires calibration before taking the picture.
    - Navigation costs are ignored.
    - Specific rover/camera/store assignments and capabilities are not explicitly
      tracked in the heuristic calculation per goal, only the *existence* of
      the required data (`have_...`) is checked.

    # Heuristic Initialization
    - The heuristic stores the set of goal conditions from the task.
    - Static facts are available in `self.static` but not explicitly pre-processed or used
      in this simple version of the heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of costs for each goal condition that is
    not yet satisfied in the current state.

    For each goal predicate `G` in the task's goals:
    1. Check if `G` is present in the current state. If yes, the cost for this goal is 0.
    2. If `G` is not in the current state, determine its type:
       - If `G` is `(communicated_soil_data ?w)`:
         - Check if `(have_soil_analysis ?r ?w)` exists in the state for *any* rover `?r`.
         - If a matching `have_soil_analysis` fact exists: Cost = 1 (represents the communication step).
         - If no matching `have_soil_analysis` fact exists: Cost = 2 (represents the sampling step + communication step).
       - If `G` is `(communicated_rock_data ?w)`:
         - Check if `(have_rock_analysis ?r ?w)` exists in the state for *any* rover `?r`.
         - If a matching `have_rock_analysis` fact exists: Cost = 1 (represents the communication step).
         - If no matching `have_rock_analysis` fact exists: Cost = 2 (represents the sampling step + communication step).
       - If `G` is `(communicated_image_data ?o ?m)`:
         - Check if `(have_image ?r ?o ?m)` exists in the state for *any* rover `?r`.
         - If a matching `have_image` fact exists: Cost = 1 (represents the communication step).
         - If no matching `have_image` fact exists: Cost = 3 (represents the calibration step + imaging step + communication step).
    3. Sum the costs for all unsatisfied goals to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic with the task goals and static facts.
        """
        super().__init__(task) # Call the base class constructor
        # self.goals is now available from the base class
        # self.static is now available from the base class

        # Static facts are available in self.static but not used in this simple heuristic.
        # If a more complex heuristic were needed (e.g., shortest path),
        # static facts like 'visible' and 'can_traverse' would be processed here.
        # Example: Building a graph for navigation.
        # self.navigation_graph = self._build_navigation_graph(self.static)
        # self.lander_location = self._find_lander_location(self.static)
        # self.calibration_targets = self._find_calibration_targets(self.static)
        # self.visible_from_objectives = self._find_visible_from_objectives(self.static)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        total_cost = 0

        # Iterate through each goal condition
        for goal in self.goals:
            # Check if the goal is already satisfied in the current state
            if goal in state:
                continue # Goal is met, cost is 0 for this goal

            # Goal is not satisfied, calculate its cost
            goal_parts = get_parts(goal)
            if not goal_parts: # Handle malformed goal fact
                # print(f"Warning: Skipping malformed goal fact: {goal}")
                continue

            predicate = goal_parts[0]

            if predicate == "communicated_soil_data":
                if len(goal_parts) < 2:
                    # print(f"Warning: Malformed soil goal fact: {goal}")
                    total_cost += 1 # Assign a minimal cost
                    continue
                waypoint = goal_parts[1]
                # Check if the soil sample has been collected by any rover
                have_soil = any(match(fact, "have_soil_analysis", "*", waypoint) for fact in state)
                if have_soil:
                    # Needs communication
                    total_cost += 1
                else:
                    # Needs sampling and communication
                    total_cost += 2
            elif predicate == "communicated_rock_data":
                 if len(goal_parts) < 2:
                    # print(f"Warning: Malformed rock goal fact: {goal}")
                    total_cost += 1 # Assign a minimal cost
                    continue
                 waypoint = goal_parts[1]
                 # Check if the rock sample has been collected by any rover
                 have_rock = any(match(fact, "have_rock_analysis", "*", waypoint) for fact in state)
                 if have_rock:
                     # Needs communication
                     total_cost += 1
                 else:
                     # Needs sampling and communication
                     total_cost += 2
            elif predicate == "communicated_image_data":
                 if len(goal_parts) < 3:
                    # print(f"Warning: Malformed image goal fact: {goal}")
                    total_cost += 1 # Assign a minimal cost
                    continue
                 objective = goal_parts[1]
                 mode = goal_parts[2]
                 # Check if the image has been taken by any rover
                 have_image = any(match(fact, "have_image", "*", objective, mode) for fact in state)
                 if have_image:
                     # Needs communication
                     total_cost += 1
                 else:
                     # Needs calibration, imaging, and communication
                     total_cost += 3
            else:
                 # Handle any other unexpected goal predicates
                 # print(f"Warning: Unknown goal predicate encountered: {goal}")
                 total_cost += 1 # Assign a minimal cost for unknown goals

        return total_cost
