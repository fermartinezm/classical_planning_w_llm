from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use zip to handle cases where parts might be longer than args (e.g., extra parameters)
    # or where args has wildcards. fnmatch handles the wildcard matching.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It breaks down the process into phases: making sandwiches, putting them on
    trays, moving trays to children's locations, and finally serving the children.
    It counts the number of actions needed for each phase based on the current
    state, summing them up. It differentiates between regular and gluten-free
    requirements.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Making a sandwich requires ingredients and an available sandwich object (not explicitly checked for sufficiency in this heuristic, assuming solvable problem).
    - Putting a sandwich on a tray requires the sandwich to be in the kitchen and a tray to be in the kitchen.
    - Moving a tray takes one action to get it to any desired location from any current location.
    - Serving a child requires a suitable sandwich on a tray at the child's waiting location.
    - Resource contention (e.g., limited ingredients, limited trays, limited sandwich objects) is simplified or ignored for efficiency, focusing on the steps required per item/location.

    # Heuristic Initialization
    - Identify all children, their allergy status, and their waiting locations from static facts and initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who are not yet served.
    2. For each unserved child, determine if they need a regular or gluten-free sandwich and note their waiting location.
    3. Count the total number of unserved children needing regular sandwiches (`needed_reg`).
    4. Count the total number of unserved children needing gluten-free sandwiches (`needed_gf`).
    5. Identify all unique locations where unserved children are waiting (`needed_locations`).
    6. Identify all locations where trays are currently present (`tray_present_locations`).
    7. Count available sandwiches:
       - Gluten-free sandwiches currently on trays (`avail_gf_ontray`).
       - Regular sandwiches currently on trays (`avail_reg_ontray`).
       - Gluten-free sandwiches currently in the kitchen (`avail_gf_kitchen`).
       - Regular sandwiches currently in the kitchen (`avail_reg_kitchen`).
    8. Calculate the number of sandwiches that still need to end up on trays:
       - GF: `needed_gf_ontray = max(0, needed_gf - avail_gf_ontray)`
       - Reg: `needed_reg_ontray = max(0, needed_reg - avail_reg_ontray)`
    9. Calculate the number of sandwiches that need to be *made* (those needed on trays minus those already in the kitchen):
       - GF: `make_gf = max(0, needed_gf_ontray - avail_gf_kitchen)`
       - Reg: `make_reg = max(0, needed_reg_ontray - avail_reg_kitchen)`
    10. Estimate costs for each phase:
        - Cost to *make* sandwiches: `make_gf + make_reg` (1 action per sandwich made).
        - Cost to *put sandwiches on trays*: `needed_gf_ontray + needed_reg_ontray` (1 action per sandwich needing to go on a tray).
        - Cost to *move trays* to locations needing them: `len(needed_locations - tray_present_locations)` (1 action per location needing a tray that doesn't have one).
        - Cost to *serve* children: `needed_gf + needed_reg` (1 action per child served).
    11. The total heuristic value is the sum of the costs from these phases.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        self.goals = task.goals # Not strictly needed for this heuristic but good practice
        self.static = task.static # Static facts

        # Pre-process child information: allergy status and waiting place
        self.children = {} # {child_name: {'allergic': bool, 'place': place_name}}

        # Scan static facts and initial state for child info
        # Initial state might contain initial waiting facts not in static
        for fact in task.initial_state | task.static:
            parts = get_parts(fact)
            if parts[0] == 'waiting' and len(parts) == 3:
                child, place = parts[1], parts[2]
                if child not in self.children:
                    self.children[child] = {'place': place, 'allergic': False} # Default non-allergic
                else:
                    self.children[child]['place'] = place # Update place if already seen
            elif parts[0] == 'allergic_gluten' and len(parts) == 2:
                child = parts[1]
                if child not in self.children:
                     # This case might happen if 'waiting' is only in initial state
                     # and allergy is only in static. Initialize with default place.
                    self.children[child] = {'place': None, 'allergic': True}
                else:
                    self.children[child]['allergic'] = True
            elif parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                 # Ensure child exists if allergy status is the first fact seen
                child = parts[1]
                if child not in self.children:
                    self.children[child] = {'place': None, 'allergic': False}
                else:
                    self.children[child]['allergic'] = False


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their needs/locations
        needed_gf = 0
        needed_reg = 0
        needed_locations = set()

        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for child, info in self.children.items():
            if child not in served_children:
                if info['allergic']:
                    needed_gf += 1
                else:
                    needed_reg += 1
                if info['place']: # Ensure place info exists
                    needed_locations.add(info['place'])

        # If no children need serving, the goal is reached.
        if needed_gf == 0 and needed_reg == 0:
            return 0

        # 7. Count available sandwiches and tray locations
        avail_gf_ontray = 0
        avail_reg_ontray = 0
        avail_gf_kitchen = 0
        avail_reg_kitchen = 0
        tray_present_locations = set()
        kitchen_sandwiches = set() # Keep track of sandwiches in kitchen to check GF status

        # First pass to find kitchen sandwiches and tray locations
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "?s"):
                kitchen_sandwiches.add(get_parts(fact)[1])
            elif match(fact, "ontray", "?s", "?t"):
                 # Don't count sandwiches on trays yet, just identify them
                 pass # Will count in second pass
            elif match(fact, "at", "?t", "?p"):
                tray_present_locations.add(get_parts(fact)[2])

        # Second pass to count sandwich types (GF status might be anywhere in state)
        gf_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "?s")}

        for fact in state:
             if match(fact, "at_kitchen_sandwich", "?s"):
                 s = get_parts(fact)[1]
                 if s in gf_sandwiches:
                     avail_gf_kitchen += 1
                 else:
                     avail_reg_kitchen += 1
             elif match(fact, "ontray", "?s", "?t"):
                 s = get_parts(fact)[1]
                 if s in gf_sandwiches:
                     avail_gf_ontray += 1
                 else:
                     avail_reg_ontray += 1


        # 8. Calculate sandwiches needed on trays
        needed_gf_ontray = max(0, needed_gf - avail_gf_ontray)
        needed_reg_ontray = max(0, needed_reg - avail_reg_ontray)

        # 9. Calculate sandwiches that need to be made
        make_gf = max(0, needed_gf_ontray - avail_gf_kitchen)
        make_reg = max(0, needed_reg_ontray - avail_reg_kitchen)

        # 10. Estimate costs for each phase
        cost_make = make_gf + make_reg
        cost_put_on_tray = needed_gf_ontray + needed_reg_ontray # Cost to get needed sandwiches onto trays
        cost_move_tray = len(needed_locations - tray_present_locations)
        cost_serve = needed_gf + needed_reg

        # 11. Total heuristic value
        total_cost = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        return total_cost

