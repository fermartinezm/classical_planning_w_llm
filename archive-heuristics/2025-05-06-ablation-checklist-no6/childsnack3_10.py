from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    # Ensure fact has at least as many parts as args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all children.
    The heuristic breaks down the problem into stages: making sandwiches,
    putting them on trays, moving trays to children's locations, and serving.
    It counts the number of items/goals that need to pass through each stage,
    considering sandwich types (gluten-free) and tray locations.

    Heuristic value is the sum of:
    1. Number of sandwiches that still need to be made (considering GF needs).
    2. Number of sandwiches that are made but not yet on a tray (and are needed).
    3. Number of locations with unserved children that do not currently have a tray.
    4. Number of unserved children (representing the final serve action).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children, allergic children,
        and waiting locations from the task definition and static facts.
        """
        self.goal_children = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'served':
                self.goal_children.add(parts[1])

        self.allergic_children = set()
        self.waiting_locations = {} # child -> place (static mapping)
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                # parts[1] is the child name
                self.allergic_children.add(parts[1])
            elif parts[0] == 'waiting':
                 # Static facts contain initial waiting locations
                 # parts[1] is child, parts[2] is place
                 self.waiting_locations[parts[1]] = parts[2]

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, 'served', '*')}
        unserved_children = self.goal_children - served_children

        # If all goal children are served, the heuristic is 0.
        if not unserved_children:
            return 0

        # Count unserved children requiring GF vs any sandwich
        unserved_gf_req = len([c for c in unserved_children if c in self.allergic_children])
        unserved_any_req = len(unserved_children) - unserved_gf_req

        # 2. Count available sandwiches in different states and types

        # Identify which sandwiches are gluten-free based on the state
        gf_sandwiches_in_state = {get_parts(fact)[1] for fact in state if match(fact, 'no_gluten_sandwich', '*')}

        # Count available sandwiches by state (ontray/kitchen) and type (GF/Reg)
        avail_gf_ontray = 0
        avail_reg_ontray = 0
        avail_gf_kitchen = 0
        avail_reg_kitchen = 0
        # avail_notexist = 0 # Not strictly needed for this heuristic's calculation

        for fact in state:
            if match(fact, 'ontray', '*', '*'):
                s_name = get_parts(fact)[1]
                if s_name in gf_sandwiches_in_state:
                    avail_gf_ontray += 1
                else:
                    # Assume non-GF if not explicitly marked GF
                    avail_reg_ontray += 1
            elif match(fact, 'at_kitchen_sandwich', '*'):
                s_name = get_parts(fact)[1]
                if s_name in gf_sandwiches_in_state:
                    avail_gf_kitchen += 1
                else:
                     # Assume non-GF if not explicitly marked GF
                    avail_reg_kitchen += 1
            # elif match(fact, 'notexist', '*'):
            #     avail_notexist += 1 # Count notexist objects (potential new sandwiches)

        # 3. Estimate actions needed for each stage

        # Cost to Make Sandwiches:
        # We need 'unserved_gf_req' GF sandwiches and 'unserved_any_req' regular sandwiches.
        # Prioritize using existing GF sandwiches for GF requests.
        # Total available made GF sandwiches (ontray + kitchen)
        total_avail_gf_made = avail_gf_ontray + avail_gf_kitchen
        gf_used_for_gf = min(unserved_gf_req, total_avail_gf_made)
        rem_avail_gf_made = total_avail_gf_made - gf_used_for_gf

        # GF sandwiches that must be made
        cost_make_gf = max(0, unserved_gf_req - total_avail_gf_made)

        # Regular sandwiches that must be made (after using available regular and remaining available GF)
        # Total available made Reg sandwiches (ontray + kitchen)
        total_avail_reg_made = avail_reg_ontray + avail_reg_kitchen
        cost_make_reg = max(0, unserved_any_req - total_avail_reg_made - rem_avail_gf_made)

        cost_make = cost_make_gf + cost_make_reg

        # Cost to Put On Tray:
        # Total sandwiches needed is the number of unserved children.
        # Sandwiches already on trays don't need this step.
        # The number of sandwiches that need to be put on trays is the total needed minus those already on trays.
        total_avail_ontray = avail_gf_ontray + avail_reg_ontray
        cost_put_ontray = max(0, len(unserved_children) - total_avail_ontray)

        # Cost to Move Tray:
        # Identify unique locations where unserved children are waiting.
        # Note: waiting_locations is static, so we use it directly for unserved children.
        locations_with_unserved = set(self.waiting_locations[c] for c in unserved_children)

        # Count how many of these locations do not currently have a tray.
        cost_move_tray = 0
        for loc in locations_with_unserved:
            # Check if any tray is currently at this location
            tray_at_loc = any(match(fact, 'at', 'tray*', loc) for fact in state)
            if not tray_at_loc:
                cost_move_tray += 1 # Need to move one tray to this location

        # Cost to Serve:
        # One serve action is needed for each unserved child.
        cost_serve = len(unserved_children)

        # Total heuristic value is the sum of estimated costs for each stage.
        heuristic_value = cost_make + cost_put_ontray + cost_move_tray + cost_serve

        return heuristic_value
