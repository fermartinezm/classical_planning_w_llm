from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

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

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children and adds costs based on the
    "readiness" of suitable sandwiches needed to serve them. Sandwiches
    already on trays at the correct location are cheapest to use, followed
    by sandwiches on trays elsewhere, then sandwiches in the kitchen, and
    finally sandwiches that need to be made.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Sandwiches of the correct type (gluten-free or regular) are interchangeable.
    - Trays are available when needed for putting sandwiches on or moving.
    - The costs assigned to different stages (serve, move, put, make) are fixed
      and represent a simplified view of the actions involved.
    - The heuristic assumes solvable instances up to the point of checking
      total available vs total needed sandwiches.

    # Heuristic Initialization
    - Extracts static information about child allergies (`allergic_gluten`, `not_allergic_gluten`).
    - Extracts static information about gluten-free ingredients (`no_gluten_bread`, `no_gluten_content`).
    - Identifies the set of all children who need to be served (from the goal state).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are in the goal state (`served`) but are not yet served in the current state. These are the unserved children.
    2. If there are no unserved children, the heuristic is 0 (goal state).
    3. For each unserved child, determine their waiting place and allergy status.
    4. Count the total number of gluten-free (GF) and regular (Reg) sandwiches needed across all unserved children.
    5. Count the available GF and Reg sandwiches in the current state, categorized by their "readiness":
       - On a tray at a place where a suitable child is waiting.
       - On a tray at a place where no suitable child is waiting.
       - In the kitchen (`at_kitchen_sandwich`).
       - Makable from ingredients and `notexist` slots in the kitchen.
    6. Calculate the number of GF and Reg sandwiches that can be made based on available ingredients and `notexist` slots. Prioritize making GF sandwiches if ingredients allow.
    7. Check if the total available suitable sandwiches (on trays + in kitchen + makable) are sufficient to meet the total needed sandwiches. If not, return infinity (unsolvable state).
    8. Calculate the heuristic cost by assigning available sandwiches to needed sandwiches, starting from the most ready state (on tray at waiting place), then trays elsewhere, then kitchen, then makable.
    9. Assign costs for each "stage transition" required for the assigned sandwiches:
       - Sandwiches on tray at waiting place: Cost 1 (serve).
       - Sandwiches on tray elsewhere: Cost 1 (move tray) + 1 (serve) = 2.
       - Sandwiches in kitchen: Cost 1 (put on tray) + 1 (move tray) + 1 (serve) = 3.
       - Makable sandwiches: Cost 1 (make) + 1 (put on tray) + 1 (move tray) + 1 (serve) = 4.
    10. Sum up the costs for all needed sandwiches. This sum is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract allergy information
        self.child_allergy = {}
        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergy[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergy[child] = False

        # Extract gluten-free ingredient information
        self.is_gf_bread = {}
        self.is_gf_content = {}
        for fact in static_facts:
            if match(fact, "no_gluten_bread", "*"):
                bread = get_parts(fact)[1]
                self.is_gf_bread[bread] = True
            elif match(fact, "no_gluten_content", "*"):
                content = get_parts(fact)[1]
                self.is_gf_content[content] = True

        # Identify all children who are goals
        self.goal_children = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 child = get_parts(goal)[1]
                 self.goal_children.add(child)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Identify unserved children
        unserved_children = {c for c in self.goal_children if f"(served {c})" not in state}

        # 2. If no unserved children, return 0
        if not unserved_children:
            return 0

        # 3. Count unserved children by allergy status and location
        unserved_gf_at_place = defaultdict(int)
        unserved_reg_at_place = defaultdict(int)
        waiting_places = set()

        for fact in state:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                if child in unserved_children:
                    waiting_places.add(place)
                    if self.child_allergy.get(child, False): # Default to False if allergy info missing
                        unserved_gf_at_place[place] += 1
                    else:
                        unserved_reg_at_place[place] += 1

        # 4. Count total needed sandwiches
        total_gf_needed = sum(unserved_gf_at_place.values())
        total_reg_needed = sum(unserved_reg_at_place.values())

        # 5. Count available sandwiches by type and state
        gf_ontray_at_p = defaultdict(int)
        reg_ontray_at_p = defaultdict(int)
        gf_kitchen_avail = 0
        reg_kitchen_avail = 0
        num_bread_gf = 0
        num_content_gf = 0
        num_bread_reg = 0
        num_content_reg = 0
        num_notexist = 0

        ontray_sandwiches = {} # map sandwich to tray
        tray_locations = {} # map tray to place

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:]
                ontray_sandwiches[s] = t
            elif match(fact, "at", "*", "*"):
                t, p = get_parts(fact)[1:]
                tray_locations[t] = p
            elif match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                # Check if this sandwich is gluten-free
                is_gf = f"(no_gluten_sandwich {s})" in state
                if is_gf:
                    gf_kitchen_avail += 1
                else:
                    reg_kitchen_avail += 1
            elif match(fact, "at_kitchen_bread", "*"):
                b = get_parts(fact)[1]
                if self.is_gf_bread.get(b, False):
                    num_bread_gf += 1
                else:
                    num_bread_reg += 1
            elif match(fact, "at_kitchen_content", "*"):
                c = get_parts(fact)[1]
                if self.is_gf_content.get(c, False):
                    num_content_gf += 1
                else:
                    num_content_reg += 1
            elif match(fact, "notexist", "*"):
                num_notexist += 1

        # Populate gf_ontray_at_p and reg_ontray_at_p based on tray locations
        for s, t in ontray_sandwiches.items():
            if t in tray_locations:
                p = tray_locations[t]
                is_gf = f"(no_gluten_sandwich {s})" in state
                if is_gf:
                    gf_ontray_at_p[p] += 1
                else:
                    reg_ontray_at_p[p] += 1

        # Count sandwiches on trays at waiting places vs elsewhere
        gf_ontray_at_waiting_place = sum(gf_ontray_at_p[p] for p in waiting_places)
        reg_ontray_at_waiting_place = sum(reg_ontray_at_p[p] for p in waiting_places)

        gf_ontray_anywhere = sum(gf_ontray_at_p.values())
        reg_ontray_anywhere = sum(reg_ontray_at_p.values())

        gf_ontray_not_at_waiting_place = gf_ontray_anywhere - gf_ontray_at_waiting_place
        reg_ontray_not_at_waiting_place = reg_ontray_anywhere - reg_ontray_at_waiting_place


        # 6. Calculate makable sandwiches
        makable_gf = min(num_bread_gf, num_content_gf, num_notexist)
        remaining_notexist = num_notexist - makable_gf
        makable_reg = min(num_bread_reg, num_content_reg, remaining_notexist)

        # 7. Check solvability based on total available vs total needed
        total_avail_gf = gf_ontray_anywhere + gf_kitchen_avail + makable_gf
        total_avail_reg = reg_ontray_anywhere + reg_kitchen_avail + makable_reg

        if total_avail_gf < total_gf_needed or total_avail_reg < total_reg_needed:
             return float('inf') # State is likely unsolvable with available resources

        # 8. Calculate heuristic cost by assigning resources
        h = 0
        remaining_gf_needed = total_gf_needed
        remaining_reg_needed = total_reg_needed

        # 9. Assign costs based on stages

        # Stage 1: Sandwiches on tray at waiting places (Cost 1: serve)
        serve_gf = min(remaining_gf_needed, gf_ontray_at_waiting_place)
        h += serve_gf * 1
        remaining_gf_needed -= serve_gf

        serve_reg = min(remaining_reg_needed, reg_ontray_at_waiting_place)
        h += serve_reg * 1
        remaining_reg_needed -= serve_reg

        # Stage 2: Sandwiches on trays not at waiting places (Cost 2: move + serve)
        move_gf = min(remaining_gf_needed, gf_ontray_not_at_waiting_place)
        h += move_gf * 2
        remaining_gf_needed -= move_gf

        move_reg = min(remaining_reg_needed, reg_ontray_not_at_waiting_place)
        h += move_reg * 2
        remaining_reg_needed -= move_reg

        # Stage 3: Sandwiches in kitchen (Cost 3: put + move + serve)
        kitchen_gf = min(remaining_gf_needed, gf_kitchen_avail)
        h += kitchen_gf * 3
        remaining_gf_needed -= kitchen_gf

        kitchen_reg = min(remaining_reg_needed, reg_kitchen_avail)
        h += kitchen_reg * 3
        remaining_reg_needed -= kitchen_reg

        # Stage 4: Makable sandwiches (Cost 4: make + put + move + serve)
        makable_gf_used = min(remaining_gf_needed, makable_gf)
        h += makable_gf_used * 4
        remaining_gf_needed -= makable_gf_used

        makable_reg_used = min(remaining_reg_needed, makable_reg)
        h += makable_reg_used * 4
        remaining_reg_needed -= makable_reg_used

        # 10. The total cost is the sum calculated.
        # If remaining_gf_needed > 0 or remaining_reg_needed > 0, it means the total available was less than needed,
        # which should have been caught by the float('inf') check.
        # So, remaining_gf_needed and remaining_reg_needed should be 0 here.

        return h
