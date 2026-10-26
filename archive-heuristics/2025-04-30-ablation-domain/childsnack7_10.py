from fnmatch import fnmatch
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    waiting children. It sums the estimated cost for each unserved child,
    considering the effort to get a suitable sandwich to their table and the
    robot actions needed for pickup and serving.

    # Assumptions
    - Each child needs exactly one sandwich.
    - All required ingredients (bread, content) and trays are initially available
      in the kitchen, unless explicitly stated otherwise in the state (e.g.,
      ingredients already used).
    - Robot starts in the kitchen if its location is not specified in the state.
    - Trays are typically either in the kitchen or at a table.
    - Unit cost for all actions.
    - The heuristic calculates costs for each child independently, which might
      overestimate in cases of shared resources (robot, trays) or shared locations,
      but aims to guide greedy search effectively.

    # Heuristic Initialization
    - Extracts static information such as child allergies, waiting locations,
      and gluten-free ingredient types from the task's static facts.
    - Collects all potential sandwich objects mentioned in the problem's initial
      state, goals, or static facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each child that has
    not yet been served. For each unserved child C:

    1.  **Determine the child's needs:** Identify the table where the child is
        waiting and whether the child is allergic to gluten based on static facts.

    2.  **Estimate cost to get a suitable sandwich to the child's table:**
        Find the minimum cost to get *any* sandwich that is ready (has bread and content)
        and valid (meets allergy requirements for child C) onto a tray located at the
        child's table. This minimum is calculated by considering two options:
        a.  **Using an existing suitable sandwich:** Iterate through all known
            sandwich objects. For each sandwich, determine if it is ready and valid
            for child C based on the current state and static facts (checking ingredients
            if necessary). If it is suitable, calculate the cost to get it onto a
            tray at the child's table:
            -   If already on a tray at the child's table: Cost = 0.
            -   If on a tray in the kitchen: Cost = 3 (pick tray, move tray, put tray).
            -   If not on a tray: Cost = 5 (get tray, put S on tray, pick tray, move tray, put tray).
            Keep track of the minimum transport cost among all existing suitable sandwiches.
        b.  **Building a new suitable sandwich:** Estimate the cost to build a
            ready and valid sandwich on a tray in the kitchen (Cost = 6: get tray,
            place empty S, get bread, put bread, get content, put content). Then,
            add the cost to move this tray to the child's table (Cost = 3). Total
            build+transport cost = 9. This option is only possible if the necessary
            ingredients (including gluten-free if needed for an allergic child) are
            available in the kitchen in the current state. If ingredients are missing,
            this cost is considered infinite.
        The cost for this step (`cost_to_get_sandw_to_table`) is the minimum of the
        cost using the best existing sandwich (if any) and the cost of building a
        new one (if possible). If neither is possible (e.g., no suitable existing
        sandwich and cannot build one), the problem is unsolvable for this child,
        and the heuristic returns infinity for the entire state.

    3.  **Estimate robot actions at the table:** Assuming a suitable sandwich is
        now on a tray at the child's table, the robot needs to perform the final
        steps. Calculate the cost for the robot to:
        -   Move to the child's table (Cost = 1 if not already there, based on
            the robot's current location).
        -   Ensure it is free (Cost = 1 if holding something else).
        -   Pick up the sandwich from the tray (Cost = 1).
        -   Perform the serve action (Cost = 1).
        The total cost for robot actions is the sum of these individual costs.

    4.  **Sum costs per child:** The total estimated cost for the child is the
        sum of the minimum sandwich transport cost (from step 2) and the robot
        action costs (from step 3).

    The overall heuristic value is the sum of these estimated costs for all
    unserved children.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.task = task # Store task to access goals later
        self.goals = task.goals
        self.static_facts = task.static

        # Extract static information
        self.child_allergies = {
            get_parts(fact)[1] for fact in self.static_facts if match(fact, "allergic_gluten", "*")
        }
        self.gf_breads = {
            get_parts(fact)[1] for fact in self.static_facts if match(fact, "no_gluten_bread", "*")
        }
        self.gf_contents = {
            get_parts(fact)[1] for fact in self.static_facts if match(fact, "no_gluten_content", "*")
        }
        self.child_tables = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in self.static_facts
            if match(fact, "waiting", "*", "*")
        }

        # Collect all potential sandwich objects mentioned in the problem
        self.all_sandwiches = set()
        relevant_predicates = {"notexist", "ontray", "has_bread", "has_content",
                               "is_gluten_free", "is_ready_to_serve", "is_valid",
                               "at_kitchen_sandwich"} # Include at_kitchen_sandwich

        def collect_sandwiches_from_facts(facts):
            sandwiches = set()
            for fact in facts:
                parts = get_parts(fact)
                if len(parts) > 1 and parts[0] in relevant_predicates:
                    if parts[0] == "is_valid": # (is_valid sandw child)
                        sandwiches.add(parts[1])
                    else: # (predicate sandw ...)
                        sandwiches.add(parts[1])
            return sandwiches

        self.all_sandwiches.update(collect_sandwiches_from_facts(task.initial_state))
        self.all_sandwiches.update(collect_sandwiches_from_facts(task.goals))
        self.all_sandwiches.update(collect_sandwiches_from_facts(self.static_facts))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        unserved_children = [
            get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*") and goal not in state
        ]

        if not unserved_children:
            return 0 # Goal reached

        # Find robot's current location and what it's holding
        robot_loc = 'kitchen' # Default assumption if not specified
        robot_holding = None
        for fact in state:
             if match(fact, "robot_at", "*"):
                  robot_loc = get_parts(fact)[2]
             if match(fact, "holding", "*"):
                  robot_holding = get_parts(fact)[2]

        # Pre-compute sandwich properties in the current state for efficiency
        sandw_properties = {}
        for sandw in self.all_sandwiches:
             props = {}
             props['is_ready'] = f"(is_ready_to_serve {sandw})" in state
             if not props['is_ready']:
                  has_b = any(match(fact, "has_bread", sandw, "*") for fact in state)
                  has_c = any(match(fact, "has_content", sandw, "*") for fact in state)
                  props['is_ready'] = has_b and has_c

             props['is_gf'] = f"(is_gluten_free {sandw})" in state
             if not props['is_gf']:
                  current_bread = None
                  current_content = None
                  for fact in state:
                       if match(fact, "has_bread", sandw, "*"):
                            current_bread = get_parts(fact)[2]
                       if match(fact, "has_content", sandw, "*"):
                            current_content = get_parts(fact)[2]
                  if current_bread and current_content:
                       bread_is_gf = f"(no_gluten_bread {current_bread})" in self.static_facts
                       content_is_gf = f"(no_gluten_content {current_content})" in self.static_facts
                       props['is_gf'] = bread_is_gf and content_is_gf
                  else:
                       props['is_gf'] = False # Cannot be GF if ingredients unknown/missing

             props['on_tray'] = False
             props['tray_obj'] = None
             props['tray_loc'] = None
             for fact in state:
                  if match(fact, "ontray", sandw, "*"):
                       props['on_tray'] = True
                       props['tray_obj'] = get_parts(fact)[2]
                       break
             if props['on_tray']:
                  for fact in state:
                       if match(fact, "at", props['tray_obj'], "*"):
                            props['tray_loc'] = get_parts(fact)[2]
                            break
                  if props['tray_loc'] is None:
                       # Tray location not specified, assume kitchen?
                       # Check if it's at any known table from static facts
                       is_at_known_table = False
                       for table in self.child_tables.values():
                            if f"(at {props['tray_obj']} {table})" in state:
                                 is_at_known_table = True
                                 props['tray_loc'] = table
                                 break
                       if not is_at_known_table:
                            props['tray_loc'] = 'kitchen' # Default if not at a known table


             sandw_properties[sandw] = props

        # Check ingredient availability in kitchen in the current state
        any_bread_in_kitchen = any(match(f, "at_kitchen_bread", "*") for f in state)
        any_content_in_kitchen = any(match(f, "at_kitchen_content", "*") for f in state)
        any_gf_bread_in_kitchen = any(match(f, "at_kitchen_bread", b) for f in state for b in self.gf_breads)
        any_gf_content_in_kitchen = any(match(f, "at_kitchen_content", c) for f in state for c in self.gf_contents)


        total_heuristic = 0

        for child in unserved_children:
            child_cost = 0
            child_table = self.child_tables.get(child)
            if child_table is None:
                 # Should not happen in valid problems, but handle defensively
                 return float('inf')

            is_allergic = child in self.child_allergies

            # --- Step 2: Estimate cost to get a suitable sandwich to the child's table ---
            min_sandw_transport_cost = float('inf')

            # Option a: Using an existing suitable sandwich
            for sandw, props in sandw_properties.items():
                 is_valid = props['is_ready'] and (not is_allergic or props['is_gf'])

                 if is_valid:
                      cost_this_sandw_transport = 0
                      if props['on_tray']:
                           if props['tray_loc'] == child_table:
                                cost_this_sandw_transport = 0 # Already at table
                           else: # Assume kitchen if not at table
                                cost_this_sandw_transport = 3 # Move tray (pick, move, put)
                      else: # Not on tray
                           cost_this_sandw_transport = 2 # Put on tray (get tray, put S)
                           cost_this_sandw_transport += 3 # Move tray (pick, move, put)
                           # Total = 5

                      min_sandw_transport_cost = min(min_sandw_transport_cost, cost_this_sandw_transport)

            # Option b: Building a new suitable sandwich
            can_build_valid_sandwich = False
            if not is_allergic:
                 # Need any bread and any content in kitchen
                 if any_bread_in_kitchen and any_content_in_kitchen:
                      can_build_valid_sandwich = True
            elif is_allergic:
                 # Need GF bread and GF content in kitchen
                 if any_gf_bread_in_kitchen and any_gf_content_in_kitchen:
                      can_build_valid_sandwich = True

            build_and_transport_cost = float('inf')
            if can_build_valid_sandwich:
                 # Cost to build ready+valid on tray in kitchen (6) + move tray (3)
                 build_and_transport_cost = 6 + 3 # = 9

            # Cost for this child is the minimum transport cost (either existing or new build)
            cost_to_get_sandw_to_table = min(min_sandw_transport_cost, build_and_transport_cost)

            # If cost_to_get_sandw_to_table is still infinity, this child cannot be served.
            if cost_to_get_sandw_to_table == float('inf'):
                return float('inf') # Problem is unsolvable from this state

            child_cost += cost_to_get_sandw_to_table

            # --- Step 3: Estimate robot actions at the table ---
            # Cost to get robot to table
            if robot_loc != child_table:
                child_cost += 1 # move robot

            # Cost to pick up sandwich (assuming robot is now at table and sandwich is there)
            # This pick action requires the robot to be free.
            if robot_holding is not None: # Robot is holding something
                child_cost += 1 # drop what's held
            child_cost += 1 # pick up S

            # Cost to serve
            child_cost += 1 # serve

            total_heuristic += child_cost

        return total_heuristic
