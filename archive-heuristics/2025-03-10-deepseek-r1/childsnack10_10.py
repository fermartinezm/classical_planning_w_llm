from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class Childsnack10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnack domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children by considering the steps needed to make sandwiches, place them on trays, move trays to the correct locations, and serve the children.

    # Assumptions
    - All required bread and content portions are available to make the necessary sandwiches.
    - Trays can be moved freely, but moving a tray to the kitchen is required to place sandwiches on it.
    - Each sandwich serves one child.

    # Heuristic Initialization
    - Extract static information about allergic and non-allergic children from the task's static facts.
    - Identify the goal children that need to be served.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved child:
        a. If allergic:
            i. Check for a gluten-free sandwich on a tray at their location. If found, cost +=1.
            ii. Else, check for a gluten-free sandwich on any tray. If found, cost +=2 (move tray and serve).
            iii. Else, check for a gluten-free sandwich made but not on a tray. If found, cost +=3 or 4 (depending on tray availability at kitchen).
            iv. Else, cost +=4 or 5 (make sandwich, place on tray, move, serve).
        b. If non-allergic:
            i. Check for any sandwich on a tray at their location. If found, cost +=1.
            ii. Else, check for any sandwich on any tray. If found, cost +=2.
            iii. Else, check for any sandwich made but not on a tray. If found, cost +=3 or 4.
            iv. Else, cost +=4 or 5.
    2. Sum the costs for all unserved children to get the heuristic value.
    """

    def __init__(self, task):
        self.allergic_children = set()
        self.non_allergic_children = set()
        self.goal_children = set()

        # Extract allergic and non-allergic children from static facts
        for fact in task.static:
            parts = self.get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.non_allergic_children.add(parts[1])

        # Extract goal children
        for goal in task.goals:
            parts = self.get_parts(goal)
            if parts[0] == 'served':
                self.goal_children.add(parts[1])

    def get_parts(self, fact):
        return fact.strip('()').split()

    def __call__(self, node):
        state = node.state
        served = set()
        waiting = {}
        ontray = {}
        tray_locations = {}
        gluten_free_sandwiches = set()
        made_sandwiches = set()

        # Parse current state
        for fact in state:
            parts = self.get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'served':
                served.add(parts[1])
            elif predicate == 'waiting':
                waiting[parts[1]] = parts[2]
            elif predicate == 'ontray':
                ontray[parts[1]] = parts[2]
            elif predicate == 'at' and parts[1].startswith('tray'):
                tray_locations[parts[1]] = parts[2]
            elif predicate == 'no_gluten_sandwich':
                gluten_free_sandwiches.add(parts[1])
            elif predicate == 'at_kitchen_sandwich':
                made_sandwiches.add(parts[1])

        # Check if any trays are at kitchen
        available_trays_at_kitchen = any(
            loc == 'kitchen' for loc in tray_locations.values()
        )

        total_cost = 0

        for child in self.goal_children:
            if child in served:
                continue
            if child not in waiting:
                continue  # Child not waiting, skip
            target_place = waiting[child]

            if child in self.allergic_children:
                # Handle allergic child
                steps = 0
                found = False

                # Check for existing suitable sandwich at target_place
                for s in gluten_free_sandwiches:
                    tray = ontray.get(s)
                    if tray and tray_locations.get(tray) == target_place:
                        steps = 1
                        found = True
                        break
                if found:
                    total_cost += steps
                    continue

                # Check for any gluten-free sandwich on any tray
                for s in gluten_free_sandwiches:
                    if s in ontray:
                        steps = 2  # move tray + serve
                        found = True
                        break
                if found:
                    total_cost += steps
                    continue

                # Check for made gluten-free sandwiches not on tray
                for s in gluten_free_sandwiches:
                    if s in made_sandwiches and s not in ontray:
                        if available_trays_at_kitchen:
                            steps = 3  # put_on_tray + move + serve
                        else:
                            steps = 4  # move tray to kitchen + put + move + serve
                        found = True
                        break
                if found:
                    total_cost += steps
                    continue

                # Need to make new gluten-free sandwich
                if available_trays_at_kitchen:
                    steps = 4  # make + put + move + serve
                else:
                    steps = 5  # move tray + make + put + move + serve
                total_cost += steps

            else:
                # Handle non-allergic child
                steps = 0
                found = False

                # Check for any sandwich at target_place
                for s, tray in ontray.items():
                    if tray_locations.get(tray) == target_place:
                        steps = 1
                        found = True
                        break
                if found:
                    total_cost += steps
                    continue

                # Check for any sandwich on any tray
                if ontray:
                    steps = 2  # move tray + serve
                    found = True
                if found:
                    total_cost += steps
                    continue

                # Check for made sandwiches not on tray
                if made_sandwiches:
                    if available_trays_at_kitchen:
                        steps = 3  # put + move + serve
                    else:
                        steps = 4  # move tray + put + move + serve
                    found = True
                if found:
                    total_cost += steps
                    continue

                # Need to make new sandwich
                if available_trays_at_kitchen:
                    steps = 4  # make + put + move + serve
                else:
                    steps = 5  # move tray + make + put + move + serve
                total_cost += steps

        return total_cost
