from string import Template

REORDERED_PROMPT = Template("""
<problem-description>
You are a highly-skilled professor in AI planning and a proficient Python programmer creating a domain-dependent heuristic function for the PDDL domain <domain>$name</domain>. The heuristic function you create will be used to guide a greedy best-first search to solve instances from this domain. Therefore, the heuristic does not need to be admissible. For a given state, the heuristic function should estimate the required number of actions to reach a goal state as accurately as possible, while remaining efficiently computable. The name of the heuristic should be $heuristic_name. The heuristic should be efficiently computable, and it should minimize the number of expanded nodes during the search. Next, you will receive a sequence of file contents to help you with your task and to show you the definition of the $name domain.
</problem-description>

This is the PDDL domain file of the $name domain, for which you need to create a domain-dependent heuristic:
<domain-file>
$domain
</domain-file>

This is an example of a PDDL instance file of the $name domain:
<instance-file-example-1>
$instance1
</instance-file-example-1>

This is a second example of a PDDL instance file of the $name domain:
<instance-file-example-2>
$instance2
</instance-file-example-2>

This is the PDDL domain file of another domain, called Gripper, which serves as an example:
<gripper-domain-file>
(define (domain gripper-strips)
   (:predicates (room ?r)
		(ball ?b)
		(gripper ?g)
		(at-robby ?r)
		(at ?b ?r)
		(free ?g)
		(carry ?o ?g))

   (:action move
       :parameters  (?from ?to)
       :precondition (and  (room ?from) (room ?to) (at-robby ?from))
       :effect (and  (at-robby ?to)
		     (not (at-robby ?from))))

   (:action pick
       :parameters (?obj ?room ?gripper)
       :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
			    (at ?obj ?room) (at-robby ?room) (free ?gripper))
       :effect (and (carry ?obj ?gripper)
		    (not (at ?obj ?room))
		    (not (free ?gripper))))

   (:action drop
       :parameters  (?obj  ?room ?gripper)
       :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
			    (carry ?obj ?gripper) (at-robby ?room))
       :effect (and (at ?obj ?room)
		    (free ?gripper)
		    (not (carry ?obj ?gripper)))))
</gripper-domain-file>

This is an example of an instance file from the Gripper domain:
<gripper-instance-file-example>
(define (problem strips-gripper-x-20)
   (:domain gripper-strips)
   (:objects rooma roomb ball42 ball41 ball40 ball39 ball38 ball37
             ball36 ball35 ball34 ball33 ball32 ball31 ball30 ball29 ball28
             ball27 ball26 ball25 ball24 ball23 ball22 ball21 ball20 ball19
             ball18 ball17 ball16 ball15 ball14 ball13 ball12 ball11 ball10
             ball9 ball8 ball7 ball6 ball5 ball4 ball3 ball2 ball1 left right)
   (:init (room rooma)
          (room roomb)
          (ball ball42)
          (ball ball41)
          (ball ball40)
          (ball ball39)
          (ball ball38)
          (ball ball37)
          (ball ball36)
          (ball ball35)
          (ball ball34)
          (ball ball33)
          (ball ball32)
          (ball ball31)
          (ball ball30)
          (ball ball29)
          (ball ball28)
          (ball ball27)
          (ball ball26)
          (ball ball25)
          (ball ball24)
          (ball ball23)
          (ball ball22)
          (ball ball21)
          (ball ball20)
          (ball ball19)
          (ball ball18)
          (ball ball17)
          (ball ball16)
          (ball ball15)
          (ball ball14)
          (ball ball13)
          (ball ball12)
          (ball ball11)
          (ball ball10)
          (ball ball9)
          (ball ball8)
          (ball ball7)
          (ball ball6)
          (ball ball5)
          (ball ball4)
          (ball ball3)
          (ball ball2)
          (ball ball1)
          (at-robby rooma)
          (free left)
          (free right)
          (at ball42 rooma)
          (at ball41 rooma)
          (at ball40 rooma)
          (at ball39 rooma)
          (at ball38 rooma)
          (at ball37 rooma)
          (at ball36 rooma)
          (at ball35 rooma)
          (at ball34 rooma)
          (at ball33 rooma)
          (at ball32 rooma)
          (at ball31 rooma)
          (at ball30 rooma)
          (at ball29 rooma)
          (at ball28 rooma)
          (at ball27 rooma)
          (at ball26 rooma)
          (at ball25 rooma)
          (at ball24 rooma)
          (at ball23 rooma)
          (at ball22 rooma)
          (at ball21 rooma)
          (at ball20 rooma)
          (at ball19 rooma)
          (at ball18 rooma)
          (at ball17 rooma)
          (at ball16 rooma)
          (at ball15 rooma)
          (at ball14 rooma)
          (at ball13 rooma)
          (at ball12 rooma)
          (at ball11 rooma)
          (at ball10 rooma)
          (at ball9 rooma)
          (at ball8 rooma)
          (at ball7 rooma)
          (at ball6 rooma)
          (at ball5 rooma)
          (at ball4 rooma)
          (at ball3 rooma)
          (at ball2 rooma)
          (at ball1 rooma)
          (gripper left)
          (gripper right))
   (:goal (and (at ball42 roomb)
               (at ball41 roomb)
               (at ball40 roomb)
               (at ball39 roomb)
               (at ball38 roomb)
               (at ball37 roomb)
               (at ball36 roomb)
               (at ball35 roomb)
               (at ball34 roomb)
               (at ball33 roomb)
               (at ball32 roomb)
               (at ball31 roomb)
               (at ball30 roomb)
               (at ball29 roomb)
               (at ball28 roomb)
               (at ball27 roomb)
               (at ball26 roomb)
               (at ball25 roomb)
               (at ball24 roomb)
               (at ball23 roomb)
               (at ball22 roomb)
               (at ball21 roomb)
               (at ball20 roomb)
               (at ball19 roomb)
               (at ball18 roomb)
               (at ball17 roomb)
               (at ball16 roomb)
               (at ball15 roomb)
               (at ball14 roomb)
               (at ball13 roomb)
               (at ball12 roomb)
               (at ball11 roomb)
               (at ball10 roomb)
               (at ball9 roomb)
               (at ball8 roomb)
               (at ball7 roomb)
               (at ball6 roomb)
               (at ball5 roomb)
               (at ball4 roomb)
               (at ball3 roomb)
               (at ball2 roomb)
               (at ball1 roomb))))
</gripper-instance-file-example>

This is an example of a domain-dependent heuristic for Gripper:
<code-file-heuristic-1>
$heuristic1
</code-file-heuristic-1>

This is the PDDL domain file of another domain, called Logistics, to serve as a second example:
<logistics-domain-file>
(define (domain logistics-strips)
  (:requirements :strips)
  (:predicates 	(OBJ ?obj)
	       	(TRUCK ?truck)
               	(LOCATION ?loc)
		(AIRPLANE ?airplane)
                (CITY ?city)
                (AIRPORT ?airport)
		(at ?obj ?loc)
		(in ?obj1 ?obj2)
		(in-city ?obj ?city))

(:action LOAD-TRUCK
  :parameters
   (?obj
    ?truck
    ?loc)
  :precondition
   (and (OBJ ?obj) (TRUCK ?truck) (LOCATION ?loc)
   (at ?truck ?loc) (at ?obj ?loc))
  :effect
   (and (not (at ?obj ?loc)) (in ?obj ?truck)))

(:action LOAD-AIRPLANE
  :parameters
   (?obj
    ?airplane
    ?loc)
  :precondition
   (and (OBJ ?obj) (AIRPLANE ?airplane) (LOCATION ?loc)
   (at ?obj ?loc) (at ?airplane ?loc))
  :effect
   (and (not (at ?obj ?loc)) (in ?obj ?airplane)))

(:action UNLOAD-TRUCK
  :parameters
   (?obj
    ?truck
    ?loc)
  :precondition
   (and (OBJ ?obj) (TRUCK ?truck) (LOCATION ?loc)
        (at ?truck ?loc) (in ?obj ?truck))
  :effect
   (and (not (in ?obj ?truck)) (at ?obj ?loc)))

(:action UNLOAD-AIRPLANE
  :parameters
   (?obj
    ?airplane
    ?loc)
  :precondition
   (and (OBJ ?obj) (AIRPLANE ?airplane) (LOCATION ?loc)
        (in ?obj ?airplane) (at ?airplane ?loc))
  :effect
   (and (not (in ?obj ?airplane)) (at ?obj ?loc)))

(:action DRIVE-TRUCK
  :parameters
   (?truck
    ?loc-from
    ?loc-to
    ?city)
  :precondition
   (and (TRUCK ?truck) (LOCATION ?loc-from) (LOCATION ?loc-to) (CITY ?city)
   (at ?truck ?loc-from)
   (in-city ?loc-from ?city)
   (in-city ?loc-to ?city))
  :effect
   (and (not (at ?truck ?loc-from)) (at ?truck ?loc-to)))

(:action FLY-AIRPLANE
  :parameters
   (?airplane
    ?loc-from
    ?loc-to)
  :precondition
   (and (AIRPLANE ?airplane) (AIRPORT ?loc-from) (AIRPORT ?loc-to)
	(at ?airplane ?loc-from))
  :effect
   (and (not (at ?airplane ?loc-from)) (at ?airplane ?loc-to)))
)
</logistics-domain-file>

This is an example of an instance file from the Logistics domain:
<logistics-instance-file-example>
(define (problem strips-log-y-5)
   (:domain logistics-strips)
   (:objects package5 package4 package3 package2 package1 city8
             city7 city6 city5 city4 city3 city2 city1 truck15 truck14
             truck13 truck12 truck11 truck10 truck9 truck8 truck7 truck6
             truck5 truck4 truck3 truck2 truck1 plane1 city8-2 city8-1
             city7-2 city7-1 city6-2 city6-1 city5-2 city5-1 city4-2
             city4-1 city3-2 city3-1 city2-2 city2-1 city1-2 city1-1
             city8-3 city7-3 city6-3 city5-3 city4-3 city3-3 city2-3
             city1-3)
   (:init (obj package5)
          (obj package4)
          (obj package3)
          (obj package2)
          (obj package1)
          (city city8)
          (city city7)
          (city city6)
          (city city5)
          (city city4)
          (city city3)
          (city city2)
          (city city1)
          (truck truck15)
          (truck truck14)
          (truck truck13)
          (truck truck12)
          (truck truck11)
          (truck truck10)
          (truck truck9)
          (truck truck8)
          (truck truck7)
          (truck truck6)
          (truck truck5)
          (truck truck4)
          (truck truck3)
          (truck truck2)
          (truck truck1)
          (airplane plane1)
          (location city8-2)
          (location city8-1)
          (location city7-2)
          (location city7-1)
          (location city6-2)
          (location city6-1)
          (location city5-2)
          (location city5-1)
          (location city4-2)
          (location city4-1)
          (location city3-2)
          (location city3-1)
          (location city2-2)
          (location city2-1)
          (location city1-2)
          (location city1-1)
          (airport city8-3)
          (location city8-3)
          (airport city7-3)
          (location city7-3)
          (airport city6-3)
          (location city6-3)
          (airport city5-3)
          (location city5-3)
          (airport city4-3)
          (location city4-3)
          (airport city3-3)
          (location city3-3)
          (airport city2-3)
          (location city2-3)
          (airport city1-3)
          (location city1-3)
          (in-city city8-3 city8)
          (in-city city8-2 city8)
          (in-city city8-1 city8)
          (in-city city7-3 city7)
          (in-city city7-2 city7)
          (in-city city7-1 city7)
          (in-city city6-3 city6)
          (in-city city6-2 city6)
          (in-city city6-1 city6)
          (in-city city5-3 city5)
          (in-city city5-2 city5)
          (in-city city5-1 city5)
          (in-city city4-3 city4)
          (in-city city4-2 city4)
          (in-city city4-1 city4)
          (in-city city3-3 city3)
          (in-city city3-2 city3)
          (in-city city3-1 city3)
          (in-city city2-3 city2)
          (in-city city2-2 city2)
          (in-city city2-1 city2)
          (in-city city1-3 city1)
          (in-city city1-2 city1)
          (in-city city1-1 city1)
          (at plane1 city3-3)
          (at truck15 city8-2)
          (at truck14 city7-2)
          (at truck13 city6-1)
          (at truck12 city5-2)
          (at truck11 city4-1)
          (at truck10 city3-2)
          (at truck9 city2-1)
          (at truck8 city1-2)
          (at truck7 city6-1)
          (at truck6 city3-2)
          (at truck5 city1-3)
          (at truck4 city4-3)
          (at truck3 city1-3)
          (at truck2 city7-1)
          (at truck1 city8-1)
          (at package5 city3-2)
          (at package4 city5-1)
          (at package3 city1-1)
          (at package2 city5-2)
          (at package1 city2-1))
   (:goal (and (at package5 city6-3)
               (at package4 city5-3)
               (at package3 city8-3)
               (at package2 city4-3)
               (at package1 city6-3))))
</logistics-instance-file-example>

This is an example of a domain-dependent heuristic for Logistics:
<code-file-heuristic-2>
$heuristic2
</code-file-heuristic-2>

This is how an example state from the $name domain is represented internally by the planner. Note that PDDL facts are represented as strings, for example '(predicate_name object1 object2)'.
<state>
$state
</state>

This is an example for how the static information is represented internally by the planner:
<static>
$static
</static>

This is the source code for representing operators and tasks in the planner:
<code-file-task>
$task
</code-file-task>

Provide only the Python code of the domain-dependent heuristic for the $name domain. Here is a checklist to help you with your code:
1) The code for extracting objects from facts remembers to ignore the surrounding brackets.
2) The heuristic is 0 only for goal states.
3) The heuristic value is finite for solvable states.
4) All used modules are imported.
5) The information from static facts is extracted into suitable data structures in the constructor.
6) Provide a detailed docstring explaining the heuristic calculation. For this, divide the docstring into sections "Summary", "Assumptions", "Heuristic Initialization" and "Step-By-Step Thinking for Computing Heuristic".
""")
