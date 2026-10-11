(define (domain plan_trial_t20190909_044715_250790-domain)
 (:requirements :strips :typing :negative-preconditions :disjunctive-preconditions :equality :conditional-effects :universal-preconditions)
 (:types
    object_ - object
    item receptacle rtype otype - object_
 )
 (:constants
   fridgetype sinkbasintype microwavetype - rtype
   knifetype butterknifetype - otype
 )
 (:predicates 
             (isvisibleobject ?target - item)
             (isvisiblereceptacle ?target_0 - receptacle)
             (openable ?r - receptacle)
             (opened ?r - receptacle)
             (inreceptacle ?o - item ?r - receptacle)
             (receptacletype ?r - receptacle ?t - rtype)
             (cancontain ?rt - rtype ?ot - otype)
             (objecttype ?o - item ?t_0 - otype)
             (holds ?o - item)
             (holdsany)
             (isclean ?o - item)
             (cleanable ?o - item)
             (ishot ?o - item)
             (heatable ?o - item)
             (iscool ?o - item)
             (coolable ?o - item)
             (pickupable ?o - item)
             (toggleable ?o - item)
             (istoggled ?o - item)
             (sliceable ?o - item)
             (issliced ?o - item)
 )
 (:action gotointeractableobject
  :parameters ( ?target - item)
  :precondition (and (or (cleanable ?target) (heatable ?target) (coolable ?target) (pickupable ?target) (toggleable ?target) (sliceable ?target)) (forall (?r_0 - receptacle)
 (or (not (inreceptacle ?target ?r_0)) (not (openable ?r_0)) (opened ?r_0))))
  :effect (and (isvisibleobject ?target)(forall (?r_0 - receptacle) (not (isvisiblereceptacle ?r_0)))(forall (?o_0 - item) (when (not (= ?o_0 ?target)) (not (isvisibleobject ?o_0))))))
 (:action gotointeractablereceptacle
  :parameters ( ?target_0 - receptacle)
  :effect (and (isvisiblereceptacle ?target_0)(forall (?o_0 - item) (not (isvisibleobject ?o_0)))(forall (?r_0 - receptacle) (when (not (= ?r_0 ?target_0)) (not (isvisiblereceptacle ?r_0))))))
 (:action openobject
  :parameters ( ?r - receptacle)
  :precondition (and (openable ?r))
  :effect (and (when (isvisiblereceptacle ?r) (opened ?r))))
 (:action closeobject
  :parameters ( ?r - receptacle)
  :precondition (and (openable ?r))
  :effect (and (when (isvisiblereceptacle ?r) (not (opened ?r)))))
 (:action pickupobject
  :parameters ( ?o - item)
  :precondition (and (pickupable ?o) (not (holdsany)))
  :effect (and (when (isvisibleobject ?o) (holds ?o)) (when (isvisibleobject ?o) (holdsany)) (when (isvisibleobject ?o) (not (isvisibleobject ?o)))(forall (?r_0 - receptacle) (when (inreceptacle ?o ?r_0) (not (inreceptacle ?o ?r_0))))))
 (:action putobject
  :parameters ( ?o - item ?r - receptacle ?ot - otype ?rt - rtype)
  :precondition (and (isvisiblereceptacle ?r) (or (not (openable ?r)) (opened ?r)) (objecttype ?o ?ot) (receptacletype ?r ?rt) (cancontain ?rt ?ot))
  :effect (and (when (holds ?o) (inreceptacle ?o ?r)) (when (holds ?o) (not (holds ?o))) (when (holds ?o) (not (holdsany))) (when (holds ?o) (not (isvisiblereceptacle ?r))) (when (holds ?o) (isvisibleobject ?o))))
 (:action cleanobject
  :parameters ( ?r - receptacle ?o - item)
  :precondition (and (cleanable ?o) (receptacletype ?r sinkbasintype) (isvisiblereceptacle ?r))
  :effect (and (when (holds ?o) (isclean ?o))))
 (:action heatobject
  :parameters ( ?r - receptacle ?o - item)
  :precondition (and (heatable ?o) (receptacletype ?r microwavetype) (isvisiblereceptacle ?r))
  :effect (and (when (holds ?o) (ishot ?o))))
 (:action coolobject
  :parameters ( ?r - receptacle ?o - item)
  :precondition (and (coolable ?o) (receptacletype ?r fridgetype) (isvisiblereceptacle ?r))
  :effect (and (when (holds ?o) (iscool ?o))))
 (:action toggleobject
  :parameters ( ?o - item)
  :precondition (and (toggleable ?o))
  :effect (and (when (and (isvisibleobject ?o) (istoggled ?o)) (not (istoggled ?o))) (when (and (isvisibleobject ?o) (not (istoggled ?o))) (istoggled ?o))))
 (:action sliceobject
  :parameters ( ?co - item ?ko - item ?r - receptacle)
  :precondition (and (sliceable ?co) (not (issliced ?co)) (isvisibleobject ?co) (holds ?ko) (not (= ?co ?ko)) (or (objecttype ?ko knifetype) (objecttype ?ko butterknifetype)) (inreceptacle ?co ?r) (not (receptacletype ?r sinkbasintype)) (not (receptacletype ?r fridgetype)) (not (receptacletype ?r microwavetype)))
  :effect (and (issliced ?co) (not (isvisibleobject ?co))))
)
