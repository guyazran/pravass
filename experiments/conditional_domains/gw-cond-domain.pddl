(define (domain item-sorting)
  (:requirements :strips :typing :conditional-effects :equality)
  (:types item table)
  (:predicates (on-table ?i - item ?t - table)
	           (robot-gripper-empty)
	           (robot-gripping ?i - item)
 )

  (:action pick-up
	     :parameters (?i - item ?t - table)
	     :precondition (robot-gripper-empty)
	     :effect
	     (when (on-table ?i ?t)
	       (and (not (on-table ?i ?t))
		        (not (robot-gripper-empty))
		        (robot-gripping ?i))))

  (:action put-down
	     :parameters (?i - item ?t - table)
	     :effect
	     (and (robot-gripper-empty)
	          (when (robot-gripping ?i)
	            (and (not (robot-gripping ?i))
		             (on-table ?i ?t)))))
)
