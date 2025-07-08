find lm_eval/tasks/science/advanced_reasoning/materials \
	-name '*.yaml' \
	| xargs -n1 basename -s .yaml \
	| sort -u \
	| paste -sd, - \
