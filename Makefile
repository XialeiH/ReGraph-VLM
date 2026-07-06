.PHONY: syntax parameter-counts

syntax:
	python3 -m compileall models scripts

parameter-counts:
	python3 scripts/verify_model_parameter_counts.py
