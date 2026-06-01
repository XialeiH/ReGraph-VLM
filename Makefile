.PHONY: preflight compile bundle-check bundle manuscript-audit parameter-counts

preflight:
	python3 scripts/run_publication_preflight.py

compile:
	python3 scripts/run_publication_preflight.py --compile

bundle-check:
	python3 scripts/make_anonymous_submission_bundle.py --dry-run

bundle:
	python3 scripts/make_anonymous_submission_bundle.py

manuscript-audit:
	python3 scripts/audit_manuscript_publication_claims.py \
	  --tex reports/neurips_report/may30.tex \
	  --manuscript-only \
	  --output-dir /tmp/regraph_report_preflight

parameter-counts:
	python3 scripts/verify_model_parameter_counts.py
