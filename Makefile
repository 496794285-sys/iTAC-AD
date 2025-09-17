.PHONY: fmt lint test smoke bench release

fmt:
	black .
	ruff --fix .

lint:
	ruff .
	mypy itacad itac_eval tools

test:
	pytest -q

smoke:
	ITAC_FORCE_CPU=1 ITAC_EPOCHS=1 ITAC_MAX_STEPS=2 ITAC_SKIP_EVAL=1 \
	python main.py --model iTAC_AD --dataset synthetic --retrain

bench:
	bash scripts/bench_grid.sh && python tools/make_tables.py --csv results/all.csv

release:
	TAG=v0.1.0 CKPT=$(CKPT) bash scripts/make_release.sh
