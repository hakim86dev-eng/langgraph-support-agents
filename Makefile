.PHONY: install lint format typecheck test run docker

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy src tests

test:
	pytest --cov --cov-report=term-missing

run:
	uvicorn support_agents.main:app --reload

docker:
	docker compose up --build
