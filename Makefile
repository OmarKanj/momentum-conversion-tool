.PHONY= all

environment_unix:
	python3 -m venv .venv
	source .venv/bin/activate
	pip3 install -r requirements.txt

environment_win:
	python3 -m venv .venv
	.venv\bin\Activate.ps1
	pip3 install -r requirements.txt

