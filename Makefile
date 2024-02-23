PYTHON=python3.10
VENV=.video_gpt_env
BIN=$(VENV)/bin
PYTHON_VENV=$(BIN)/python3

.SILENT:

$(VENV)/bin/activate: requirements.txt
	test -d $(VENV) || $(PYTHON) -m venv --upgrade-deps --prompt . $(VENV)
	$(PYTHON_VENV) -m pip install wheel
	$(PYTHON_VENV) -m pip install -r requirements.txt

init: $(VENV)/bin/activate