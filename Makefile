venv: venv/touchfile

venv/touchfile: requirements.txt
	if not exist .venv python -m venv .venv
	".venv/Scripts/activate" && pip install -Ur requirements.txt
	type NUL > .venv/touchfile

run: venv
	".venv/Scripts/activate" && python ./demos/app.py

test: venv
	".venv/Scripts/activate" && nosetests ./tests

clean:
	rmdir /s /q .venv