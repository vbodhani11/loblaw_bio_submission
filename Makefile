.PHONY: setup pipeline dashboard test clean

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

pipeline:
	python load_data.py
	python analysis.py

dashboard:
	streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501

test:
	pytest -q

clean:
	rm -f clinical_trial.db
	rm -f outputs/*.csv outputs/*.png outputs/*.json outputs/*.txt
