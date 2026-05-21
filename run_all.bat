@echo off

echo Running BioASQ...
python scripts/test_graph.py ^
  --file_path data/raw/bioasq/train.csv ^
  --question_col question ^
  --report_path data/raw/bioasq/report.json

echo Running BiQA...
python scripts/test_graph.py ^
  --file_path data/raw/biqa/medicalsciences_202004.csv ^
  --question_col question_text ^
  --report_path data/raw/biqa/report.json

echo Running COVIDQA...
python scripts/test_graph.py ^
  --file_path data/raw/covidqa/train.csv ^
  --question_col question ^
  --report_path data/raw/covidqa/report.json

echo Running HealthSearchQA...
python scripts/test_graph.py ^
  --file_path data/raw/healthsearchqa/train.csv ^
  --question_col question ^
  --report_path data/raw/healthsearchqa/report.json

echo Running MedAESQA...
python scripts/test_graph.py ^
  --file_path data/raw/medaesqa/train.csv ^
  --question_col question ^
  --report_path data/raw/medaesqa/report.json

echo Running MedChangeQA...
python scripts/test_graph.py ^
  --file_path data/raw/medchangeqa/MedChangeQA.csv ^
  --question_col Question ^
  --report_path data/raw/medchangeqa/report.json

echo Running MedHallu...
python scripts/test_graph.py ^
  --file_path data/raw/medhallu/train.csv ^
  --question_col Question ^
  --report_path data/raw/medhallu/report.json

echo Running MedHalt...
python scripts/test_graph.py ^
  --file_path data/raw/medhalt/train.csv ^
  --question_col question ^
  --report_path data/raw/medhalt/report.json

echo Running MedicationQA...
python scripts/test_graph.py ^
  --file_path data/raw/medicationqa/train.csv ^
  --question_col Question ^
  --report_path data/raw/medicationqa/report.json

echo Running MediQA...
python scripts/test_graph.py ^
  --file_path data/raw/mediqa/0000.parquet ^
  --question_col question ^
  --report_path data/raw/mediqa/report.json

echo Running MedMCQA...
python scripts/test_graph.py ^
  --file_path data/raw/medmcqa/train.csv ^
  --question_col question ^
  --report_path data/raw/medmcqa/report.json

echo Running MedQA...
python scripts/test_graph.py ^
  --file_path data/raw/medqa/train.csv ^
  --question_col question ^
  --report_path data/raw/medqa/report.json

echo Running MedQuAD...
python scripts/test_graph.py ^
  --file_path data/raw/medquad/train.csv ^
  --question_col question ^
  --report_path data/raw/medquad/report.json

echo Running MedRevQA...
python scripts/test_graph.py ^
  --file_path data/raw/medrevqa/MedRevQA.csv ^
  --question_col Question ^
  --report_path data/raw/medrevqa/report.json

echo Running PubMedQA...
python scripts/test_graph.py ^
  --file_path data/raw/pubmedqa/train.csv ^
  --question_col question ^
  --report_path data/raw/pubmedqa/report.json

echo Running HealthBench...
python scripts/test_graph.py ^
  --file_path data/raw/healthbench/train.csv ^
  --question_col question ^
  --report_path data/raw/healthbench/report.json

echo All datasets processed!

echo Running Plots
python .\scripts\plot_reports_metrics.py

pause