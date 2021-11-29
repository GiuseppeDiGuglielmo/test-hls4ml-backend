SCRIPT_PY := test_backends.py

help:
	@echo "INFO: make<TAB> to show targets"
.PHONY: help

run:
	python $(SCRIPT_PY)
.PHONY: run	

run-profile:
	python $(SCRIPT_PY) profile
.PHONY: run-profile	

vivado-gui:
	vivado ./test_axi_m_backend/myproject_vivado_accelerator/project_1.xpr
.PHONY: vivado-gui

clean:
	rm -rf training_dir
	rm -rf __pycache__
	rm -rf axi_m_backend
	rm -f axi_m_backend.tar.gz
	rm -f *.npy
	rm -f *.log
	rm -f *.jou
	rm -f *.str
	rm -rf NA
.PHONY: clean
