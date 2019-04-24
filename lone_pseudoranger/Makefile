MCMC_STEPS = 5
N_WALKERS  = 500

all: model_data monte_carlo
	@echo
	@echo  Finished.


monte_carlo:
	mkdir -p monte_carlo_chain_register/
	@echo Running MonteCarlo with $(MCMC_STEPS) steps and $(N_WALKERS) random walkers.
	python 01_run_montecarlo.py $(MCMC_STEPS) $(N_WALKERS)

model_data:
	mkdir -p measured_data
	@echo Modelling Satellite Doppler
	python 00_modelling_satellite_doppler_shift.py
