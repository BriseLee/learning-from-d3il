python run.py --config-name=picking_config \
              --multirun seed=0,1 \
              agents=act_agent \
              agent_name=act \
              window_size=3 \
              group=aligning_act_seeds \
              simulation.n_cores=30 \
              simulation.n_contexts=30 \
              simulation.n_trajectories_per_context=16