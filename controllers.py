import time

class Controller(object):
	def __init__(self, env):
		self.env = env
	
	def nb_actions(self):
		return self.env.action_space.shape[0]

	def nb_inputs(self):
		return self.env.observation_space.shape[0]
	
	def reset(self):
		pass

	def get_action(self, state):
		return NotImplementedError
	
	def get_name(self):
		return 'NAN'

class MultiStepController(Controller):
	def get_multistep_actions(self, state, nb_steps):
		return NotImplementedError
	


class RandomController(MultiStepController):
	def get_name(self):
		return 'RND'
	def get_action(self, state):
		return self.env.action_space.sample()
	def get_multistep_actions(self, state, nb_steps):
		return [self.get_action(None) for _ in range(nb_steps)], {}


class MPCcontroller(Controller):
	def __init__(self,
				 env,
				 dyn_model,
				 cost_fn=None,
				 horizon=10,
				 num_simulated_paths=1000,
				 num_mpc_steps=4,
				):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths
		self.num_mpc_steps = min(num_mpc_steps, horizon)
		self.planned_actions = iter([])
	
	def get_name(self):
		return 'MPC'
	
	def plan(self, state, horizon):
		best_plan = None
		best_cost = None
		for _ in range(self.num_simulated_paths):
			c_state = state
			actions = []
			predictions = []
			cost = 0
			for _ in range(horizon):
				actions.append(self.env.action_space.sample())
				new_state = self.dyn_model(c_state, actions[-1])
				# print(self.cost_fn(c_state))
				cost += self.cost_fn(c_state, actions[-1], new_state)
				c_state = new_state
				predictions.append(new_state)
			if best_cost is None or cost < best_cost:
				best_plan = actions
				best_cost = cost
				best_plan_predictions = predictions
		return best_plan, best_cost, best_plan_predictions
	
	def reset(self):
		self.planned_actions = iter([])

	def get_action(self, state):
		try:
			return self.planned_actions.__next__()
		except StopIteration:
			best_plan, _, _ = self.plan(state, self.horizon)
			self.planned_actions = reversed(best_plan[:self.num_mpc_steps])
			return self.planned_actions.__next__()
	
	def get_multistep_actions(self, state, nb_steps):
		plan, cost, predictions = self.plan(state, max(self.horizon, nb_steps))
		return list(reversed(plan[:nb_steps])), {'cost': cost, 'predictions': list(reversed(predictions[:nb_steps]))}

