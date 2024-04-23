import ray



def objective(config):  # ①
    score = config["a"] ** 2 + config["b"]
    return {"score": score}


search_space = {  # ②
    "a": ray.tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    "b": ray.tune.choice([1, 2, 3]),
}

tuner = ray.tune.Tuner(objective, param_space=search_space)  # ③

results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)