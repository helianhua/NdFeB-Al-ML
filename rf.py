import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from bayes_opt import BayesianOptimization
import os
import time
import warnings
warnings.filterwarnings("ignore")

# === Configuration Parameters ===
INIT_TRAIN_SIZE = 32
ACTIVE_ROUNDS = 13
BUDGET_PER_ROUND = 120
SAMPLE_COST = 1
LAMBDA_REP = 0.7
N_REPEATS = 5
RANDOM_SEEDS = [42, 1337, 2025, 99, 101]

BO_PARAM_BOUNDS = {
    'max_depth': (3, 20),
    'n_estimators': (50, 300)
}


def bo_evaluate(max_depth, n_estimators, X_train, y_train, X_val, y_val):
    max_depth, n_estimators = int(round(max_depth)), int(round(n_estimators))
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return r2_score(y_val, model.predict(X_val))

def calc_uncertainty(model, X):
    all_preds = np.stack([tree.predict(X) for tree in model.estimators_], axis=0)
    return np.var(all_preds, axis=0)

def select_samples_with_budget(X_pool, X_train, uncertainty, budget_per_round, sample_cost=1, lambda_rep=0.7):
    train_center = X_train.mean(axis=0)
    distances = np.linalg.norm(X_pool - train_center, axis=1)
    norm_dist = distances / distances.max() if distances.max() > 0 else distances
    scores = lambda_rep * uncertainty + (1 - lambda_rep) * norm_dist
    sorted_idx = np.argsort(-scores)
    num_samples_to_select = budget_per_round // sample_cost
    return sorted_idx[:num_samples_to_select].tolist()

def evaluate_performance(y_true, y_pred):
    return r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

def select_samples_randomly(X_pool, budget_per_round, sample_cost=1):
    num_samples_to_select = budget_per_round // sample_cost
    if len(X_pool) <= num_samples_to_select: return np.arange(len(X_pool))
    return np.random.choice(np.arange(len(X_pool)), size=num_samples_to_select, replace=False)

# === Core Learning Loop (Returns all intermediate states) ===
def run_learning_process(strategy, X_train_base, y_train_base, X_pool_base, y_pool_base, X_val, y_val, X_test, y_test):
    X_train, y_train, X_pool, y_pool = X_train_base.copy(), y_train_base.copy(), X_pool_base.copy(), y_pool_base.copy()
    test_metrics_list, train_size_list = [], []
    all_captured_states = {} 

    for round_i in range(ACTIVE_ROUNDS):
        train_size_list.append(len(X_train))
        def bo_target(max_depth, n_estimators): return bo_evaluate(max_depth, n_estimators, X_train, y_train, X_val, y_val)
        optimizer = BayesianOptimization(f=bo_target, pbounds=BO_PARAM_BOUNDS, random_state=round_i*10, verbose=0)
        optimizer.maximize(init_points=3, n_iter=5)
        best_params = optimizer.max['params']
        model = RandomForestRegressor(max_depth=int(round(best_params['max_depth'])), n_estimators=int(round(best_params['n_estimators'])), random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        test_metrics_list.append(evaluate_performance(y_test, model.predict(X_test)))


        all_captured_states[round_i] = {'model': model, 'X_train': X_train.copy(), 'y_train': y_train.copy()}
        
        if len(X_pool) == 0: break
        
        if strategy == 'Active Learning':
            selected_idx = select_samples_with_budget(X_pool, X_train, calc_uncertainty(model, X_pool), BUDGET_PER_ROUND, SAMPLE_COST, LAMBDA_REP)
        else: # Random Learning
            selected_idx = select_samples_randomly(X_pool, BUDGET_PER_ROUND, SAMPLE_COST)
            
        X_selected, y_selected = X_pool[selected_idx], y_pool[selected_idx]
        X_train, y_train = np.vstack([X_train, X_selected]), np.concatenate([y_train, y_selected])
        X_pool, y_pool = np.delete(X_pool, selected_idx, axis=0), np.delete(y_pool, selected_idx, axis=0)
    
    return test_metrics_list, train_size_list, all_captured_states

def main_robustness_workflow(data_path, results_save_path="model_output/br_rf_robustness_results.joblib"):
    data = pd.read_csv(data_path); data.fillna(data.median(), inplace=True)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]; scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    os.makedirs('model_output', exist_ok=True)

    all_run_results = {'Active Learning': [], 'Random Learning': []}
    full_dataset_metrics_scores = []
    representative_run_data = {}
    final_train_sizes = None

    for i, seed in enumerate(RANDOM_SEEDS):
        print(f"\n{'='*20}\n Running Experiment Repeat {i+1}/{N_REPEATS} (Seed: {seed}) \n{'='*20}")
        X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=seed)
        X_pool_base, X_val, y_pool_base, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=seed)
        X_train_base, X_pool_base, y_train_base, y_pool_base = train_test_split(
            X_pool_base, y_pool_base, train_size=INIT_TRAIN_SIZE, random_state=seed)
        
        print("--- Tuning hyperparameters for Full Dataset model for this split ---")
        def bo_full_model(max_depth, n_estimators):
            X_bo_train, _, y_bo_train, _ = train_test_split(X_temp, y_temp, test_size=0.1, random_state=seed)
            return bo_evaluate(max_depth, n_estimators, X_bo_train, y_bo_train, X_val, y_val)
        full_optimizer = BayesianOptimization(f=bo_full_model, pbounds=BO_PARAM_BOUNDS, random_state=seed, verbose=0)
        full_optimizer.maximize(init_points=5, n_iter=10)
        best_full_params = full_optimizer.max['params']
        full_model = RandomForestRegressor(max_depth=int(round(best_full_params['max_depth'])), n_estimators=int(round(best_full_params['n_estimators'])), random_state=42, n_jobs=-1)
        full_model.fit(X_temp, y_temp)
        full_metrics = evaluate_performance(y_test, full_model.predict(X_test))
        full_dataset_metrics_scores.append(full_metrics)
        print(f"Optimized Full Dataset R² for this split: {full_metrics[0]:.4f}")

        for strategy in ['Active Learning', 'Random Learning']:
            print(f"\n--- Strategy: {strategy} ---")
            metrics_curve, size_curve, all_states = run_learning_process(
                strategy, X_train_base, y_train_base, X_pool_base, y_pool_base, X_val, y_val, X_test, y_test)
            all_run_results[strategy].append(metrics_curve)
            final_train_sizes = size_curve
            
            if i == 0 and strategy == 'Active Learning':
                representative_run_data['all_captured_al_states'] = all_states
        
        if i == 0:
            representative_run_data['full_model'] = full_model
            representative_run_data['data'] = {'X_temp': X_temp, 'y_temp': y_temp, 'X_test': X_test, 'y_test': y_test}

    results_to_save = {
        'all_run_results': all_run_results,
        'train_sizes': final_train_sizes,
        'full_dataset_metrics_scores': full_dataset_metrics_scores,
        'n_repeats': N_REPEATS,
        'representative_run': representative_run_data
    }
    joblib.dump(results_to_save, results_save_path)
    print(f"\n All robustness results successfully saved to '{results_save_path}'.")

if __name__ == "__main__":
    main_robustness_workflow("Br.csv")
