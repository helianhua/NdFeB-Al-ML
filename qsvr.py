import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVR
from qiskit_aer import AerSimulator
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings("ignore")


try:
    gpu_sim = AerSimulator(method="statevector", device="GPU")
    FidelityQuantumKernel._DEFAULT_SIMULATOR = gpu_sim
    print(" FidelityQuantumKernel now uses GPU backend.")
except Exception as e:
    print(f" GPU backend for AerSimulator not available, falling back to CPU. Error: {e}")


INIT_TRAIN_SIZE = 32
ACTIVE_ROUNDS = 14
BUDGET_PER_ROUND = 100
LAMBDA_REP = 0.7
MAX_FEATURES = 6
REPS = 1
PATIENCE = 3
INIT_POINTS = 2  
N_ITER = 5       

# XGBoost
# def feature_selection(X_raw, y, top_k=MAX_FEATURES):
#     xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=0, importance_type='gain')
#     xgb_model.fit(X_raw, y); importances = xgb_model.feature_importances_
#     topk_idx = np.argsort(importances)[-top_k:]; topk_features = X_raw.columns[topk_idx]
#     plt.figure(figsize=(8, 5)); plt.barh(topk_features, importances[topk_idx], color='darkgreen')
#     plt.xlabel('Feature Importance'); plt.title('Top Features Importance (XGBoost)'); plt.tight_layout();
#     plt.savefig("model_output/feature_importance.png", dpi=300); plt.show()
#     return topk_features

#Random Forest
def feature_selection(X_raw, y, top_k=MAX_FEATURES):
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X_raw, y)
    importances = rf.feature_importances_
    topk_idx = np.argsort(importances)[-top_k:]
    topk_features = X_raw.columns[topk_idx]

    plt.figure(figsize=(8, 5))
    plt.barh(topk_features, importances[topk_idx], color='steelblue')
    plt.xlabel('Feature Importance')
    plt.title('Top Features Importance')
    plt.tight_layout()
    plt.show()

    return topk_features

def evaluate_performance(y_true, y_pred):
    return r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

def select_samples_randomly(X_pool, budget_per_round):
    num_samples_to_select = min(budget_per_round, len(X_pool))
    if len(X_pool) <= num_samples_to_select: return np.arange(len(X_pool))
    return np.random.choice(np.arange(len(X_pool)), size=num_samples_to_select, replace=False)

def plot_learning_curves(learning_curves, full_dataset_r2):
    al_data = learning_curves['Active Learning']
    rl_data = learning_curves['Random Learning']
    al_r2, rl_r2 = [m[0] for m in al_data['metrics']], [m[0] for m in rl_data['metrics']]
    plt.figure(figsize=(10, 6))
    plt.plot(al_data['size'], al_r2, marker='o', linestyle='-', color='red', label='Active Learning (AL)')
    plt.plot(rl_data['size'], rl_r2, marker='s', linestyle='--', color='gray', label='Random Learning (RL)')
    plt.axhline(y=full_dataset_r2, color='blue', linestyle=':', label=f'Full Dataset R² ({full_dataset_r2:.4f})')
    plt.xlabel('Number of Training Samples'); plt.ylabel('Test R² Score'); plt.title('$H_{{cj}}$: AL vs. RL Convergence (QSVR)')
    plt.legend(); plt.grid(True); plt.tight_layout(); 
    plt.savefig("model_output/Br_QSVR_AL_vs_RL_comparison.png", dpi=300); 
    plt.show()

def plot_round_vs_full_comparison(model_round, X_train_round, y_train_round, full_model, X_train_full, y_train_full, X_test, y_test, round_label):
    print(f"\n--- Generating diagnostic plot: {round_label} AL vs Full Dataset ---")
    pred_train_round, pred_test_round = model_round.predict(X_train_round), model_round.predict(X_test)
    pred_train_full, pred_test_full = full_model.predict(X_train_full), full_model.predict(X_test)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train_round, pred_train_round, color='blue', alpha=0.5, label=f'AL Train ({round_label})')
    plt.scatter(y_test, pred_test_round, color='cyan', alpha=0.8, s=50, edgecolors='k', linewidth=0.5, label=f'AL Test ({round_label})')
    plt.scatter(y_train_full, pred_train_full, color='orange', alpha=0.3, label='Full Train')
    plt.scatter(y_test, pred_test_full, color='red', alpha=0.8, s=50, marker='X', label='Full Test')
    min_val = min(np.concatenate([y_train_round, y_test, y_train_full]).min(), np.concatenate([pred_train_round, pred_test_round, pred_train_full, pred_test_full]).min())
    max_val = max(np.concatenate([y_train_round, y_test, y_train_full]).max(), np.concatenate([pred_train_round, pred_test_round, pred_train_full, pred_test_full]).max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)
    plt.xlabel('True Values'); 
    plt.ylabel('Predicted Values'); 
    plt.title(f'$B_{{r}}$: True vs Predicted ({round_label} / Full)'); 
    plt.legend(); 
    plt.grid(True); 
    plt.axis('equal')
    plt.subplot(1, 2, 2)
    plt.scatter(pred_train_round, y_train_round - pred_train_round, color='blue', alpha=0.5, label=f'AL Train ({round_label})')
    plt.scatter(pred_test_round, y_test - pred_test_round, color='cyan', alpha=0.8, s=50, edgecolors='k', linewidth=0.5, label=f'AL Test ({round_label})')
    plt.scatter(pred_train_full, y_train_full - pred_train_full, color='orange', alpha=0.3, label='Full Train')
    plt.scatter(pred_test_full, y_test - pred_test_full, color='red', alpha=0.8, s=50, marker='X', label='Full Test')
    plt.axhline(0, color='k', linestyle='--', lw=1.5)
    plt.xlabel('Predicted Values'); 
    plt.ylabel('Residuals'); 
    plt.title(f'$B_{{r}}$: Residuals vs Predicted ({round_label} / Full)'); 
    plt.legend(); 
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("model_output/Br_QSVR_performance_comparison_round_vs_full.png", dpi=300)
    plt.show()


def run_learning_process(strategy, X_train_base, y_train_base, X_pool_base, y_pool_base, X_val, y_val, X_test, y_test, fmap, round_to_capture=-1):
    X_train, y_train = X_train_base.copy(), y_train_base.copy()
    X_pool, y_pool = X_pool_base.copy(), y_pool_base.copy()
    
    test_metrics_list, train_size_list = [], []
    best_r2_val = -np.inf
    best_params_global = {'C': 10, 'epsilon': 0.1}
    rounds_no_improve = 0
    
    all_captured_states = {}

    for round_i in range(ACTIVE_ROUNDS):
        print(f"\n=== Round {round_i + 1}/{ACTIVE_ROUNDS} ({strategy}) ===")
        train_size_list.append(len(X_train))
        
        def bo_func(C, epsilon):
            try:
                start_eval = time.time()
                model = QSVR(quantum_kernel=FidelityQuantumKernel(feature_map=fmap), C=max(C, 0.1), epsilon=max(epsilon, 1e-3))
                model.fit(X_train, y_train)
                score = r2_score(y_val, model.predict(X_val))
                end_eval = time.time()
                print(f"  BO Eval: C={C:.3f}, eps={epsilon:.3f}, R2={score:.4f}, time={end_eval - start_eval:.2f} s")
                return score
            except Exception as e:
                print(f"  BO Eval failed: C={C:.3f}, eps={epsilon:.3f}, error: {e}")
                return -1

        print(" Starting Bayesian Optimization...")
        start_bo = time.time()
        optimizer = BayesianOptimization(f=bo_func, pbounds={'C': (5, 25), 'epsilon': (0.1, 0.5)}, random_state=round_i*10, verbose=2)
        

        optimizer.maximize(init_points=INIT_POINTS, n_iter=N_ITER)
        
        end_bo = time.time()
        print(f" Bayesian Optimization total time: {end_bo - start_bo:.2f} s")
        
        best_params_round = optimizer.max['params']
        
        start_train = time.time()
        model = QSVR(quantum_kernel=FidelityQuantumKernel(feature_map=fmap), C=best_params_round['C'], epsilon=best_params_round['epsilon'])
        model.fit(X_train, y_train)
        end_train = time.time()

        start_pred = time.time()
        val_metrics = evaluate_performance(y_val, model.predict(X_val))
        test_metrics = evaluate_performance(y_test, model.predict(X_test))
        end_pred = time.time()
        
        test_metrics_list.append(test_metrics)
        r2_val = val_metrics[0]
        
        print(f"Round {round_i+1}: Val R2={r2_val:.4f}, Test R2={test_metrics[0]:.4f}")
        print(f"    Training time: {end_train - start_train:.2f} s, Prediction time: {end_pred - start_pred:.2f} s")
        
        if r2_val > best_r2_val + 1e-4:
            best_r2_val, best_params_global, rounds_no_improve = r2_val, best_params_round, 0
        else:
            rounds_no_improve += 1
            print(f"No validation R² improvement for {rounds_no_improve} round(s).")
        
        all_captured_states[round_i] = {'model': model, 'X_train': X_train.copy(), 'y_train': y_train.copy()}
        
        if rounds_no_improve >= PATIENCE:
            print(f"Early stopping triggered after {rounds_no_improve} rounds of no improvement.")
            break
        
        if len(X_pool) == 0:
            print("Training pool exhausted. Stopping.")
            break
        
        if strategy == 'Active Learning':
            # This logic is from your original code
            preds = []
            for _ in range(3):
                try:
                    boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
                    boot_model = QSVR(quantum_kernel=FidelityQuantumKernel(feature_map=fmap), C=best_params_round['C'], epsilon=best_params_round['epsilon'])
                    boot_model.fit(X_train[boot_idx], y_train[boot_idx])
                    preds.append(boot_model.predict(X_pool))
                except Exception: continue
            
            uncertainty = np.std(np.array(preds), axis=0) if preds else np.zeros(len(X_pool))
            
            train_center = X_train.mean(axis=0)
            distances = np.linalg.norm(X_pool - train_center, axis=1)
            norm_dist = distances / distances.max() if distances.max() > 0 else distances
            scores = LAMBDA_REP * uncertainty + (1 - LAMBDA_REP) * norm_dist
            selected_idx = np.argsort(-scores)[:min(BUDGET_PER_ROUND, len(scores))].tolist()
        else:
            selected_idx = select_samples_randomly(X_pool, BUDGET_PER_ROUND)
            
        X_selected, y_selected = X_pool[selected_idx], y_pool[selected_idx]
        X_train, y_train = np.vstack([X_train, X_selected]), np.concatenate([y_train, y_selected])
        X_pool, y_pool = np.delete(X_pool, selected_idx, axis=0), np.delete(y_pool, selected_idx, axis=0)
        print(f"Selected {len(selected_idx)} new samples. Train size: {len(X_train)}")
    
    final_model = QSVR(quantum_kernel=FidelityQuantumKernel(feature_map=fmap), C=best_params_global.get('C', 10), epsilon=best_params_global.get('epsilon', 0.1))
    final_model.fit(X_train, y_train)
    
    return final_model, test_metrics_list, train_size_list, all_captured_states



def main_run_workflow(data_path, results_save_path="model_output/br_qsvr_results.joblib"):
    # (This function remains the same)
    data = pd.read_csv(data_path); data.fillna(data.median(), inplace=True)
    X_raw, y = data.iloc[:, :-1], data.iloc[:, -1]
    topk_features = feature_selection(X_raw, y); X = X_raw[topk_features]
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42)
    X_pool_base, X_val, y_pool_base, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=1)
    X_train_base, X_pool_base, y_train_base, y_pool_base = train_test_split(
        X_pool_base, y_pool_base, train_size=INIT_TRAIN_SIZE, random_state=0)
    os.makedirs('model_output', exist_ok=True)
    fmap = PauliFeatureMap(feature_dimension=X_train_base.shape[1], reps=REPS, entanglement='linear')
    learning_curves, captured_states_all_rounds = {}, {}
    for strategy in ['Active Learning', 'Random Learning']:
        final_model, metrics_list, size_list, all_rounds_state = run_learning_process(
            strategy, X_train_base, y_train_base, X_pool_base, y_pool_base, 
            X_val, y_val, X_test, y_test, fmap
        )
        learning_curves[strategy] = {'metrics': metrics_list, 'size': size_list}
        if strategy == 'Active Learning':
             captured_states_all_rounds['Active Learning'] = all_rounds_state
    print("\n" + "="*20 + "\n Training Full Dataset Model \n" + "="*20)
    al_final_params = captured_states_all_rounds['Active Learning'][max(captured_states_all_rounds['Active Learning'].keys())]['model'].get_params()
    full_model = QSVR(quantum_kernel=FidelityQuantumKernel(feature_map=fmap), C=al_final_params['C'], epsilon=al_final_params['epsilon'])
    full_model.fit(X_temp, y_temp)
    full_model_metrics = evaluate_performance(y_test, full_model.predict(X_test))
    results_to_save = {
        'learning_curves': learning_curves,
        'all_captured_al_states': captured_states_all_rounds['Active Learning'],
        'full_model': full_model,
        'full_model_metrics': full_model_metrics,
        'data': { 'X_temp': X_temp, 'y_temp': y_temp, 'X_test': X_test, 'y_test': y_test }
    }
    joblib.dump(results_to_save, results_save_path)
    print(f"\n All results successfully saved to '{results_save_path}'.")
    return results_to_save

def main_plotting_workflow(results_path="model_output/qsvr_results.joblib", round_to_plot=12):
    # (This function remains the same)
    if not os.path.exists(results_path):
        print(f"Error: Results file '{results_path}' not found. Please run main_run_workflow() first.")
        return
    print(f"\n{'='*20}\n Loading results and generating plots for Round {round_to_plot} \n{'='*20}")
    results = joblib.load(results_path)
    learning_curves, all_captured_al_states, full_model = \
        results['learning_curves'], results['all_captured_al_states'], results['full_model']
    X_temp, y_temp, X_test, y_test = \
        results['data']['X_temp'], results['data']['y_temp'], results['data']['X_test'], results['data']['y_test']
    
    round_to_plot_idx = round_to_plot - 1
    table_results = {}
    if len(learning_curves['Active Learning']['metrics']) > round_to_plot_idx:
        table_results[f'Active Learning (Round {round_to_plot})'] = learning_curves['Active Learning']['metrics'][round_to_plot_idx]
    if len(learning_curves['Random Learning']['metrics']) > round_to_plot_idx:
        table_results[f'Random Learning (Round {round_to_plot})'] = learning_curves['Random Learning']['metrics'][round_to_plot_idx]
    table_results['Full Dataset'] = results['full_model_metrics']
    perf_df = pd.DataFrame(table_results, index=['R2_test', 'MAE_test', 'RMSE_test']).T
    print("\n" + "="*40 + "\nFinal Performance Comparison (Test set):\n")
    print(perf_df.round(6))
    print("\n" + "="*40 + "\n")
    plot_learning_curves(learning_curves, results['full_model_metrics'][0])
    if round_to_plot_idx in all_captured_al_states:
        captured_state = all_captured_al_states[round_to_plot_idx]
        model_r, X_train_r, y_train_r = captured_state['model'], captured_state['X_train'], captured_state['y_train']
        plot_round_vs_full_comparison(model_r, X_train_r, y_train_r, full_model, X_temp, y_temp, X_test, y_test, f"Round {round_to_plot}")
    else:
        print(f"Warning: Could not find captured state for Round {round_to_plot}. Skipping diagnostic plot.")


if __name__ == "__main__":
    main_run_workflow("Br.csv", results_save_path="model_output/br_qsvr_results.joblib")
    main_plotting_workflow("model_output/br_qsvr_results.joblib", round_to_plot=13)
