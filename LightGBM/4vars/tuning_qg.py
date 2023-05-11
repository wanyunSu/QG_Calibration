# %%
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.datasets import load_wine
import optuna
from optuna.samplers import TPESampler
import pickle
import pandas as pd 
import joblib
import os, sys 
sys.path.append('/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/BDT_EB4/LightGBM')
from LightGBM_BDT_train import *


# %%
use_full_dataset = 1

if use_full_dataset:
    sample_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/LightGBM/training_sample.pkl'
    output_path = './full_dataset'
    n_trails = 1
    
else:
    sample_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/BDT_EB4/samples/sample_testweight_123'
    output_path = "./small_dataset"
    n_trails = 10
# Use for full dataset tuning
# sample_path = '/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/BDT_EB4/samples/sample_all_jets.pkl'

# Use for code dev with a small dataset 

output_folder = Path(output_path)
if not output_folder.exists():
    output_folder.mkdir(parents=True, exist_ok=True)

study_output = output_folder / 'study.pkl'
gbdt_filename = output_folder / 'lightgbm_gbdt_1.pkl'
eval_result_filename = output_folder / 'eval_result.pkl'

training_vars = ['jet_pt', 'jet_nTracks', 'jet_trackWidth', 'jet_trackC1']
training_weight = ['flatpt_weight']

label_pt_bin = [500, 600, 800, 1000, 1200, 1500, 2000]


# %%
sample = pd.read_pickle(sample_path)

# %%
sample.head()

# %%


target_idx = sample.columns.get_loc('target')
y = sample.iloc[:, target_idx]
X = sample.drop(columns = 'target')

X_dev,X_test, y_dev,y_test = train_test_split(X, y, test_size=0.1, random_state=456)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.1/0.9, random_state=789)

# %% [markdown]
# ## Train a base model

# %%
# After test, eval_sample_weight already pass the event weight 
# def auc_weighted(y_true, y_pred, weight):
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred, sample_weight = weight)
#     roc_auc = auc(fpr, tpr)

#     return ('auc_weighted', eval_result, is_higher_better)

# %%
#### Train a base model with default setting. Only set random_state for reproducibility. 
# base_model = lgb.LGBMClassifier(random_state=42)
# # base_model.fit(X_train[training_vars], y_train, sample_weight=X_train[training_weight].to_numpy().flatten())

# base_model.fit(X = X_train[training_vars], y = y_train, sample_weight=X_train[training_weight].to_numpy().flatten(),
#                eval_set = [(X_val[training_vars], y_val)], eval_sample_weight = [X_val[training_weight].to_numpy().flatten()],
#                eval_metric = ['binary_logloss', 'auc'])

# # %%
# y_test_pred = base_model.predict_proba(X_test[training_vars])[:,1]
# accuracy_score(y_test, y_test_pred)
# print(classification_report(y_test, y_test_pred))

# # %%
# y_train_decisions = base_model.predict_proba(X_train[training_vars])[:,1]
# y_test_decisions = base_model.predict_proba(X_test[training_vars])[:,1]

# # %%
# plot_decision_func(X_test, y_test_decisions, y_test, output_path)

# # %%
# plot_ROC(y_decisions=y_test_decisions, y_tmva=X_test.iloc[:,X_test.columns.get_loc('jet_trackBDT')], 
#                  y_ntrk=X_test.iloc[:,X_test.columns.get_loc('jet_nTracks')], target=y_test, 
#                  X_weight=X_test['event_weight'], features=" 4 vars", output_path=output_path)


# # %%
# plot_overtraining_validation(X_dev=X_train, X_test=X_test, y_dev=y_train, y_test=y_test, 
#                                     y_dev_decisions=y_train_decisions, y_test_decisions=y_test_decisions, 
#                                     output_path=output_path)

# %% [markdown]
# ## Tune the model

# %%
def objective(trial):
    """
    Objective function to be minimized.
    """
    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    # TODO Add a pruner to observe intermediate results and stop unpromising trials.
    # https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_integration.py 
    # https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    
    gbm = lgb.LGBMClassifier(**param)
    # gbm.fit(X = X_train[training_vars], y = y_train, sample_weight=X_train[training_weight].to_numpy().flatten())
    gbm.fit(X = X_train[training_vars], y = y_train, sample_weight=X_train[training_weight].to_numpy().flatten(),
            eval_set = [(X_val[training_vars], y_val)], eval_sample_weight = [X_val[training_weight].to_numpy().flatten()],
            callbacks=[pruning_callback])

    # choose the highest auc score with event weight  
    y_val_decisions = gbm.predict_proba(X_val[training_vars])[:,1]
    fpr, tpr, thresholds = roc_curve(y_val, y_val_decisions, sample_weight = X_val['event_weight'])
    roc_auc = auc(fpr, tpr)

    return roc_auc


# %%
sampler = TPESampler(seed=1)
study = optuna.create_study(study_name="lightgbm_qgtagging", direction="maximize", sampler = sampler, pruner=optuna.pruners.HyperbandPruner())

study.optimize(objective, n_trials=n_trails, n_jobs = 1)
# n_jobs doesn't make tuning faster because it uses multi-threading.
# https://optuna.readthedocs.io/en/stable/faq.html#multi-threading-parallelization-with-a-single-node

# %%
import joblib
joblib.dump(study, study_output)

# %%
# study = joblib.load("study.pkl")

# %%
study.best_params

# %%
study.best_value

# %% [markdown]
# ## Use the best params to train

# %%
eval_result={}
best_model = lgb.LGBMClassifier(**study.best_params)
best_model.fit(X = X_train[training_vars], y = y_train, sample_weight=X_train[training_weight].to_numpy().flatten(),
               eval_set = [(X_val[training_vars], y_val)], eval_sample_weight = [X_val[training_weight].to_numpy().flatten()],
               eval_metric = ['binary_logloss', 'auc'], callbacks=[lgb.early_stopping(5), lgb.record_evaluation(eval_result)])

# %%
print(best_model)
joblib.dump(best_model, gbdt_filename)
joblib.dump(eval_result, eval_result_filename)

# %%



