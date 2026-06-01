import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import joblib
import sys

# 1. Konfigurasi
TRAIN_PATH = 'UNSW_NB15_training-set.parquet'
TEST_PATH = 'UNSW_NB15_testing-set.parquet'
TARGET_MULTI = 'attack_cat'
TARGET_BINARY = 'label'

def log_print(msg):
    print(msg)
    sys.stdout.flush()

def load_data():
    log_print("Memuat data...")
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    return train, test

def preprocess_data(train, test):
    log_print("Pra-pemrosesan (Stabil)...")
    
    cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
    if TARGET_MULTI in cat_cols:
        cat_cols.remove(TARGET_MULTI)
    
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        
    le_target = LabelEncoder()
    train[TARGET_MULTI] = le_target.fit_transform(train[TARGET_MULTI].astype(str))
    test[TARGET_MULTI] = le_target.transform(test[TARGET_MULTI].astype(str))
    
    return train, test, cat_cols, le_target

def train_final(train, test, cat_cols, target_col):
    log_print(f"Melatih model final untuk {target_col}...")
    
    X = train.drop([TARGET_MULTI, TARGET_BINARY], axis=1)
    y = train[target_col]
    X_test = test.drop([TARGET_MULTI, TARGET_BINARY], axis=1)
    
    # Parameter yang dioptimasi secara manual berdasarkan hasil sebelumnya
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': len(np.unique(y)),
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'learning_rate': 0.05,
        'num_leaves': 127,      # Lebih banyak leaves untuk menangkap pola kompleks
        'max_depth': -1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'class_weight': 'balanced',
        'n_estimators': 300     # Estimator yang cukup tanpa berlebihan
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y, categorical_feature=cat_cols)
    
    return model

def main():
    try:
        train, test = load_data()
        train, test, cat_cols, le_target = preprocess_data(train, test)
        
        # Train
        model = train_final(train, test, cat_cols, TARGET_MULTI)
        
        # Predict
        X_test = test.drop([TARGET_MULTI, TARGET_BINARY], axis=1)
        final_preds = model.predict(X_test)
        
        log_print("\n--- Evaluasi Akhir (Best Stable Version) ---")
        y_test = test[TARGET_MULTI]
        print(classification_report(y_test, final_preds, target_names=le_target.classes_))
        log_print(f"Macro F1-Score Final: {f1_score(y_test, final_preds, average='macro'):.4f}")
        
        # Simpan
        joblib.dump(model, 'model_final_v2.pkl')
        joblib.dump(le_target, 'label_encoder_final.pkl')
        
        submission = pd.DataFrame({
            'Index': test.index,
            'Predicted_Attack_Cat': le_target.inverse_transform(final_preds)
        })
        submission.to_csv('submission_final_v2.csv', index=False)
        log_print("\nSelesai! Model final v2 telah dibuat.")
    except Exception as e:
        log_print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()
