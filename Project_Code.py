# Aseef Nazrul 

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB  
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier 
import xgboost as xgb  

plt.style.use('ggplot')
sns.set(style="whitegrid")

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
    column_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 
                    'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
    df = pd.read_csv(url, header=None, names=column_names)
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]  
    y = y.map({'g': 1, 'h': 0})
    return X, y

def explore_data(X, y):
    print("\n=== Exploratory Data Analysis ===")
    
    data = X.copy()
    data['class'] = y
    class_counts = y.value_counts()
    gamma_count = class_counts.get(1, 0)  
    hadron_count = class_counts.get(0, 0)  
    total_count = gamma_count + hadron_count
    
    print("\nFeature Statistics:")
    stats_df = X.describe()  
    print(f"Total count = {total_count}")
    print(f"Class distribution: Gamma (signal): {gamma_count} ({gamma_count/total_count:.1%}), Hadron (background): {hadron_count} ({hadron_count/total_count:.1%})")
    print()
    stats_df = stats_df.drop('count')
    stats_df_transposed = stats_df.transpose()
    stats_df_transposed = stats_df_transposed.rename(columns={
        '25%': 'LQ',
        '50%': 'Median',
        '75%': 'UQ'
    })
    stats_df_transposed['mean'] = stats_df_transposed['mean'].round(1)
    stats_df_transposed['std'] = stats_df_transposed['std'].round(1)
    stats_df_transposed['min'] = stats_df_transposed['min'].round(2)
    stats_df_transposed['LQ'] = stats_df_transposed['LQ'].round(1)
    stats_df_transposed['Median'] = stats_df_transposed['Median'].round(1)
    stats_df_transposed['UQ'] = stats_df_transposed['UQ'].round(1)
    stats_df_transposed['max'] = stats_df_transposed['max'].round(1)
    print(stats_df_transposed)
    return data

def preprocess_data(X, y):
    print("\n=== Data Preprocessing ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    print("\n=== Model Training and Evaluation ===")
    available_models = {
        '1': ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        '2': ('Naive Bayes', GaussianNB()),
        '3': ('SVM (RBF)', SVC(kernel='rbf', probability=True, random_state=42)),
        '4': ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        '5': ('XGBoost', xgb.XGBClassifier(n_estimators=100, random_state=42))
    }
    
    print("\nModels:")
    for key, (model_name, _) in available_models.items():
        print(f"{key}: {model_name}")
    print("6: All models")
    
    selection = input("\nEnter the number of the model(s) to run: ")
    
    selected_models = {}
    if selection.strip() == '6':
        selected_models = {name: model for _, (name, model) in available_models.items()}
    else:
        for model_id in selection.split(','):
            model_id = model_id.strip()
            if model_id in available_models:
                name, model = available_models[model_id]
                selected_models[name] = model
            else:
                print(f"Warning: Invalid model ID '{model_id}' - skipping")
    
    if not selected_models:
        print("No valid models selected. Exiting...")
        return {}
    results = {}
    
    for name, model in selected_models.items():
        print(f"\n{name}:\n---------------\n")
        #5-fold cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"5-fold Cross-Validation: Accuracy = {cv_mean:.4f} ± {cv_std:.4f}")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            if hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test_scaled)
            else:
                y_prob = y_pred 
        
        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        hadron_count = np.sum(y_test == 0)
        gamma_count = np.sum(y_test == 1)
        report = "\nClassification Report:\n\n"
        report += "                precision   recall    f1-score\n"
        report += "\n"
        report += f"       Hadron       {report_dict['0']['precision']:.3f}      {report_dict['0']['recall']:.3f}      {report_dict['0']['f1-score']:.3f}\n"
        report += f"       Gamma        {report_dict['1']['precision']:.3f}      {report_dict['1']['recall']:.3f}      {report_dict['1']['f1-score']:.3f}\n"
        report += "\n"
        report += f"    accuracy  {report_dict['accuracy']:.3f}\n"
        conf_matrix = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'conf_matrix': conf_matrix,
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        print(report)
        
        print("Confusion Matrix:")
        print(f"[{conf_matrix[0,0]} {conf_matrix[0,1]}]")
        print(f"[{conf_matrix[1,0]} {conf_matrix[1,1]}]")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrBr_r',
                    xticklabels=['Hadron', 'Gamma'],
                    yticklabels=['Hadron', 'Gamma'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {name}')
        plt.tight_layout()
        plt.show()
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\nFeature Importance:")
            for i, idx in enumerate(indices):
                if i < 10: 
                    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
            
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importance - {name}")
            bars = plt.bar(range(len(importances)), importances[indices])
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', rotation=0)
            
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()

    if results:
        if 'XGBoost' in results:
            plt.figure(figsize=(10, 8))
            plt.plot(results['XGBoost']['fpr'], results['XGBoost']['tpr'], 
                    label=f'XGBoost (AUC = {results["XGBoost"]["auc"]:.3f})', color='darkgreen', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for XGBoost Classifier')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
            
        plt.figure(figsize=(10, 8))
        for name in results:
            plt.plot(results[name]['fpr'], results[name]['tpr'], 
                    label=f'{name} (AUC = {results[name]["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Selected Classifiers')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        if len(results) > 1:
            plt.figure(figsize=(12, 6))
            model_names = list(results.keys())
            accuracies = [results[name]['accuracy'] for name in model_names]
            aucs = [results[name]['auc'] for name in model_names]
            cv_accuracies = [results[name]['cv_mean'] for name in model_names]
            
            x = np.arange(len(model_names))
            width = 0.25
            
            plt.bar(x - width, cv_accuracies, width, label='CV Accuracy')
            plt.bar(x, accuracies, width, label='Test Accuracy')
            plt.bar(x + width, aucs, width, label='AUC')
            
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x, model_names, rotation=45)
            plt.legend()
            plt.grid(axis='y')
            plt.tight_layout()
            plt.show()
    
    print("\nGraphs displayed")
    return results

def perform_feature_selection(X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    print("\n=== Feature Selection ===")
    
    selector_f = SelectKBest(f_classif, k=5)
    X_train_selected_f = selector_f.fit_transform(X_train_scaled, y_train)
    X_test_selected_f = selector_f.transform(X_test_scaled)
    
    selected_indices_f = selector_f.get_support(indices=True)
    selected_features_f = [feature_names[i] for i in selected_indices_f]
    feature_scores_f = selector_f.scores_
    
    feature_score_pairs = [(feat, score) for feat, score in zip(selected_features_f, feature_scores_f[selected_indices_f])]
    feature_score_pairs.sort(key=lambda x: x[1], reverse=True)
    sorted_features_f = [feat for feat, _ in feature_score_pairs]
    print("The top five selected features using F-score were:", ", ".join(sorted_features_f))
   
    print("\nFeature Selection using RFE with Random Forest:")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    selector_rfe = RFE(estimator=rf, n_features_to_select=5, step=1)
    X_train_selected_rfe = selector_rfe.fit_transform(X_train_scaled, y_train)
    X_test_selected_rfe = selector_rfe.transform(X_test_scaled)
    
    selected_indices_rfe = np.where(selector_rfe.support_)[0]
    selected_features_rfe = [feature_names[i] for i in selected_indices_rfe]
    
    print("RFE with Random Forest selected", ", ".join(selected_features_rfe), "as the most important features.")
    
    plt.figure(figsize=(12, 6))
    
    all_features = list(feature_names)
    fscore_selected = [feature in selected_features_f for feature in all_features]
    rfe_selected = [feature in selected_features_rfe for feature in all_features]
    
    selection_df = pd.DataFrame({
        'Feature': all_features,
        'F-Score': fscore_selected,
        'RFE': rfe_selected
    })
    
   
    plt.figure(figsize=(14, 8))
    
    fscore_values = feature_scores_f  
    
    rf.fit(X_train_scaled, y_train)
    rfe_values = rf.feature_importances_
    
    fscore_values = fscore_values / fscore_values.max()
    rfe_values = rfe_values / rfe_values.max()
    
    x = np.arange(len(all_features))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - width/2, fscore_values, width, label='F-score', color='cornflowerblue')
    bars2 = ax.bar(x + width/2, rfe_values, width, label='RFE', color='lightcoral')
    
    for i, feature in enumerate(all_features):
        if feature in sorted_features_f:
            bars1[i].set_color('darkblue')
            bars1[i].set_edgecolor('black')
            bars1[i].set_linewidth(1.5)
        
        if feature in selected_features_rfe:
            bars2[i].set_color('darkred')
            bars2[i].set_edgecolor('black')
            bars2[i].set_linewidth(1.5)
    ax.set_xlabel('Features')
    ax.set_ylabel('Normalized Importance')
    ax.set_title('Feature Importance for All Features: F-score vs RFE')
    ax.set_xticks(x)
    ax.set_xticklabels(all_features, rotation=45, ha='right')
    legend = ax.legend(loc='upper right')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkblue', edgecolor='black', label='Top 5 by F-score'),
        Patch(facecolor='darkred', edgecolor='black', label='Top 5 by RFE')
    ]
    ax.legend(handles=legend_elements, loc='upper left')  
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return {
        'fscore': {
            'features': selected_features_f,
            'X_train': X_train_selected_f,
            'X_test': X_test_selected_f
        },
        'rfe': {
            'features': selected_features_rfe,
            'X_train': X_train_selected_rfe,
            'X_test': X_test_selected_rfe
        }
    }

def compare_models_with_feature_selection(X_train_scaled, X_test_scaled, feature_selection_results, y_train, y_test):
    print("\n=== Model Comparison: With and Without Feature Selection ===")
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
    }
    comparison_results = {}

    print("\nBaseline (All Features):")
    baseline_results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            if hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test_scaled)
            else:
                y_prob = y_pred 
                
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        baseline_results[name] = {
            'accuracy': accuracy,
            'auc': roc_auc
        }
        print(f"{name}: Accuracy = {accuracy:.4f}, AUC = {roc_auc:.4f}")
    comparison_results['All Features'] = baseline_results
    
    print("\nF-Score Selected Features:")
    fscore_results = {}
    for name, model in models.items():
        model.fit(feature_selection_results['fscore']['X_train'], y_train)
        y_pred = model.predict(feature_selection_results['fscore']['X_test'])
        accuracy = accuracy_score(y_test, y_pred)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(feature_selection_results['fscore']['X_test'])[:, 1]
        else:
            if hasattr(model, "decision_function"):
                y_prob = model.decision_function(feature_selection_results['fscore']['X_test'])
            else:
                y_prob = y_pred
                
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fscore_results[name] = {
            'accuracy': accuracy,
            'auc': roc_auc
        }
        print(f"{name}: Accuracy = {accuracy:.4f}, AUC = {roc_auc:.4f}")
    
    comparison_results['F-Score'] = fscore_results
    
    print("\nRFE Selected Features:")
    rfe_results = {}
    for name, model in models.items():
        model.fit(feature_selection_results['rfe']['X_train'], y_train)
        y_pred = model.predict(feature_selection_results['rfe']['X_test'])
        accuracy = accuracy_score(y_test, y_pred)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(feature_selection_results['rfe']['X_test'])[:, 1]
        else:
            if hasattr(model, "decision_function"):
                y_prob = model.decision_function(feature_selection_results['rfe']['X_test'])
            else:
                y_prob = y_pred
                
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        rfe_results[name] = {
            'accuracy': accuracy,
            'auc': roc_auc
        }
        
        print(f"{name}: Accuracy = {accuracy:.4f}, AUC = {roc_auc:.4f}")
    
    comparison_results['RFE'] = rfe_results
    methods = list(comparison_results.keys())
    model_names = list(models.keys())
    
    plt.figure(figsize=(14, 8))
    width = 0.2
    x = np.arange(len(model_names))
    
    for i, method in enumerate(methods):
        accuracies = [comparison_results[method][model]['accuracy'] for model in model_names]
        plt.bar(x + (i - 1) * width, accuracies, width, label=method)
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison with Different Feature Selection Methods')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(14, 8))
    
    for i, method in enumerate(methods):
        aucs = [comparison_results[method][model]['auc'] for model in model_names]
        plt.bar(x + (i - 1) * width, aucs, width, label=method)
    
    plt.xlabel('Models')
    plt.ylabel('AUC')
    plt.title('Model AUC Comparison with Different Feature Selection Methods')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show() 
    return comparison_results

def main():
    start_time = time.time()
    print("\nCPS844 Group 17")
    X, y = load_data()
    if X is None or y is None:
        return
    
    data = explore_data(X, y)
    
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocess_data(X, y)
    model_results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, X.columns)
    feature_selection_results = perform_feature_selection(X_train_scaled, X_test_scaled, y_train, y_test, X.columns)
    comparison_results = compare_models_with_feature_selection(
        X_train_scaled, X_test_scaled, feature_selection_results, y_train, y_test)
    
    elapsed_time = round(time.time() - start_time, 2)
    print(f"\ndone in {elapsed_time} seconds")
    
if __name__ == "__main__":
    main()