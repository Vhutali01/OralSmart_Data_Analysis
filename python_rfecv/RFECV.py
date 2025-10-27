import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RFECVAnalyzer:
    def __init__(self, data, target_column, test_size=0.2, random_state=42):
        """
        Initialize RFECV Analyzer
        
        Parameters:
        data: pandas DataFrame
        target_column: string, name of target column
        test_size: float, proportion of data for testing
        random_state: int, for reproducibility
        """
        self.data = data.copy()
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}
        self.processed_data = None
        self.feature_names = None
        
    def preprocess_data(self):
        """Preprocess the data for RFECV analysis"""
        print("Preprocessing data...")
        
        # Separate features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        # Convert low-cardinality numerical variables to categorical
        for col in numerical_cols:
            if X[col].nunique() <= 10:
                X[col] = X[col].astype('category')
                categorical_cols = categorical_cols.append(pd.Index([col]))
                numerical_cols = numerical_cols.drop(col)
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Encode target variable if it's categorical
        target_encoder = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
        
        # Store preprocessing info
        self.label_encoders = label_encoders
        self.target_encoder = target_encoder
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        
        # Scale numerical features
        scaler = StandardScaler()
        if len(numerical_cols) > 0:
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        self.scaler = scaler
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Features processed: {len(self.feature_names)}")
        print(f"Categorical features: {len(categorical_cols)}")
        print(f"Numerical features: {len(numerical_cols)}")
        print(f"Target classes: {len(np.unique(y))}")
        
        return X, y
    
    def run_rfecv_analysis(self, estimators=None, cv_folds=5, scoring='accuracy'):
        """
        Run RFECV analysis with multiple estimators
        
        Parameters:
        estimators: dict of estimators to test
        cv_folds: int, number of cross-validation folds
        scoring: string, scoring metric
        """
        if estimators is None:
            estimators = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'SVM': SVC(kernel='linear', random_state=self.random_state, probability=True)
            }
        
        # Preprocess data
        X, y = self.preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Store test data
        self.X_test = X_test
        self.y_test = y_test
        
        # Cross-validation strategy
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        print(f"\nRunning RFECV analysis with {cv_folds}-fold cross-validation...")
        print(f"Scoring metric: {scoring}")
        
        for name, estimator in estimators.items():
            print(f"\nProcessing {name}...")
            
            # Run RFECV
            rfecv = RFECV(
                estimator=estimator,
                step=1,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )
            
            rfecv.fit(X_train, y_train)
            
            # Get selected features
            selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) if rfecv.support_[i]] # type: ignore
            feature_rankings = [(self.feature_names[i], rfecv.ranking_[i]) for i in range(len(self.feature_names))] # type: ignore
            feature_rankings.sort(key=lambda x: x[1])
            
            # Calculate performance metrics
            best_score = rfecv.cv_results_['mean_test_score'][rfecv.n_features_ - 1]
            
            # Test set performance
            y_pred = rfecv.predict(X_test)
            test_score = rfecv.score(X_test, y_test)
            
            # Store results
            self.results[name] = {
                'rfecv_object': rfecv,
                'selected_features': selected_features,
                'n_features_selected': rfecv.n_features_,
                'feature_rankings': feature_rankings,
                'cv_scores': rfecv.cv_results_['mean_test_score'],
                'cv_scores_std': rfecv.cv_results_['std_test_score'],
                'best_cv_score': best_score,
                'test_score': test_score,
                'y_pred': y_pred,
                'support_': rfecv.support_,
                'ranking_': rfecv.ranking_
            }
            
            print(f"  Optimal features: {rfecv.n_features_}")
            print(f"  Best CV score: {best_score:.4f}")
            print(f"  Test score: {test_score:.4f}")
    
    def create_summary_table(self):
        """Create summary table of RFECV results"""
        summary_data = []
        
        for estimator_name, result in self.results.items():
            summary_data.append({
                'Estimator': estimator_name,
                'Optimal_Features': result['n_features_selected'],
                'Best_CV_Score': result['best_cv_score'],
                'Test_Score': result['test_score'],
                'Total_Features': len(self.feature_names), # type: ignore
                'Feature_Reduction_%': (1 - result['n_features_selected'] / len(self.feature_names)) * 100 # type: ignore
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def plot_rfecv_results(self, save_plots=True):
        """Create comprehensive plots of RFECV results"""
        n_estimators = len(self.results)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: CV scores vs number of features
        plt.subplot(3, 3, 1)
        for name, result in self.results.items():
            n_features_range = range(1, len(result['cv_scores']) + 1)
            plt.plot(n_features_range, result['cv_scores'], 
                    marker='o', label=f'{name}', linewidth=2, markersize=4)
            plt.fill_between(n_features_range, 
                           result['cv_scores'] - result['cv_scores_std'],
                           result['cv_scores'] + result['cv_scores_std'], 
                           alpha=0.1)
            # Mark optimal number of features
            optimal_idx = result['n_features_selected'] - 1
            plt.axvline(x=result['n_features_selected'], linestyle='--', alpha=0.7,
                       color=plt.gca().lines[-1].get_color())
        
        plt.xlabel('Number of Features')
        plt.ylabel('Cross Validation Score')
        plt.title('RFECV: CV Score vs Number of Features')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Number of optimal features comparison
        plt.subplot(3, 3, 2)
        estimators = list(self.results.keys())
        optimal_features = [self.results[est]['n_features_selected'] for est in estimators]
        bars = plt.bar(estimators, optimal_features, color=['skyblue', 'lightcoral', 'lightgreen'][:len(estimators)])
        plt.xlabel('Estimator')
        plt.ylabel('Optimal Number of Features')
        plt.title('Optimal Number of Features by Estimator')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, optimal_features):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: CV vs Test scores
        plt.subplot(3, 3, 3)
        cv_scores = [self.results[est]['best_cv_score'] for est in estimators]
        test_scores = [self.results[est]['test_score'] for est in estimators]
        
        x = np.arange(len(estimators))
        width = 0.35
        
        plt.bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.8)
        plt.bar(x + width/2, test_scores, width, label='Test Score', alpha=0.8)
        
        plt.xlabel('Estimator')
        plt.ylabel('Score')
        plt.title('CV Score vs Test Score')
        plt.xticks(x, estimators, rotation=45)
        plt.legend()
        
        # Plot 4-6: Feature importance/ranking for each estimator
        for idx, (name, result) in enumerate(self.results.items()):
            plt.subplot(3, 3, 4 + idx)
            
            # Get top 15 features (or all if less than 15)
            top_features = result['feature_rankings'][:min(15, len(result['feature_rankings']))]
            feature_names = [f[0] for f in top_features]
            rankings = [f[1] for f in top_features]
            
            # Color selected features differently
            colors = ['green' if rank == 1 else 'lightblue' for rank in rankings]
            
            bars = plt.barh(range(len(feature_names)), rankings, color=colors)
            plt.yticks(range(len(feature_names)), feature_names)
            plt.xlabel('Feature Ranking')
            plt.title(f'Top Features Ranking - {name}')
            plt.gca().invert_yaxis()
            
            # Add ranking labels
            for i, (bar, rank) in enumerate(zip(bars, rankings)):
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        str(rank), ha='left', va='center', fontweight='bold')
        
        # Plot 7: Feature selection overlap
        if len(self.results) > 1:
            plt.subplot(3, 3, 7)
            
            # Create feature selection matrix
            feature_selection_matrix = []
            for name, result in self.results.items():
                feature_selection_matrix.append([1 if f in result['selected_features'] else 0 
                                               for f in self.feature_names]) # type: ignore
            
            feature_selection_df = pd.DataFrame(feature_selection_matrix, 
                                              index=list(self.results.keys()),
                                              columns=self.feature_names)
            
            # Show features selected by at least one method
            selected_by_any = feature_selection_df.sum(axis=0) > 0
            subset_df = feature_selection_df.loc[:, selected_by_any]
            
            # Limit to top features for readability
            if subset_df.shape[1] > 20:
                # Sort by total selection count
                selection_counts = subset_df.sum(axis=0).sort_values(ascending=False)
                top_features_idx = selection_counts.head(20).index
                subset_df = subset_df[top_features_idx]
            
            sns.heatmap(subset_df, annot=True, cmap='RdYlGn', cbar=False)
            plt.title('Feature Selection Overlap')
            plt.xlabel('Features')
            plt.ylabel('Estimators')
            plt.xticks(rotation=90)
        
        # Plot 8: Performance comparison
        plt.subplot(3, 3, 8)
        performance_data = []
        for name, result in self.results.items():
            performance_data.extend([
                {'Estimator': name, 'Score_Type': 'CV Score', 'Score': result['best_cv_score']},
                {'Estimator': name, 'Score_Type': 'Test Score', 'Score': result['test_score']}
            ])
        
        perf_df = pd.DataFrame(performance_data)
        sns.boxplot(data=perf_df, x='Estimator', y='Score', hue='Score_Type')
        plt.title('Performance Distribution')
        plt.xticks(rotation=45)
        
        # Plot 9: Feature reduction percentage
        plt.subplot(3, 3, 9)
        reduction_pct = [(1 - self.results[est]['n_features_selected'] / len(self.feature_names)) * 100  # type: ignore
                        for est in estimators]
        bars = plt.bar(estimators, reduction_pct, color=['coral', 'lightsteelblue', 'lightseagreen'][:len(estimators)])
        plt.xlabel('Estimator')
        plt.ylabel('Feature Reduction (%)')
        plt.title('Feature Reduction Percentage')
        plt.xticks(rotation=45)
        
        # Add percentage labels
        for bar, pct in zip(bars, reduction_pct):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('rfecv_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_detailed_feature_report(self):
        """Create detailed feature selection report"""
        print("\n" + "="*80)
        print("DETAILED FEATURE SELECTION REPORT")
        print("="*80)
        
        # Overall statistics
        all_selected_features = set()
        for result in self.results.values():
            all_selected_features.update(result['selected_features'])
        
        print(f"Total original features: {len(self.feature_names)}") # type: ignore
        print(f"Features selected by at least one method: {len(all_selected_features)}")
        
        # Feature frequency analysis
        feature_frequency = {}
        for feature in self.feature_names: # type: ignore
            count = sum(1 for result in self.results.values() if feature in result['selected_features'])
            if count > 0:
                feature_frequency[feature] = count
        
        print(f"\nFeatures selected by multiple methods:")
        for feature, count in sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                print(f"  {feature}: selected by {count}/{len(self.results)} methods")
        
        # Method-specific results
        for name, result in self.results.items():
            print(f"\n{name.upper()}:")
            print(f"  Optimal features: {result['n_features_selected']}")
            print(f"  Best CV score: {result['best_cv_score']:.4f}")
            print(f"  Test score: {result['test_score']:.4f}")
            print(f"  Top 10 features: {', '.join(result['selected_features'][:10])}")
            if len(result['selected_features']) > 10:
                print(f"  ... and {len(result['selected_features']) - 10} more")
    
    def save_results(self):
        """Save all results to files"""
        # Save summary table
        summary_df = self.create_summary_table()
        summary_df.to_csv('rfecv_summary.csv', index=False)
        
        # Save detailed feature selections
        for name, result in self.results.items():
            # Selected features
            selected_df = pd.DataFrame({
                'Feature': result['selected_features'],
                'Selected': True
            })
            selected_df.to_csv(f'rfecv_selected_features_{name.replace(" ", "_").lower()}.csv', index=False)
            
            # Full ranking
            ranking_df = pd.DataFrame({
                'Feature': [f[0] for f in result['feature_rankings']],
                'Ranking': [f[1] for f in result['feature_rankings']],
                'Selected': [f[1] == 1 for f in result['feature_rankings']]
            })
            ranking_df.to_csv(f'rfecv_feature_rankings_{name.replace(" ", "_").lower()}.csv', index=False)
        
        print("\nResults saved:")
        print("- rfecv_summary.csv")
        print("- rfecv_selected_features_*.csv")
        print("- rfecv_feature_rankings_*.csv")
        print("- rfecv_analysis_plots.png")

def main():
    """Main function to run RFECV analysis"""
    print("RFECV Feature Selection Analysis")
    print("="*50)
    
    # Load data
    try:
        data = pd.read_csv('training_data.csv')
        print(f"Data loaded successfully. Shape: {data.shape}")
    except FileNotFoundError:
        print("Error: 'training_data.csv' not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize analyzer
    target_column = 'risk_level'
    
    if target_column not in data.columns:
        print(f"Error: Target column '{target_column}' not found.")
        print(f"Available columns: {list(data.columns)}")
        return
    
    analyzer = RFECVAnalyzer(data, target_column)
    
    # Define estimators to test
    estimators = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM Linear': SVC(kernel='linear', random_state=42, probability=True)
    }
    
    # Run analysis
    analyzer.run_rfecv_analysis(estimators=estimators, cv_folds=5, scoring='accuracy')
    
    # Create visualizations
    analyzer.plot_rfecv_results()
    
    # Generate reports
    summary_df = analyzer.create_summary_table()
    print("\n" + "="*80)
    print("RFECV SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    analyzer.create_detailed_feature_report()
    
    # Save results
    analyzer.save_results()
    
    print("\nRFECV analysis completed successfully!")

if __name__ == "__main__":
    main()