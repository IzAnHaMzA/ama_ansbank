// AMA Answer Bank JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Navigation functionality
    const navItems = document.querySelectorAll('.nav-item');
    const contentSections = document.querySelectorAll('.content-section');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetSection = item.getAttribute('data-section');
            
            // Remove active class from all nav items and sections
            navItems.forEach(nav => nav.classList.remove('active'));
            contentSections.forEach(section => section.classList.remove('active'));
            
            // Add active class to clicked nav item and corresponding section
            item.classList.add('active');
            document.getElementById(targetSection).classList.add('active');
        });
    });

    // Search functionality
    const searchInput = document.getElementById('searchInput');
    const searchResults = document.getElementById('searchResults');
    
    searchInput.addEventListener('input', function() {
        const query = this.value.toLowerCase().trim();
        if (query.length < 2) {
            searchResults.style.display = 'none';
            return;
        }
        
        const results = performSearch(query);
        displaySearchResults(results);
    });

    // Close search results when clicking outside
    document.addEventListener('click', function(e) {
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.style.display = 'none';
        }
    });

    // Initialize content
    initializeContent();
});

// Question and Answer Data
const questionsData = {
    twoMark: [
        {
            id: 1,
            question: "Define Multiclass classification",
            answer: "Multiclass classification is a type of machine learning problem where the algorithm needs to classify instances into one of three or more classes. Unlike binary classification which has only two classes, multiclass classification deals with multiple discrete categories. Each instance belongs to exactly one class from a set of mutually exclusive classes. Common algorithms used include Logistic Regression (with multinomial variant), Decision Trees, Random Forest, Support Vector Machines, and Neural Networks."
        },
        {
            id: 2,
            question: "State any four important supervised machine learning algorithms",
            answer: "Four important supervised machine learning algorithms are:<br><br><strong>1. Linear Regression:</strong> Used for predicting continuous numerical values by finding the best linear relationship between input features and target variable.<br><br><strong>2. Logistic Regression:</strong> Used for binary and multiclass classification problems, predicts probabilities using the logistic function.<br><br><strong>3. Decision Tree:</strong> Creates a tree-like model of decisions to classify or predict outcomes, easy to interpret and visualize.<br><br><strong>4. Random Forest:</strong> An ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting."
        },
        {
            id: 3,
            question: "Define term MSE, RMSE, MAE",
            answer: "These are common evaluation metrics for regression models:<br><br><strong>MSE (Mean Squared Error):</strong> The average of the squared differences between predicted and actual values. MSE = (1/n) × Σ(actual - predicted)². It penalizes larger errors more heavily.<br><br><strong>RMSE (Root Mean Squared Error):</strong> The square root of MSE. RMSE = √MSE. It's in the same units as the target variable, making it easier to interpret.<br><br><strong>MAE (Mean Absolute Error):</strong> The average of absolute differences between predicted and actual values. MAE = (1/n) × Σ|actual - predicted|. It treats all errors equally regardless of their magnitude."
        },
        {
            id: 4,
            question: "Define different unsupervised algorithm",
            answer: "Unsupervised learning algorithms work with data that has no labeled outputs. Key algorithms include:<br><br><strong>1. K-Means Clustering:</strong> Groups data into k clusters by minimizing within-cluster sum of squares. Each data point belongs to the cluster with the nearest centroid.<br><br><strong>2. Hierarchical Clustering:</strong> Creates a tree of clusters using either agglomerative (bottom-up) or divisive (top-down) approaches.<br><br><strong>3. DBSCAN:</strong> Density-based clustering that groups together points that are closely packed, identifying outliers as noise.<br><br><strong>4. Principal Component Analysis (PCA):</strong> Reduces dimensionality by finding principal components that capture maximum variance in the data."
        },
        {
            id: 5,
            question: "Define Training Dataset & Testing Dataset",
            answer: "<strong>Training Dataset:</strong> A subset of data used to train machine learning models. It contains labeled examples that the algorithm learns from to identify patterns and relationships. Typically represents 70-80% of the total dataset.<br><br><strong>Testing Dataset:</strong> A separate subset of data used to evaluate the trained model's performance on unseen data. It contains examples not used during training to assess the model's generalization ability. Typically represents 20-30% of the total dataset. The separation ensures unbiased evaluation of model performance."
        },
        {
            id: 6,
            question: "What is Data Cleaning",
            answer: "Data Cleaning is the process of detecting, correcting, or removing corrupt, incomplete, or irrelevant data from a dataset:<br><br><strong>Common Issues:</strong> Missing values, duplicates, outliers, inconsistent formatting, and incorrect data types.<br><br><strong>Techniques:</strong> Imputation for missing values, removal of duplicates, outlier detection and treatment, data type conversion, and standardization.<br><br><strong>Importance:</strong> Ensures data quality, improves model performance, reduces bias, and enables accurate analysis.<br><br><strong>Tools:</strong> Python pandas, R dplyr, SQL, and specialized data cleaning software."
        },
        {
            id: 7,
            question: "Define Linear Regression",
            answer: "Linear Regression is a supervised machine learning algorithm used for predicting continuous numerical values. It assumes a linear relationship between input features and the target variable.<br><br><strong>Mathematical Form:</strong> y = mx + b (simple) or y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ (multiple)<br><br><strong>Goal:</strong> Find the best line (or hyperplane) that minimizes the sum of squared differences between predicted and actual values.<br><br><strong>Applications:</strong> Price prediction, sales forecasting, risk assessment, and trend analysis.<br><br><strong>Advantages:</strong> Simple, interpretable, fast training, and provides statistical significance testing."
        },
        {
            id: 8,
            question: "What is Binary Classification?",
            answer: "Binary Classification is a type of supervised learning problem where the goal is to classify instances into exactly two classes or categories:<br><br><strong>Characteristics:</strong> Only two possible outcomes (e.g., Yes/No, True/False, Spam/Not Spam, Fraud/Not Fraud)<br><br><strong>Output:</strong> Usually represented as 0 or 1, or probabilities between 0 and 1<br><br><strong>Common Algorithms:</strong> Logistic Regression, Support Vector Machines, Decision Trees, Naive Bayes<br><br><strong>Evaluation Metrics:</strong> Accuracy, Precision, Recall, F1-Score, AUC-ROC<br><br><strong>Examples:</strong> Email spam detection, medical diagnosis (disease/no disease), credit approval (approved/denied)"
        },
        {
            id: 9,
            question: "Define the term multiple linear regression & logistic regression",
            answer: "<strong>Multiple Linear Regression:</strong> An extension of simple linear regression that uses multiple independent variables to predict a continuous dependent variable. The relationship is modeled as: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε<br><br><strong>Logistic Regression:</strong> A classification algorithm that predicts the probability of an instance belonging to a particular class using the logistic function. Despite its name, it's used for classification, not regression. It uses the sigmoid function to map any input to a value between 0 and 1.<br><br><strong>Key Difference:</strong> Multiple linear regression predicts continuous values, while logistic regression predicts probabilities for classification problems."
        },
        {
            id: 10,
            question: "Define supervised and unsupervised learning",
            answer: "<strong>Supervised Learning:</strong> A type of machine learning where the algorithm learns from labeled training data (input-output pairs). The model learns to map inputs to correct outputs and can make predictions on new, unseen data. Examples include classification and regression problems.<br><br><strong>Unsupervised Learning:</strong> A type of machine learning where the algorithm finds hidden patterns in data without labeled examples. The goal is to discover the underlying structure or distribution in the data. Examples include clustering, dimensionality reduction, and association rule mining.<br><br><strong>Key Difference:</strong> Supervised learning uses labeled data, while unsupervised learning works with unlabeled data to find patterns."
        }
    ],
    fourMark: [
        {
            id: 1,
            question: "Differentiate between overfitting and underfitting",
            answer: "<strong>Overfitting vs Underfitting:</strong><br><br><strong>Overfitting:</strong><br>• Model complexity is too high relative to the data<br>• Captures noise and irrelevant patterns in training data<br>• High training accuracy, low validation accuracy<br>• Large gap between training and validation performance<br>• Model memorizes training data instead of learning general patterns<br><br><strong>Underfitting:</strong><br>• Model complexity is too low relative to the data<br>• Fails to capture underlying patterns in the data<br>• Low accuracy on both training and validation data<br>• Small gap between training and validation performance<br>• Model is too simple for the problem complexity<br><br><strong>Solutions:</strong><br>• Overfitting: Regularization, dropout, early stopping, more data<br>• Underfitting: Increase model complexity, add features, reduce regularization"
        },
        {
            id: 2,
            question: "State and explain different types of learning",
            answer: "<strong>Types of Machine Learning:</strong><br><br><strong>1. Supervised Learning:</strong><br>• Uses labeled training data (input-output pairs)<br>• Goal: Learn mapping from inputs to outputs<br>• Examples: Classification, Regression<br>• Algorithms: Linear Regression, Decision Trees, SVM<br><br><strong>2. Unsupervised Learning:</strong><br>• Uses unlabeled data<br>• Goal: Find hidden patterns or structures<br>• Examples: Clustering, Dimensionality Reduction<br>• Algorithms: K-Means, PCA, DBSCAN<br><br><strong>3. Semi-supervised Learning:</strong><br>• Uses both labeled and unlabeled data<br>• Combines benefits of supervised and unsupervised learning<br>• Useful when labeled data is scarce<br><br><strong>4. Reinforcement Learning:</strong><br>• Learns through interaction with environment<br>• Uses rewards and penalties to guide learning<br>• Examples: Game playing, robotics, autonomous vehicles"
        },
        {
            id: 3,
            question: "Implement simple linear regression algorithm in python",
            answer: "<strong>Simple Linear Regression Implementation in Python:</strong><br><br><strong>Mathematical Foundation:</strong><br>• Equation: y = mx + b<br>• Where m is slope, b is y-intercept<br>• Goal: Find best line through data points<br><br><strong>Cost Function (Mean Squared Error):</strong><br>• MSE = (1/n) × Σ(y_actual - y_predicted)²<br><br><strong>Python Implementation:</strong><br><div class='code-block'>import numpy as np<br>import matplotlib.pyplot as plt<br><br>class SimpleLinearRegression:<br>    def __init__(self, learning_rate=0.01, epochs=1000):<br>        self.learning_rate = learning_rate<br>        self.epochs = epochs<br>        self.m = 0  # slope<br>        self.b = 0  # y-intercept<br>    <br>    def fit(self, X, y):<br>        n = len(X)<br>        <br>        for epoch in range(self.epochs):<br>            # Forward pass<br>            y_pred = self.m * X + self.b<br>            <br>            # Calculate cost<br>            cost = np.mean((y - y_pred) ** 2)<br>            <br>            # Calculate gradients<br>            dm = -(2/n) * np.sum(X * (y - y_pred))<br>            db = -(2/n) * np.sum(y - y_pred)<br>            <br>            # Update parameters<br>            self.m -= self.learning_rate * dm<br>            self.b -= self.learning_rate * db<br>            <br>            if epoch % 100 == 0:<br>                print(f'Epoch {epoch}, Cost: {cost:.4f}')<br>    <br>    def predict(self, X):<br>        return self.m * X + self.b<br><br># Usage example<br>X = np.array([1, 2, 3, 4, 5])<br>y = np.array([2, 4, 6, 8, 10])<br><br>model = SimpleLinearRegression()<br>model.fit(X, y)<br>predictions = model.predict(X)</div>"
        },
        {
            id: 4,
            question: "Describe in detail data cleaning",
            answer: "<strong>Data Cleaning - Comprehensive Guide:</strong><br><br><strong>1. Handling Missing Values:</strong><br>• <strong>Imputation:</strong> Replace with mean, median, mode<br>• <strong>Deletion:</strong> Remove records with missing values<br>• <strong>Forward/Backward Fill:</strong> For time series data<br>• <strong>Advanced:</strong> Regression imputation, multiple imputation<br><br><strong>2. Removing Duplicates:</strong><br>• Identify duplicate records based on key attributes<br>• Decide on duplicate handling strategy<br>• Remove exact duplicates or merge similar records<br>• Use fuzzy matching for near-duplicates<br><br><strong>3. Data Type Conversion:</strong><br>• Convert strings to numbers where appropriate<br>• Standardize date formats<br>• Convert categorical data to appropriate types<br>• Handle mixed data types in single columns<br><br><strong>4. Outlier Detection:</strong><br>• <strong>Statistical Methods:</strong> Z-score, IQR method<br>• <strong>Visualization:</strong> Box plots, scatter plots<br>• <strong>Machine Learning:</strong> Isolation Forest, DBSCAN<br>• <strong>Domain Knowledge:</strong> Expert-defined thresholds<br><br><strong>5. Data Standardization:</strong><br>• Normalize numerical data (0-1 scale)<br>• Standardize text formats<br>• Consistent naming conventions<br>• Handle encoding issues<br><br><strong>Python Example:</strong><br><div class='code-block'>import pandas as pd<br>import numpy as np<br><br># Load data<br>df = pd.read_csv('data.csv')<br><br># Handle missing values<br>df['column'].fillna(df['column'].mean(), inplace=True)<br><br># Remove duplicates<br>df.drop_duplicates(inplace=True)<br><br># Convert data types<br>df['date_column'] = pd.to_datetime(df['date_column'])<br><br># Remove outliers using IQR<br>Q1 = df['column'].quantile(0.25)<br>Q3 = df['column'].quantile(0.75)<br>IQR = Q3 - Q1<br>df = df[(df['column'] >= Q1 - 1.5*IQR) & (df['column'] <= Q3 + 1.5*IQR)]</div>"
        },
        {
            id: 5,
            question: "What is training and testing of model",
            answer: "<strong>Training and Testing of Machine Learning Models:</strong><br><br><strong>Training Phase:</strong><br>• <strong>Purpose:</strong> Teach the model to recognize patterns in data<br>• <strong>Process:</strong> Model learns from labeled training data<br>• <strong>Data:</strong> 70-80% of total dataset<br>• <strong>Goal:</strong> Minimize training error by adjusting parameters<br>• <strong>Output:</strong> Trained model with learned weights/parameters<br><br><strong>Testing Phase:</strong><br>• <strong>Purpose:</strong> Evaluate model performance on unseen data<br>• <strong>Process:</strong> Model makes predictions on test data<br>• <strong>Data:</strong> 20-30% of total dataset (unseen during training)<br>• <strong>Goal:</strong> Assess generalization ability<br>• <strong>Output:</strong> Performance metrics (accuracy, precision, recall)<br><br><strong>Key Differences:</strong><br>• Training uses labeled data, testing uses unseen data<br>• Training adjusts parameters, testing evaluates performance<br>• Training error should decrease, testing error indicates generalization<br><br><strong>Best Practices:</strong><br>• Random splitting to avoid bias<br>• Stratified splitting for classification problems<br>• No data leakage between training and testing sets<br>• Representative sampling of the population<br><br><strong>Python Example:</strong><br><div class='code-block'>from sklearn.model_selection import train_test_split<br>from sklearn.linear_model import LinearRegression<br>from sklearn.metrics import mean_squared_error<br><br># Split data<br>X_train, X_test, y_train, y_test = train_test_split(<br>    X, y, test_size=0.2, random_state=42<br>)<br><br># Train model<br>model = LinearRegression()<br>model.fit(X_train, y_train)<br><br># Test model<br>y_pred = model.predict(X_test)<br>mse = mean_squared_error(y_test, y_pred)</div>"
        },
        {
            id: 6,
            question: "What is cross validation and why is it important in model evaluation",
            answer: "<strong>Cross Validation in Model Evaluation:</strong><br><br><strong>Definition:</strong><br>Cross validation is a statistical method used to evaluate machine learning models by partitioning the dataset into multiple subsets and testing the model on each subset.<br><br><strong>Types of Cross Validation:</strong><br>• <strong>K-Fold CV:</strong> Divides data into k equal folds<br>• <strong>Leave-One-Out CV:</strong> Uses n-1 samples for training, 1 for testing<br>• <strong>Stratified K-Fold:</strong> Maintains class distribution in each fold<br>• <strong>Time Series CV:</strong> Respects temporal order<br><br><strong>K-Fold Cross Validation Process:</strong><br>1. Split dataset into k equal parts (folds)<br>2. Train model on k-1 folds<br>3. Test model on remaining fold<br>4. Repeat k times with different test fold<br>5. Average the performance across all folds<br><br><strong>Why Cross Validation is Important:</strong><br>• <strong>Robust Performance Estimate:</strong> Reduces variance in performance estimates<br>• <strong>Prevents Overfitting:</strong> Tests model on multiple data subsets<br>• <strong>Better Model Selection:</strong> Helps choose best hyperparameters<br>• <strong>Data Efficiency:</strong> Uses all data for both training and testing<br>• <strong>Bias Reduction:</strong> Provides unbiased estimate of model performance<br><br><strong>Python Example:</strong><br><div class='code-block'>from sklearn.model_selection import cross_val_score, KFold<br>from sklearn.linear_model import LinearRegression<br><br># Create model<br>model = LinearRegression()<br><br># Perform 5-fold cross validation<br>cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')<br><br>print(f'CV Scores: {cv_scores}')<br>print(f'Mean CV Score: {cv_scores.mean():.4f}')<br>print(f'Std CV Score: {cv_scores.std():.4f}')</div>"
        },
        {
            id: 7,
            question: "Explain the concept of one vs one and one vs rest in multiclass classification",
            answer: "<strong>Multiclass Classification Strategies:</strong><br><br><strong>One vs Rest (OvR) / One vs All (OvA):</strong><br>• <strong>Approach:</strong> Train one binary classifier per class<br>• <strong>Process:</strong> For each class, treat it as positive and all others as negative<br>• <strong>Number of Classifiers:</strong> k classifiers for k classes<br>• <strong>Prediction:</strong> Choose class with highest confidence score<br><br><strong>Example (3 classes: A, B, C):</strong><br>• Classifier 1: A vs (B, C)<br>• Classifier 2: B vs (A, C)<br>• Classifier 3: C vs (A, B)<br><br><strong>One vs One (OvO):</strong><br>• <strong>Approach:</strong> Train binary classifier for each pair of classes<br>• <strong>Process:</strong> For each pair, train classifier to distinguish between them<br>• <strong>Number of Classifiers:</strong> k(k-1)/2 classifiers for k classes<br>• <strong>Prediction:</strong> Use voting mechanism (most votes wins)<br><br><strong>Example (3 classes: A, B, C):</strong><br>• Classifier 1: A vs B<br>• Classifier 2: A vs C<br>• Classifier 3: B vs C<br><br><strong>Comparison:</strong><br>• <strong>OvR:</strong> Fewer classifiers, faster training, may have class imbalance<br>• <strong>OvO:</strong> More classifiers, better for imbalanced datasets, slower training<br><br><strong>Python Example:</strong><br><div class='code-block'>from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier<br>from sklearn.svm import SVC<br><br># One vs Rest<br>ovr_classifier = OneVsRestClassifier(SVC())<br>ovr_classifier.fit(X_train, y_train)<br><br># One vs One<br>ovo_classifier = OneVsOneClassifier(SVC())<br>ovo_classifier.fit(X_train, y_train)</div>"
        },
        {
            id: 8,
            question: "What is confusion matrix",
            answer: "<strong>Confusion Matrix - Model Performance Evaluation:</strong><br><br><strong>Definition:</strong><br>A confusion matrix is a table used to evaluate the performance of a classification model. It shows the actual vs predicted classifications in a matrix format.<br><br><strong>Structure (Binary Classification):</strong><br><div class='code-block'>                Predicted<br>Actual     Positive  Negative<br>Positive      TP       FN<br>Negative      FP       TN</div><br><strong>Key Metrics:</strong><br>• <strong>TP (True Positive):</strong> Correctly predicted positive cases<br>• <strong>TN (True Negative):</strong> Correctly predicted negative cases<br>• <strong>FP (False Positive):</strong> Incorrectly predicted positive cases (Type I error)<br>• <strong>FN (False Negative):</strong> Incorrectly predicted negative cases (Type II error)<br><br><strong>Derived Metrics:</strong><br>• <strong>Accuracy:</strong> (TP + TN) / (TP + TN + FP + FN)<br>• <strong>Precision:</strong> TP / (TP + FP)<br>• <strong>Recall/Sensitivity:</strong> TP / (TP + FN)<br>• <strong>Specificity:</strong> TN / (TN + FP)<br>• <strong>F1-Score:</strong> 2 × (Precision × Recall) / (Precision + Recall)<br><br><strong>Multiclass Confusion Matrix:</strong><br>• Extends to multiple classes<br>• Shows performance for each class<br>• Helps identify which classes are confused with each other<br><br><strong>Python Example:</strong><br><div class='code-block'>from sklearn.metrics import confusion_matrix, classification_report<br>import seaborn as sns<br>import matplotlib.pyplot as plt<br><br># Generate confusion matrix<br>cm = confusion_matrix(y_true, y_pred)<br><br># Visualize<br>sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')<br>plt.title('Confusion Matrix')<br>plt.show()<br><br># Detailed report<br>print(classification_report(y_true, y_pred))</div>"
        },
        {
            id: 9,
            question: "What is Decision tree explain with suitable diagram",
            answer: "<strong>Decision Tree - Tree-Based Classification:</strong><br><br><strong>Definition:</strong><br>A Decision Tree is a flowchart-like tree structure where each internal node represents a feature test, each branch represents the outcome of the test, and each leaf node represents a class label or decision.<br><br><strong>Key Components:</strong><br>• <strong>Root Node:</strong> Topmost node, represents best feature to split on<br>• <strong>Decision Nodes:</strong> Internal nodes that represent feature tests<br>• <strong>Branches:</strong> Represent outcomes of feature tests (Yes/No, True/False)<br>• <strong>Leaf Nodes:</strong> Terminal nodes that represent final class predictions<br><br><strong>Algorithm Steps:</strong><br>1. Select best feature to split on (using information gain, Gini impurity)<br>2. Create branches for each possible outcome<br>3. Recursively repeat for each branch<br>4. Stop when stopping criteria met (max depth, min samples)<br><br><strong>Advantages:</strong><br>• Easy to understand and interpret<br>• Handles both numerical and categorical data<br>• Requires little data preprocessing<br>• Can handle non-linear relationships<br><br><strong>Disadvantages:</strong><br>• Prone to overfitting<br>• Sensitive to small changes in data<br>• Can create biased trees if classes are imbalanced<br><br><strong>Python Example:</strong><br><div class='code-block'>from sklearn.tree import DecisionTreeClassifier, plot_tree<br>import matplotlib.pyplot as plt<br><br># Create and train model<br>dt = DecisionTreeClassifier(max_depth=3, random_state=42)<br>dt.fit(X_train, y_train)<br><br># Visualize tree<br>plt.figure(figsize=(12, 8))<br>plot_tree(dt, feature_names=feature_names, class_names=class_names, filled=True)<br>plt.show()</div><br><br><strong>Visual Diagram:</strong><br><div style='text-align: center; margin: 20px 0;'><img src='Screenshot 2025-10-04 124006.png' alt='Decision Tree Diagram' style='max-width: 100%; height: auto; border: 2px solid #0ff; border-radius: 10px; box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);'></div>"
        },
        {
            id: 10,
            question: "What is Hierarchical Clustering explain with diagram",
            answer: "<strong>Hierarchical Clustering - Tree-Based Clustering:</strong><br><br><strong>Definition:</strong><br>Hierarchical Clustering creates a tree of clusters called a dendrogram, where each level represents a different clustering of the data.<br><br><strong>Types:</strong><br>• <strong>Agglomerative (Bottom-up):</strong> Start with each point as a cluster, merge closest pairs<br>• <strong>Divisive (Top-down):</strong> Start with all points in one cluster, recursively split<br><br><strong>Agglomerative Process:</strong><br>1. Start with each data point as its own cluster<br>2. Find two closest clusters<br>3. Merge them into a new cluster<br>4. Repeat until only one cluster remains<br><br><strong>Linkage Criteria:</strong><br>• <strong>Single Linkage:</strong> Minimum distance between clusters<br>• <strong>Complete Linkage:</strong> Maximum distance between clusters<br>• <strong>Average Linkage:</strong> Average distance between clusters<br>• <strong>Ward Linkage:</strong> Minimizes within-cluster variance<br><br><strong>Advantages:</strong><br>• No need to specify number of clusters<br>• Produces interpretable dendrogram<br>• Deterministic results<br>• Can handle any distance metric<br><br><strong>Disadvantages:</strong><br>• Computationally expensive O(n³)<br>• Sensitive to noise and outliers<br>• Difficult to handle large datasets<br><br><strong>Python Example:</strong><br><div class='code-block'>from sklearn.cluster import AgglomerativeClustering<br>from scipy.cluster.hierarchy import dendrogram, linkage<br>import matplotlib.pyplot as plt<br><br># Perform clustering<br>clustering = AgglomerativeClustering(n_clusters=3)<br>labels = clustering.fit_predict(X)<br><br># Create dendrogram<br>linkage_matrix = linkage(X, method='ward')<br>dendrogram(linkage_matrix)<br>plt.title('Hierarchical Clustering Dendrogram')<br>plt.show()</div><br><br><strong>Visual Diagram:</strong><br><div style='text-align: center; margin: 20px 0;'><img src='Screenshot 2025-10-04 124057.png' alt='Hierarchical Clustering Diagram' style='max-width: 100%; height: auto; border: 2px solid #0ff; border-radius: 10px; box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);'></div>"
        }
    ]
};

const conceptsData = [
    {
        title: "Machine Learning Fundamentals",
        icon: "fas fa-brain",
        description: "Core concepts and principles of machine learning, including definitions, importance, and applications in modern technology.",
        features: [
            "Definition and scope of ML",
            "Difference from traditional programming",
            "Real-world applications",
            "Historical evolution and milestones"
        ]
    },
    {
        title: "ML Life Cycle",
        icon: "fas fa-sync-alt",
        description: "The complete process of developing machine learning solutions from data collection to deployment.",
        features: [
            "7-step development process",
            "Data gathering and preparation",
            "Model training and testing",
            "Deployment and monitoring"
        ]
    },
    {
        title: "Learning Types",
        icon: "fas fa-graduation-cap",
        description: "Different approaches to machine learning based on the availability and type of training data.",
        features: [
            "Supervised Learning",
            "Unsupervised Learning", 
            "Semi-supervised Learning",
            "Reinforcement Learning"
        ]
    },
    {
        title: "Data Types & Forms",
        icon: "fas fa-database",
        description: "Understanding different types of data and various forms of data analysis in machine learning.",
        features: [
            "Numerical, Categorical, Ordinal data",
            "Data Mining techniques",
            "Statistical analysis methods",
            "Data Analytics approaches"
        ]
    },
    {
        title: "Supervised Algorithms",
        icon: "fas fa-chart-line",
        description: "Algorithms that learn from labeled training data to make predictions on new, unseen data.",
        features: [
            "Linear & Logistic Regression",
            "Decision Trees & Random Forest",
            "Support Vector Machines",
            "K-Nearest Neighbors"
        ]
    },
    {
        title: "Unsupervised Algorithms",
        icon: "fas fa-project-diagram",
        description: "Algorithms that find hidden patterns in data without labeled examples.",
        features: [
            "K-Means Clustering",
            "Hierarchical Clustering",
            "DBSCAN clustering",
            "Principal Component Analysis"
        ]
    },
    {
        title: "Neural Networks",
        icon: "fas fa-network-wired",
        description: "Computational models inspired by biological neural networks for complex pattern recognition.",
        features: [
            "Feedforward networks",
            "Convolutional Neural Networks",
            "Recurrent Neural Networks",
            "Deep Learning architectures"
        ]
    },
    {
        title: "Model Evaluation",
        icon: "fas fa-chart-bar",
        description: "Methods and metrics for assessing machine learning model performance and reliability.",
        features: [
            "Cross-validation techniques",
            "Performance metrics (MSE, RMSE, MAE)",
            "Overfitting and underfitting",
            "Bias-variance tradeoff"
        ]
    }
];

// Initialize content
function initializeContent() {
    populateQuestions('2mark', questionsData.twoMark);
    populateQuestions('4mark', questionsData.fourMark);
    populateConcepts();
}

// Populate questions
function populateQuestions(sectionId, questions) {
    const container = document.getElementById(`questions${sectionId}`);
    
    questions.forEach((q, index) => {
        const questionCard = document.createElement('div');
        questionCard.className = 'question-card';
        questionCard.innerHTML = `
            <div class="question-header">
                <div class="question-number">${index + 1}</div>
                <div class="question-text">${q.question}</div>
            </div>
            <div class="question-answer">
                <h4><i class="fas fa-lightbulb"></i>Answer</h4>
                <div class="answer-content">${q.answer}</div>
            </div>
        `;
        container.appendChild(questionCard);
    });
}

// Populate concepts
function populateConcepts() {
    const container = document.getElementById('conceptsGrid');
    
    conceptsData.forEach(concept => {
        const conceptCard = document.createElement('div');
        conceptCard.className = 'concept-card';
        conceptCard.innerHTML = `
            <div class="concept-title">
                <i class="${concept.icon}"></i>
                ${concept.title}
            </div>
            <div class="concept-description">${concept.description}</div>
            <ul class="concept-features">
                ${concept.features.map(feature => `<li>${feature}</li>`).join('')}
            </ul>
        `;
        container.appendChild(conceptCard);
    });
}

// Search functionality
function performSearch(query) {
    const results = [];
    
    // Search in 2-mark questions
    questionsData.twoMark.forEach((q, index) => {
        if (q.question.toLowerCase().includes(query) || q.answer.toLowerCase().includes(query)) {
            results.push({
                type: '2-Mark Question',
                title: q.question,
                content: q.answer.substring(0, 200) + '...',
                section: '2mark'
            });
        }
    });
    
    // Search in 4-mark questions
    questionsData.fourMark.forEach((q, index) => {
        if (q.question.toLowerCase().includes(query) || q.answer.toLowerCase().includes(query)) {
            results.push({
                type: '4-Mark Question',
                title: q.question,
                content: q.answer.substring(0, 200) + '...',
                section: '4mark'
            });
        }
    });
    
    // Search in concepts
    conceptsData.forEach(concept => {
        if (concept.title.toLowerCase().includes(query) || 
            concept.description.toLowerCase().includes(query) ||
            concept.features.some(feature => feature.toLowerCase().includes(query))) {
            results.push({
                type: 'Key Concept',
                title: concept.title,
                content: concept.description,
                section: 'concepts'
            });
        }
    });
    
    return results.slice(0, 10); // Limit to 10 results
}

// Display search results
function displaySearchResults(results) {
    const searchResults = document.getElementById('searchResults');
    
    if (results.length === 0) {
        searchResults.innerHTML = '<div class="search-result-item">No results found</div>';
    } else {
        searchResults.innerHTML = results.map(result => `
            <div class="search-result-item" onclick="navigateToSection('${result.section}')">
                <strong>${result.type}</strong><br>
                ${result.title}<br>
                <small>${result.content}</small>
            </div>
        `).join('');
    }
    
    searchResults.style.display = 'block';
}

// Navigate to section
function navigateToSection(sectionId) {
    // Update navigation
    document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
    document.querySelectorAll('.content-section').forEach(section => section.classList.remove('active'));
    
    // Activate target section
    document.querySelector(`[data-section="${sectionId}"]`).classList.add('active');
    document.getElementById(sectionId).classList.add('active');
    
    // Hide search results
    document.getElementById('searchResults').style.display = 'none';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
