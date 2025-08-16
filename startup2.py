import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .failure-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the training data"""
    try:
        # For demo purposes, we'll create sample data structure
        # In real implementation, you would load your actual CSV file
        # df = pd.read_csv('startup_data.csv')
        
        # Create sample data structure based on your features
        np.random.seed(42)  # For reproducible results
        n_samples = 1000
        
        sample_data = {
            'latitude': np.random.normal(37.7749, 5, n_samples),
            'longitude': np.random.normal(-122.4194, 5, n_samples),
            'zip_code': np.random.randint(10000, 99999, n_samples),
            'age_first_funding_year': np.random.randint(0, 10, n_samples),
            'age_last_funding_year': np.random.randint(0, 15, n_samples),
            'relationships': np.random.randint(0, 50, n_samples),
            'funding_rounds': np.random.randint(1, 10, n_samples),
            'funding_total_usd': np.random.lognormal(15, 2, n_samples),
            'milestones': np.random.randint(0, 20, n_samples),
            'is_CA': np.random.binomial(1, 0.3, n_samples),
            'is_NY': np.random.binomial(1, 0.2, n_samples),
            'is_MA': np.random.binomial(1, 0.1, n_samples),
            'is_TX': np.random.binomial(1, 0.1, n_samples),
            'is_otherstate': np.random.binomial(1, 0.3, n_samples),
            'is_software': np.random.binomial(1, 0.4, n_samples),
            'is_web': np.random.binomial(1, 0.3, n_samples),
            'is_mobile': np.random.binomial(1, 0.2, n_samples),
            'is_enterprise': np.random.binomial(1, 0.25, n_samples),
            'is_advertising': np.random.binomial(1, 0.1, n_samples),
            'is_gamesvideo': np.random.binomial(1, 0.05, n_samples),
            'is_ecommerce': np.random.binomial(1, 0.15, n_samples),
            'is_biotech': np.random.binomial(1, 0.08, n_samples),
            'is_consulting': np.random.binomial(1, 0.05, n_samples),
            'is_othercategory': np.random.binomial(1, 0.2, n_samples),
            'has_VC': np.random.binomial(1, 0.6, n_samples),
            'has_angel': np.random.binomial(1, 0.4, n_samples),
            'has_roundA': np.random.binomial(1, 0.5, n_samples),
            'has_roundB': np.random.binomial(1, 0.3, n_samples),
            'has_roundC': np.random.binomial(1, 0.2, n_samples),
            'has_roundD': np.random.binomial(1, 0.1, n_samples),
            'avg_participants': np.random.normal(5, 2, n_samples),
            'is_top500': np.random.binomial(1, 0.05, n_samples),
            'status': np.random.binomial(1, 0.7, n_samples)  # 70% success rate
        }
        
        df = pd.DataFrame(sample_data)
        
        # Ensure all values are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def train_model():
    """Train the Random Forest model"""
    df = load_and_prepare_data()
    if df is None:
        return None, None
    
    # Prepare features and target
    X = df.drop('status', axis=1)
    y = df['status']
    
    # Handle datetime columns
    if 'founded_at' in X.columns:
        X['founded_year'] = pd.to_datetime(X['founded_at']).dt.year
        X = X.drop('founded_at', axis=1)
    
    if 'first_funding_at' in X.columns:
        X['first_funding_year'] = pd.to_datetime(X['first_funding_at']).dt.year
        X = X.drop('first_funding_at', axis=1)
    
    if 'last_funding_at' in X.columns:
        X['last_funding_year'] = pd.to_datetime(X['last_funding_at']).dt.year
        X = X.drop('last_funding_at', axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=300, 
        class_weight='balanced', 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, X.columns.tolist()

def main():
    st.markdown('<h1 class="main-header">🚀 Startup Success Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict your startup's potential for success using machine learning")
    
    # Load model
    with st.spinner("Loading prediction model..."):
        try:
            model, scaler, accuracy, feature_columns = train_model()
            if model is None:
                st.error("Failed to load the prediction model. Please check your data file.")
                return
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return
    
    # Sidebar for model info
    st.sidebar.markdown("## Model Information")
    st.sidebar.info(f"**Model Accuracy:** {accuracy:.2%}")
    st.sidebar.markdown("**Algorithm:** Random Forest Classifier")
    st.sidebar.markdown("**Features:** Location, Funding, Category, etc.")
    
    # Main form
    st.markdown('<h2 class="sub-header">📋 Company Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📍 Location & Basic Info")
        
        # State selection
        state_options = ['California', 'New York', 'Massachusetts', 'Texas', 'Other']
        selected_state = st.selectbox("State", state_options)
        
        # Location details
        latitude = st.number_input("Latitude", value=37.7749, format="%.4f")
        longitude = st.number_input("Longitude", value=-122.4194, format="%.4f")
        zip_code = st.number_input("ZIP Code", value=94102, min_value=10000, max_value=99999)
        
        # Dates
        founded_date = st.date_input("Founded Date", value=date(2015, 1, 1))
        first_funding_date = st.date_input("First Funding Date", value=date(2016, 1, 1))
        last_funding_date = st.date_input("Last Funding Date", value=date(2018, 1, 1))
        
        # Relationships and milestones
        relationships = st.number_input("Number of Relationships", value=10, min_value=0)
        milestones = st.number_input("Number of Milestones", value=5, min_value=0)
    
    with col2:
        st.markdown("#### 💰 Funding Information")
        
        funding_rounds = st.number_input("Number of Funding Rounds", value=3, min_value=1)
        funding_total = st.number_input("Total Funding (USD)", value=1000000, min_value=0)
        avg_participants = st.number_input("Average Participants per Round", value=5.0, min_value=0.0)
        
        st.markdown("#### 🏢 Category & Type")
        
        # Category selection (multi-select)
        categories = {
            'Software': 'is_software',
            'Web': 'is_web',
            'Mobile': 'is_mobile',
            'Enterprise': 'is_enterprise',
            'Advertising': 'is_advertising',
            'Games/Video': 'is_gamesvideo',
            'E-commerce': 'is_ecommerce',
            'Biotech': 'is_biotech',
            'Consulting': 'is_consulting',
            'Other': 'is_othercategory'
        }
        
        selected_categories = st.multiselect("Business Categories", list(categories.keys()))
        
        st.markdown("#### 💼 Funding Types")
        
        funding_types = {
            'Venture Capital': 'has_VC',
            'Angel Investment': 'has_angel',
            'Series A': 'has_roundA',
            'Series B': 'has_roundB',
            'Series C': 'has_roundC',
            'Series D': 'has_roundD'
        }
        
        selected_funding_types = st.multiselect("Funding Types Received", list(funding_types.keys()))
        
        is_top500 = st.checkbox("Top 500 Company Recognition")
    
    # Prediction button
    if st.button("🔮 Predict Success", type="primary", use_container_width=True):
        with st.spinner("Analyzing your startup..."):
            try:
                # Prepare input data
                input_data = {}
                
                # Location features
                input_data['latitude'] = latitude
                input_data['longitude'] = longitude
                input_data['zip_code'] = zip_code
                
                # State encoding
                input_data['is_CA'] = 1 if selected_state == 'California' else 0
                input_data['is_NY'] = 1 if selected_state == 'New York' else 0
                input_data['is_MA'] = 1 if selected_state == 'Massachusetts' else 0
                input_data['is_TX'] = 1 if selected_state == 'Texas' else 0
                input_data['is_otherstate'] = 1 if selected_state == 'Other' else 0
                
                # Date features
                input_data['founded_year'] = founded_date.year
                input_data['first_funding_year'] = first_funding_date.year
                input_data['last_funding_year'] = last_funding_date.year
                
                # Calculate age features
                current_year = datetime.now().year
                input_data['age_first_funding_year'] = first_funding_date.year - founded_date.year
                input_data['age_last_funding_year'] = last_funding_date.year - founded_date.year
                
                # Other numeric features
                input_data['relationships'] = relationships
                input_data['funding_rounds'] = funding_rounds
                input_data['funding_total_usd'] = funding_total
                input_data['milestones'] = milestones
                input_data['avg_participants'] = avg_participants
                input_data['is_top500'] = 1 if is_top500 else 0
                
                # Category encoding
                for category, column in categories.items():
                    input_data[column] = 1 if category in selected_categories else 0
                
                # Funding type encoding
                for funding_type, column in funding_types.items():
                    input_data[column] = 1 if funding_type in selected_funding_types else 0
                
                # Create DataFrame with correct column order
                input_df = pd.DataFrame([input_data])
                
                # Ensure all required columns are present
                for col in feature_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Reorder columns to match training data
                input_df = input_df[feature_columns]
                
                # Scale the input
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Display results
                st.markdown("---")
                st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
                
                if prediction == 1:
                    st.markdown(f'''
                    <div class="prediction-box success-box">
                        <h3>✅ HIGH SUCCESS PROBABILITY</h3>
                        <p><strong>Success Probability:</strong> {prediction_proba[1]:.1%}</p>
                        <p>Your startup shows strong indicators for success! Keep up the great work.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="prediction-box failure-box">
                        <h3>⚠️ NEEDS IMPROVEMENT</h3>
                        <p><strong>Success Probability:</strong> {prediction_proba[1]:.1%}</p>
                        <p>Consider strengthening key areas like funding strategy, partnerships, or market positioning.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Feature importance visualization
                feature_importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False).head(10)
                
                st.markdown("### 📊 Key Success Factors")
                st.bar_chart(importance_df.set_index('Feature'))
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                if prediction_proba[1] < 0.6:
                    st.markdown("""
                    - **Increase funding rounds** and total funding amount
                    - **Build more strategic relationships** and partnerships
                    - **Achieve more milestones** to demonstrate progress
                    - **Consider relocating** to a major startup hub if possible
                    - **Diversify funding sources** (VC, Angel, Series rounds)
                    """)
                else:
                    st.markdown("""
                    - **Maintain current trajectory** - you're on the right path!
                    - **Scale strategically** to sustain growth
                    - **Build strong partnerships** for long-term success
                    - **Consider expansion** to new markets or categories
                    - **Prepare for later funding rounds** to fuel growth
                    """)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Please ensure all fields are filled correctly.")

if __name__ == "__main__":
    main()