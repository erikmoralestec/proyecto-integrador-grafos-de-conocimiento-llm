"""
Comprehensive Exploratory Data Analysis for User Complaints Dataset
Part of LLM-assisted Knowledge Graph Construction Research Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import networkx as nx

warnings.filterwarnings('ignore')

class ComplaintsEDA:
    """Comprehensive EDA class for user complaints dataset"""
    
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = data_path
        self.df = None
        self.numeric_cols = None
        self.categorical_cols = None
        
    def load_data(self, filter_year=None):
        """Load and preprocess data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path, encoding="ISO-8859-1", low_memory=False)
        
        # Normalize column names
        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.replace(" ", "", regex=False)
            .str.replace("_", "", regex=False)
            .str.upper()
        )
        
        # Apply data transformations
        self._apply_data_transformations()
        
        # Filter by year if specified
        if filter_year is not None:
            self._filter_by_year(filter_year)
        
        # Remove columns with 100% null data
        self._remove_empty_columns()
        
        # Identify column types (excluding identifiers)
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove identifier columns from analysis
        identifier_cols = ['CMPLID', 'ODINO']  # Both are identifiers, not analytical variables
        for col in identifier_cols:
            if col in self.numeric_cols:
                self.numeric_cols.remove(col)
            if col in self.categorical_cols:
                self.categorical_cols.remove(col)
        
        # Add ODINO to categorical if it exists (it's an identifier but can be analyzed as categorical)
        if 'ODINO' in self.df.columns:
            self.categorical_cols.append('ODINO')
        
        print(f"Data loaded: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
    def _filter_by_year(self, year):
        """Filter data by specific year based on FAILDATE"""
        print(f"Filtering data for {year} complaints...")
        
        if 'FAILDATE' in self.df.columns:
            # Ensure FAILDATE is datetime
            self.df['FAILDATE'] = pd.to_datetime(self.df['FAILDATE'], format='%Y%m%d', errors='coerce')
            
            # Filter for specified year
            original_count = len(self.df)
            self.df = self.df[self.df['FAILDATE'].dt.year == year].copy()
            filtered_count = len(self.df)
            
            print(f"Original dataset: {original_count:,} complaints")
            print(f"{year} complaints: {filtered_count:,} complaints")
            print(f"Percentage of {year} data: {filtered_count/original_count*100:.2f}%")
            
            # Show date range for filtered data
            if len(self.df) > 0:
                print(f"Date range for {year}: {self.df['FAILDATE'].min().strftime('%Y-%m-%d')} to {self.df['FAILDATE'].max().strftime('%Y-%m-%d')}")
            else:
                print(f"No complaints found for {year}!")
        else:
            print("FAILDATE column not found. Cannot filter by year.")
            print("Available columns:", self.df.columns.tolist())
        
    def _apply_data_transformations(self):
        """Apply data transformations and data cleaning pipeline"""
        print("Applying data transformations and cleaning...")
        
        # Coerción de numéricos, verificamos que no haya errores
        numeric_columns = ['YEAR', 'INJURED', 'DEATHS', 'MILES', 'OCCURENCES', 'VEHSPEED', 'NUMCYLS']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                print(f"  Converted {col} to numeric")
        
        # Transformamos el formato de algunas fechas
        date_columns = ['FAILDATE', 'DATEA', 'LDATE', 'PURCHDAE', 'MANUFDATE']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], format='%Y%m%d', errors='coerce')
                print(f"  Converted {col} to datetime")
        
        # Strip y uppercase para algunos textos
        text_columns = ['MFRNAME', 'MAKE', 'MODEL', 'COMPONENT', 'CITY', 'STATE', 'DEALERCITY', 'DEALERNAME']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().str.upper()
                self.df[col] = self.df[col].replace(["NAN", "NONE", "NULL"], pd.NA)
                print(f"  Normalized text in {col}")
        
        # Apply comprehensive data cleaning pipeline
        self._apply_data_cleaning_pipeline()
        
        print("Data transformations and cleaning completed!")
    
    def _apply_data_cleaning_pipeline(self):
        """Apply comprehensive data cleaning based on research requirements"""
        print("\n" + "="*60)
        print("APPLYING DATA CLEANING PIPELINE")
        print("="*60)
        
        original_shape = self.df.shape
        print(f"Original dataset shape: {original_shape}")
        
        # 1. Remove rows with YEAR=9999 (tire failures, accessories, unknown products)
        if 'YEAR' in self.df.columns:
            year_9999_mask = (self.df['YEAR'] == 9999)
            year_9999_count = year_9999_mask.sum()
            
            if year_9999_count > 0:
                print(f"\n1. Removing {year_9999_count} rows with YEAR=9999 (tire failures, accessories, unknown products)")
                
                # Show some examples before removal
                if year_9999_count <= 10:
                    print("   Examples of rows being removed:")
                    examples = self.df[year_9999_mask][['MAKE', 'MODEL', 'COMPONENT', 'CMPLDESCR']].head(5)
                    for idx, row in examples.iterrows():
                        print(f"     - {row['MAKE']} {row['MODEL']} | {row['COMPONENT']} | {str(row['CMPLDESCR'])[:100]}...")
                
                self.df = self.df[~year_9999_mask].copy()
                print(f"   Remaining rows: {len(self.df)}")
            else:
                print("1. No rows with YEAR=9999 found")
        
        # 2. Remove seat-related columns
        seat_columns = ['SEATTYPE', 'RESTRAINTTYPE', 'MANUFDATE']
        existing_seat_columns = [col for col in seat_columns if col in self.df.columns]
        
        if existing_seat_columns:
            print(f"\n2. Removing seat-related columns: {existing_seat_columns}")
            self.df = self.df.drop(columns=existing_seat_columns)
        else:
            print("2. No seat-related columns found")
        
        # 3. Remove dealer-related columns
        dealer_columns = ['DEALERTEL', 'DEALERZIP']
        existing_dealer_columns = [col for col in dealer_columns if col in self.df.columns]
        
        if existing_dealer_columns:
            print(f"\n3. Removing dealer-related columns: {existing_dealer_columns}")
            self.df = self.df.drop(columns=existing_dealer_columns)
        else:
            print("3. No dealer-related columns found")
        
        # 4. Filter and remove PROD_TYPE column
        if 'PRODTYPE' in self.df.columns:
            # Count rows to be removed
            prod_type_e_mask = (self.df['PRODTYPE'] == 'E')
            prod_type_empty_mask = self.df['PRODTYPE'].isna() | (self.df['PRODTYPE'] == '')
            rows_to_remove = (prod_type_e_mask | prod_type_empty_mask).sum()
            
            if rows_to_remove > 0:
                print(f"\n4. Removing {rows_to_remove} rows with PROD_TYPE='E' or empty (accessories in transit)")
                self.df = self.df[~(prod_type_e_mask | prod_type_empty_mask)].copy()
                print(f"   Remaining rows: {len(self.df)}")
            else:
                print("4. No rows with PROD_TYPE='E' or empty found")
            
            # Remove PROD_TYPE column entirely
            print("   Removing PROD_TYPE column entirely")
            self.df = self.df.drop(columns=['PRODTYPE'])
        else:
            print("4. PROD_TYPE column not found")
        
        # 5. Remove fuel type column (only 4 records)
        if 'FUELTYPE' in self.df.columns:
            fuel_type_count = self.df['FUELTYPE'].notna().sum()
            print(f"\n5. Removing FUEL_TYPE column (only {fuel_type_count} records)")
            self.df = self.df.drop(columns=['FUELTYPE'])
        else:
            print("5. FUEL_TYPE column not found")
        
        # 6. Remove columns that are now empty after filtering
        empty_columns = ['PURCHDATE', 'LOCOFTIRE', 'DOT', 'DRIVETRAIN']
        existing_empty_columns = [col for col in empty_columns if col in self.df.columns]
        
        if existing_empty_columns:
            # Check which columns are actually empty
            truly_empty = []
            for col in existing_empty_columns:
                non_null_count = self.df[col].notna().sum()
                if non_null_count == 0:
                    truly_empty.append(col)
                else:
                    print(f"   Note: {col} has {non_null_count} non-null values, keeping it")
            
            if truly_empty:
                print(f"\n6. Removing empty columns after filtering: {truly_empty}")
                self.df = self.df.drop(columns=truly_empty)
            else:
                print("6. No columns are empty after filtering")
        else:
            print("6. Target empty columns not found")
        
        # Summary
        final_shape = self.df.shape
        rows_removed = original_shape[0] - final_shape[0]
        cols_removed = original_shape[1] - final_shape[1]
        
        print(f"\n" + "="*60)
        print("CLEANING PIPELINE SUMMARY")
        print("="*60)
        print(f"Original shape: {original_shape}")
        print(f"Final shape: {final_shape}")
        print(f"Rows removed: {rows_removed}")
        print(f"Columns removed: {cols_removed}")
        print(f"Data reduction: {rows_removed/original_shape[0]*100:.2f}% rows, {cols_removed/original_shape[1]*100:.2f}% columns")
        
    def _remove_empty_columns(self):
        """Remove columns that have 100% null data"""
        print("Removing columns with 100% null data...")
        
        # Find columns with 100% missing data
        null_counts = self.df.isnull().sum()
        total_rows = len(self.df)
        empty_columns = null_counts[null_counts == total_rows].index.tolist()
        
        if empty_columns:
            print(f"  Found {len(empty_columns)} columns with 100% null data:")
            for col in empty_columns:
                print(f"    - {col}")
            
            # Remove empty columns
            self.df = self.df.drop(columns=empty_columns)
            print(f"  Removed {len(empty_columns)} empty columns")
            print(f"  Dataset shape after removal: {self.df.shape}")
        else:
            print("  No columns with 100% null data found")
        
    def analyze_missing_values(self):
        """Analyze missing values patterns"""
        print("\n" + "="*50)
        print("MISSING VALUES ANALYSIS")
        print("="*50)
        
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        print("Missing Values Summary:")
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Missing data patterns
        complete_cases = self.df.dropna()
        print(f"\nComplete cases: {len(complete_cases)} ({len(complete_cases)/len(self.df)*100:.2f}%)")
        print(f"Columns with no missing values: {sum(missing_data == 0)}")
        print(f"Columns with all missing values: {sum(missing_data == len(self.df))}")
        
        # Note about empty columns (should be 0 after removal)
        empty_cols = sum(missing_data == len(self.df))
        if empty_cols == 0:
            print("✅ No empty columns (100% missing) - all have been removed")
        else:
            print(f"⚠️  Warning: {empty_cols} columns still have 100% missing data")
        
        # Visualize missing patterns
        self._plot_missing_patterns(missing_df, missing_percentage)
        
    def _plot_missing_patterns(self, missing_df, missing_percentage):
        """Plot missing value patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Missing values heatmap
        missing_sample = self.df.sample(min(1000, len(self.df))).isnull()
        sns.heatmap(missing_sample, cbar=True, yticklabels=False, ax=axes[0,0])
        axes[0,0].set_title('Missing Values Pattern (Sample)')
        
        # Missing values bar chart
        top_missing = missing_df.head(10)
        axes[0,1].barh(range(len(top_missing)), top_missing['Missing_Percentage'])
        axes[0,1].set_yticks(range(len(top_missing)))
        axes[0,1].set_yticklabels(top_missing['Column'])
        axes[0,1].set_xlabel('Missing Percentage')
        axes[0,1].set_title('Top 10 Columns with Missing Values')
        
        # Missing values distribution
        axes[1,0].hist(missing_percentage[missing_percentage > 0], bins=20, edgecolor='black')
        axes[1,0].set_xlabel('Missing Percentage')
        axes[1,0].set_ylabel('Number of Columns')
        axes[1,0].set_title('Distribution of Missing Values')
        
        # Complete cases analysis
        complete_cases = self.df.dropna()
        axes[1,1].pie([len(complete_cases), len(self.df) - len(complete_cases)], 
                      labels=['Complete Cases', 'Incomplete Cases'],
                      autopct='%1.1f%%')
        axes[1,1].set_title('Complete vs Incomplete Cases')
        
        plt.tight_layout()
        plt.show()
        
    def analyze_outliers(self):
        """Analyze outliers in numerical columns"""
        print("\n" + "="*50)
        print("OUTLIERS ANALYSIS")
        print("="*50)
        
        if not self.numeric_cols:
            print("No numerical columns found for outlier analysis")
            return
            
        outlier_summary = []
        
        for col in self.numeric_cols:
            if self.df[col].notna().sum() > 0:  # Only analyze columns with data
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(self.df)) * 100
                
                outlier_summary.append({
                    'Column': col,
                    'Outlier_Count': outlier_count,
                    'Outlier_Percentage': outlier_percentage,
                    'Lower_Bound': lower_bound,
                    'Upper_Bound': upper_bound
                })
        
        outlier_df = pd.DataFrame(outlier_summary)
        print("Outlier Summary:")
        print(outlier_df.sort_values('Outlier_Percentage', ascending=False))
        
        # Plot outliers for top columns
        self._plot_outliers(outlier_df)
        
    def _plot_outliers(self, outlier_df):
        """Plot outlier visualizations"""
        top_outlier_cols = outlier_df.nlargest(6, 'Outlier_Percentage')['Column'].tolist()
        
        if not top_outlier_cols:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(top_outlier_cols):
            if i < 6:  # Limit to 6 plots
                # Box plot
                self.df.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'Outliers in {col}')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(top_outlier_cols), 6):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    def analyze_cardinality(self):
        """Analyze cardinality of categorical variables"""
        print("\n" + "="*50)
        print("CATEGORICAL VARIABLES CARDINALITY")
        print("="*50)
        
        cardinality_summary = []
        
        for col in self.categorical_cols:
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            cardinality_ratio = unique_count / total_count
            
            cardinality_summary.append({
                'Column': col,
                'Unique_Count': unique_count,
                'Cardinality_Ratio': cardinality_ratio,
                'Most_Frequent': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'N/A',
                'Most_Frequent_Count': self.df[col].value_counts().iloc[0] if not self.df[col].empty else 0
            })
        
        cardinality_df = pd.DataFrame(cardinality_summary)
        cardinality_df = cardinality_df.sort_values('Unique_Count', ascending=False)
        
        print("Cardinality Summary:")
        print(cardinality_df)
        
        # Categorize by cardinality
        high_cardinality = cardinality_df[cardinality_df['Cardinality_Ratio'] > 0.5]
        medium_cardinality = cardinality_df[(cardinality_df['Cardinality_Ratio'] > 0.1) & 
                                          (cardinality_df['Cardinality_Ratio'] <= 0.5)]
        low_cardinality = cardinality_df[cardinality_df['Cardinality_Ratio'] <= 0.1]
        
        print(f"\nHigh cardinality (>50% unique): {len(high_cardinality)} columns")
        print(f"Medium cardinality (10-50% unique): {len(medium_cardinality)} columns")
        print(f"Low cardinality (<10% unique): {len(low_cardinality)} columns")
        
        # Visualize top categorical variables
        self._plot_categorical_distributions()
        
    def _plot_categorical_distributions(self):
        """Plot distributions of key categorical variables"""
        print("\nPlotting categorical variable distributions...")
        
        # Select key categorical variables for visualization
        key_categorical = ['MAKE', 'MODEL', 'COMPONENT', 'STATE', 'CITY', 'MFRNAME']
        available_categorical = [col for col in key_categorical if col in self.categorical_cols]
        
        if not available_categorical:
            print("No key categorical variables found for plotting")
            return
            
        # Limit to top 6 for visualization
        top_categorical = available_categorical[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(top_categorical):
            if i < 6:
                # Get top 10 values for better visualization
                top_values = self.df[col].value_counts().head(10)
                
                # Create bar plot
                top_values.plot(kind='bar', ax=axes[i], color='skyblue', edgecolor='black')
                axes[i].set_title(f'Top 10 Values in {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add count labels on bars
                for j, v in enumerate(top_values.values):
                    axes[i].text(j, v + max(top_values.values) * 0.01, str(v), 
                               ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(top_categorical), 6):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    def analyze_distributions(self):
        """Analyze distributions and skewness"""
        print("\n" + "="*50)
        print("DISTRIBUTION ANALYSIS")
        print("="*50)
        
        if not self.numeric_cols:
            print("No numerical columns found for distribution analysis")
            return
            
        distribution_summary = []
        
        for col in self.numeric_cols:
            if self.df[col].notna().sum() > 0:
                skewness = self.df[col].skew()
                kurtosis = self.df[col].kurtosis()
                
                # Determine distribution type
                if abs(skewness) < 0.5:
                    dist_type = "Approximately Normal"
                elif abs(skewness) < 1:
                    dist_type = "Moderately Skewed"
                else:
                    dist_type = "Highly Skewed"
                
                distribution_summary.append({
                    'Column': col,
                    'Skewness': skewness,
                    'Kurtosis': kurtosis,
                    'Distribution_Type': dist_type,
                    'Needs_Transformation': abs(skewness) > 1
                })
        
        dist_df = pd.DataFrame(distribution_summary)
        print("Distribution Summary:")
        print(dist_df.sort_values('Skewness', key=abs, ascending=False))
        
        # Plot distributions
        self._plot_distributions()
        
    def _plot_distributions(self):
        """Plot distribution visualizations"""
        numeric_cols_with_data = [col for col in self.numeric_cols 
                                 if self.df[col].notna().sum() > 0]
        
        if not numeric_cols_with_data:
            return
            
        # Select top 6 columns for plotting
        top_cols = numeric_cols_with_data[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(top_cols):
            if i < 6:
                # Histogram
                self.df[col].hist(bins=50, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(top_cols), 6):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    def analyze_temporal_trends(self):
        """Analyze temporal trends in the data"""
        print("\n" + "="*50)
        print("TEMPORAL TRENDS ANALYSIS")
        print("="*50)
        
        # Check for date columns
        date_columns = [col for col in self.df.columns 
                       if any(keyword in col.upper() for keyword in ['DATE', 'YEAR', 'TIME'])]
        
        print(f"Potential date columns: {date_columns}")
        
        if 'FAILDATE' in self.df.columns:
            self._analyze_faildate_trends()
        if 'YEAR' in self.df.columns:
            self._analyze_year_trends()
            
    def _analyze_faildate_trends(self):
        """Analyze trends in failure dates"""
        print("\nAnalyzing FAILDATE trends...")
        
        # Convert to datetime
        df_temp = self.df.copy()
        df_temp['FAILDATE'] = pd.to_datetime(df_temp['FAILDATE'], format='%Y%m%d', errors='coerce')
        
        # Filter valid dates
        valid_dates = df_temp.dropna(subset=['FAILDATE'])
        
        if len(valid_dates) > 0:
            # Monthly trends
            monthly_counts = valid_dates.groupby(valid_dates['FAILDATE'].dt.to_period('M')).size()
            
            plt.figure(figsize=(15, 6))
            monthly_counts.plot(kind='line', marker='o')
            plt.title('Complaints by Month')
            plt.xlabel('Month')
            plt.ylabel('Number of Complaints')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            print(f"Date range: {valid_dates['FAILDATE'].min()} to {valid_dates['FAILDATE'].max()}")
            print(f"Total valid dates: {len(valid_dates)}")
            
    def _analyze_year_trends(self):
        """Analyze trends by year"""
        print("\nAnalyzing YEAR trends...")
        
        year_counts = self.df['YEAR'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        year_counts.plot(kind='bar')
        plt.title('Complaints by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Complaints')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def analyze_correlations(self):
        """Analyze correlations between variables"""
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        if len(self.numeric_cols) < 2:
            print("Not enough numerical columns for correlation analysis")
            return
            
        # Calculate correlation matrix
        numeric_data = self.df[self.numeric_cols].select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        plt.show()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'Variable1': correlation_matrix.columns[i],
                        'Variable2': correlation_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr_pairs:
            print("\nHigh Correlation Pairs (|r| > 0.7):")
            high_corr_df = pd.DataFrame(high_corr_pairs)
            print(high_corr_df.sort_values('Correlation', key=abs, ascending=False))
        else:
            print("\nNo high correlations found (|r| > 0.7)")
            
    def analyze_correlations_with_categorical(self):
        """Test: Analyze correlations including categorical variables"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS WITH CATEGORICAL VARIABLES (TEST)")
        print("="*60)
        
        # Select key categorical variables for encoding
        key_categorical = ['MAKE', 'MODEL', 'COMPONENT', 'STATE', 'MFRNAME']
        available_categorical = [col for col in key_categorical if col in self.categorical_cols]
        
        if not available_categorical:
            print("No categorical variables available for correlation analysis")
            return
            
        print(f"Encoding categorical variables: {available_categorical}")
        
        # Create a copy for encoding
        df_encoded = self.df.copy()
        
        # Method 1: Label Encoding (simple numerical assignment)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        encoded_cols = []
        for col in available_categorical:
            # Check unique values first
            unique_count = df_encoded[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            
            # Try encoding with higher threshold or use top categories
            if unique_count <= 100:  # Increased threshold
                df_encoded[f"{col}_ENCODED"] = le.fit_transform(df_encoded[col].astype(str))
                encoded_cols.append(f"{col}_ENCODED")
                print(f"    ✓ Encoded {col}")
            elif unique_count <= 500:  # For medium cardinality, use top categories
                # Get top 20 categories and encode only those
                top_categories = df_encoded[col].value_counts().head(20).index
                df_encoded[f"{col}_ENCODED"] = df_encoded[col].apply(
                    lambda x: top_categories.get_loc(x) if x in top_categories else -1
                )
                encoded_cols.append(f"{col}_ENCODED")
                print(f"    ✓ Encoded {col} (top 20 categories only)")
            else:
                print(f"    ✗ Skipped {col} (too many unique values: {unique_count})")
        
        if not encoded_cols:
            print("No categorical variables suitable for encoding with current thresholds")
            print("Trying alternative approach: encoding top categories only...")
            
            # Alternative: Force encode top categories for each variable
            for col in available_categorical:
                top_categories = df_encoded[col].value_counts().head(10).index
                df_encoded[f"{col}_ENCODED"] = df_encoded[col].apply(
                    lambda x: list(top_categories).index(x) if x in top_categories else -1
                )
                encoded_cols.append(f"{col}_ENCODED")
                print(f"  ✓ Force encoded {col} (top 10 categories only)")
            
            if not encoded_cols:
                print("Still no variables could be encoded. Check data quality.")
                return
            
        # Combine numerical and encoded categorical columns
        all_corr_cols = self.numeric_cols + encoded_cols
        
        # Calculate correlation matrix
        corr_data = df_encoded[all_corr_cols].select_dtypes(include=[np.number])
        correlation_matrix = corr_data.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(15, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix: Numerical + Categorical Variables')
        plt.tight_layout()
        plt.show()
        
        # Find correlations between numerical and categorical variables
        print("\nCorrelations between numerical and categorical variables:")
        for num_col in self.numeric_cols:
            for cat_col in encoded_cols:
                if num_col in correlation_matrix.columns and cat_col in correlation_matrix.columns:
                    corr_val = correlation_matrix.loc[num_col, cat_col]
                    if abs(corr_val) > 0.3:  # Lower threshold for categorical
                        print(f"  {num_col} vs {cat_col.replace('_ENCODED', '')}: r = {corr_val:.3f}")
        
        # Find high correlations among all variables
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Lower threshold for mixed analysis
                    high_corr_pairs.append({
                        'Variable1': correlation_matrix.columns[i],
                        'Variable2': correlation_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr_pairs:
            print(f"\nSignificant Correlations (|r| > 0.5):")
            high_corr_df = pd.DataFrame(high_corr_pairs)
            print(high_corr_df.sort_values('Correlation', key=abs, ascending=False))
        else:
            print("\nNo significant correlations found (|r| > 0.5)")
        
    def analyze_text_corpus(self):
        """Analyze the main text corpus (CMPLDESCR column)"""
        print("\n" + "="*60)
        print("TEXT CORPUS ANALYSIS - CMPLDESCR")
        print("="*60)
        
        if 'CMPLDESCR' not in self.df.columns:
            print("CMPLDESCR column not found in dataset")
            return
            
        # Basic text statistics
        text_col = self.df['CMPLDESCR']
        non_null_texts = text_col.dropna()
        
        print(f"Total complaints: {len(self.df)}")
        print(f"Complaints with descriptions: {len(non_null_texts)}")
        print(f"Missing descriptions: {len(self.df) - len(non_null_texts)}")
        print(f"Description completeness: {len(non_null_texts)/len(self.df)*100:.2f}%")
        
        if len(non_null_texts) == 0:
            print("No text descriptions available for analysis")
            return
            
        # Text length analysis
        text_lengths = non_null_texts.str.len()
        print(f"\nText Length Statistics:")
        print(f"  Mean length: {text_lengths.mean():.1f} characters")
        print(f"  Median length: {text_lengths.median():.1f} characters")
        print(f"  Min length: {text_lengths.min()} characters")
        print(f"  Max length: {text_lengths.max()} characters")
        print(f"  Std deviation: {text_lengths.std():.1f} characters")
        
        # Word count analysis
        word_counts = non_null_texts.str.split().str.len()
        print(f"\nWord Count Statistics:")
        print(f"  Mean words: {word_counts.mean():.1f}")
        print(f"  Median words: {word_counts.median():.1f}")
        print(f"  Min words: {word_counts.min()}")
        print(f"  Max words: {word_counts.max()}")
        
        # Visualize text characteristics
        self._plot_text_characteristics(text_lengths, word_counts)
        
        # Analyze most common words
        self._analyze_common_words(non_null_texts)
        
        # Analyze text patterns by other variables
        self._analyze_text_patterns(non_null_texts)
        
    def _plot_text_characteristics(self, text_lengths, word_counts):
        """Plot text length and word count distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Character length distribution
        axes[0,0].hist(text_lengths, bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Distribution of Text Length (Characters)')
        axes[0,0].set_xlabel('Number of Characters')
        axes[0,0].set_ylabel('Frequency')
        
        # Word count distribution
        axes[0,1].hist(word_counts, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[0,1].set_title('Distribution of Word Count')
        axes[0,1].set_xlabel('Number of Words')
        axes[0,1].set_ylabel('Frequency')
        
        # Box plot for text lengths
        axes[1,0].boxplot(text_lengths)
        axes[1,0].set_title('Text Length Box Plot')
        axes[1,0].set_ylabel('Characters')
        
        # Box plot for word counts
        axes[1,1].boxplot(word_counts)
        axes[1,1].set_title('Word Count Box Plot')
        axes[1,1].set_ylabel('Words')
        
        plt.tight_layout()
        plt.show()
        
    def _analyze_common_words(self, texts):
        """Analyze most common words in the text corpus"""
        print("\n" + "-"*40)
        print("COMMON WORDS ANALYSIS")
        print("-"*40)
        
        # Combine all texts
        all_text = ' '.join(texts.astype(str))
        
        # Basic word frequency (simple approach)
        words = all_text.lower().split()
        
        # Remove very short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        # Filter words
        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Count word frequencies
        from collections import Counter
        word_freq = Counter(filtered_words)
        
        # Get top 20 most common words
        top_words = word_freq.most_common(20)
        
        print("Top 20 Most Common Words:")
        for i, (word, count) in enumerate(top_words, 1):
            print(f"  {i:2d}. {word:15s} ({count:4d} times)")
            
        # Visualize top words
        if top_words:
            words_list, counts_list = zip(*top_words)
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(words_list)), counts_list)
            plt.yticks(range(len(words_list)), words_list)
            plt.xlabel('Frequency')
            plt.title('Top 20 Most Common Words in Complaints')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
    def _analyze_text_patterns(self, texts):
        """Analyze text patterns by other variables"""
        print("\n" + "-"*40)
        print("TEXT PATTERNS BY VARIABLES")
        print("-"*40)
        
        # Create a dataframe with text and other variables
        text_df = self.df[self.df['CMPLDESCR'].notna()].copy()
        text_df['TEXT_LENGTH'] = text_df['CMPLDESCR'].str.len()
        text_df['WORD_COUNT'] = text_df['CMPLDESCR'].str.split().str.len()
        
        # Analyze text length by key variables
        key_vars = ['MAKE', 'COMPONENT', 'STATE']
        available_vars = [var for var in key_vars if var in text_df.columns]
        
        if available_vars:
            fig, axes = plt.subplots(1, len(available_vars), figsize=(5*len(available_vars), 6))
            if len(available_vars) == 1:
                axes = [axes]
                
            for i, var in enumerate(available_vars):
                # Get top 5 categories for this variable
                top_categories = text_df[var].value_counts().head(5).index
                subset = text_df[text_df[var].isin(top_categories)]
                
                # Box plot of text length by category
                subset.boxplot(column='TEXT_LENGTH', by=var, ax=axes[i])
                axes[i].set_title(f'Text Length by {var}')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('Characters')
                axes[i].tick_params(axis='x', rotation=45)
                
            plt.tight_layout()
            plt.show()
            
        # Show some example complaints
        print("\nExample Complaints:")
        print("="*50)
        for i in range(min(3, len(texts))):
            print(f"\nComplaint {i+1}:")
            print(f"Length: {len(texts.iloc[i])} characters")
            print(f"Text: {texts.iloc[i][:200]}...")
            
    def run_complete_eda(self, filter_year=None, include_text_analysis=True):
        """Run complete EDA analysis with optional year filtering and text analysis"""
        print("Starting Comprehensive EDA Analysis...")
        print("="*60)
        
        self.load_data(filter_year=filter_year)
        self.analyze_missing_values()
        self.analyze_outliers()
        self.analyze_cardinality()
        self.analyze_distributions()
        self.analyze_temporal_trends()
        self.analyze_correlations()
        
        # Test: Include categorical variables in correlation analysis
        self.analyze_correlations_with_categorical()
        
        # Analyze the main text corpus if requested
        if include_text_analysis:
            self.analyze_text_corpus()
        
        print("\n" + "="*60)
        print("EDA Analysis Complete!")
        print("="*60)

if __name__ == "__main__":
    # Initialize and run EDA with 2025 filter and data cleaning
    eda = ComplaintsEDA("data/CMPLT_2025.csv")
    
    # Run complete EDA with data cleaning pipeline
    eda.run_complete_eda(filter_year=2025, include_text_analysis=True)
