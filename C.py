import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import mysql.connector
import hashlib
import os
import time
import streamlit.components.v1 as components

# 数据库连接配置
def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="user_db"
    )

# 密码加密
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 注册用户
def register_user(username, password, email):
    connection = create_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password, email) VALUES (%s, %s, %s)", (username, hash_password(password), email))
        connection.commit()
    except mysql.connector.Error as e:
        st.error(f"Error: {e}")
    finally:
        cursor.close()
        connection.close()

# 验证用户
def verify_user(username, password):
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    if result:
        stored_password = result[0]
        return stored_password == hash_password(password)
    return False

# 记录文件上传信息
def log_file_upload(username, filename, filepath):
    connection = create_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO user_files (username, filename, filepath) VALUES (%s, %s, %s)", 
                       (username, filename, filepath))
        connection.commit()
    except mysql.connector.Error as e:
        st.error(f"Error: {e}")
    finally:
        cursor.close()
        connection.close()

# 获取用户上传的文件列表
def get_user_files(username):
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT id, filename, filepath FROM user_files WHERE username = %s ORDER BY upload_time DESC", 
                   (username,))
    files = cursor.fetchall()
    cursor.close()
    connection.close()
    return files

# 登录界面
def login():
    st.title("Login")

    username = st.text_input("username")
    password = st.text_input("password", type="password")
    if st.button("login"):
        if verify_user(username, password):
            st.session_state["username"] = username
            st.success(f"welcome, {username}！")
            st.experimental_rerun()
        else:
            st.error("username or password wrong")

# 注册界面
def register():
    st.title("Register")

    username = st.text_input("username")
    password = st.text_input("password", type="password")
    email = st.text_input("email")
    if st.button("Register"):
        register_user(username, password, email)
        st.success("Register successful！Please login.")


# 数据处理和可视化功能
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format")
                return None
        except Exception as e:
            st.error(f"Error: {e}")
            return None
        return df
    return None

def load_data_from_path(filepath):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, encoding='utf-8')
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        st.error("Unsupported file format")
        return None
    return df

def clean_data(df):
    with st.expander("Data Cleaning Options"):
        # Handling missing values
        st.subheader("Missing Values")
        missing_value_options = st.multiselect("Select columns to fill missing values", df.columns)
        fill_value = st.text_input("Value to fill missing values with", value="0")
        if st.button("Fill Missing Values"):
            for col in missing_value_options:
                df[col] = df[col].fillna(fill_value)
            st.success("Missing values filled.")

        # Removing duplicates
        st.subheader("Duplicates")
        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.success("Duplicates removed.")
        
        # Convert data types
        st.subheader("Convert Data Types")
        data_type_options = st.multiselect("Select columns to convert data types", df.columns)
        selected_dtype = st.selectbox("Select data type", ["int64", "float64", "datetime64", "str"])
        if st.button("Convert Data Types"):
            for col in data_type_options:
                if selected_dtype == "datetime64":
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    df[col].astype(selected_dtype, errors='ignore')
            st.success("Data types converted.")
    return df

def visualize_data(df, x_axis, y_axis, plot_type):
    fig = plt.figure(figsize=(10, 6))
    if plot_type == 'Bar':
        sns.barplot(data=df, x=x_axis, y=y_axis)
    elif plot_type == 'Line':
        if pd.api.types.is_numeric_dtype(df[x_axis]) and pd.api.types.is_numeric_dtype(df[y_axis]):
            plt.plot(df[x_axis], df[y_axis], marker='o')
        else:
            st.error("Line plot requires numerical data for both X and Y axes.")
            return None
    elif plot_type == 'Scatter':
        sns.scatterplot(data=df, x=x_axis, y=y_axis)
    elif plot_type == 'Histogram':
        sns.histplot(data=df, x=x_axis)
    elif plot_type == 'Box':
        sns.boxplot(data=df, x=x_axis, y=y_axis)
    elif plot_type == 'Violin':
        sns.violinplot(data=df, x=x_axis, y=y_axis)
    elif plot_type == 'Heatmap':
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    elif plot_type == 'Pairplot':
        sns.pairplot(df)
    elif plot_type == 'Pie':
        df[x_axis].value_counts().plot.pie(autopct='%1.1f%%')
    elif plot_type == 'KDE':
        sns.kdeplot(data=df, x=x_axis)
    else:
        st.error("Unsupported plot type selected.")
        return None
    return fig

def display_summary_statistics(df):
    st.write("### Summary Statistics")
    st.write(df.describe(include='all'))

def plot_correlation_heatmap(df):
    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        st.error("No numeric data available to compute correlation matrix.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def perform_linear_regression(df, x_axis, y_axis):
    st.write("### Linear Regression Analysis")

    if not (pd.api.types.is_numeric_dtype(df[x_axis]) and pd.api.types.is_numeric_dtype(df[y_axis])):
        st.error("Regression requires numerical data for both X and Y axes.")
        return

    # Drop rows with NaN or infinite values in the selected columns
    df = df[[x_axis, y_axis]].replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        st.error("Data for regression analysis contains only NaN or infinite values after preprocessing.")
        return

    X = df[[x_axis]]
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    y = df[y_axis]

    try:
        model = sm.OLS(y, X).fit()
        predictions = model.predict(X)

        st.write(model.summary())

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
        plt.plot(df[x_axis], predictions, color='red')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in regression analysis: {e}")

# 动态仪表板功能
def dynamic_dashboard(df):
    st.header("Dynamic Dashboard")

    # 统计数据功能
    col4, col5, col6 = st.columns(3)
    with col4:
        st.subheader("MIN")
        st.write(df.min())

    with col5:
        st.subheader("MAX")
        st.write(df.max())

    with col6:
        st.subheader("STD")
        st.write(df.std())

    # Columns for each plot
    col1, col2, col3 = st.columns(3)

    # Bar Plot
    with col1:
        st.subheader("Bar Plot")
        x_axis_bar = st.selectbox("Select X-axis for Bar Plot", df.columns.tolist(), key='bar_x_axis')
        y_axis_bar = st.selectbox("Select Y-axis for Bar Plot", df.columns.tolist(), index=1, key='bar_y_axis')

        fig_bar = visualize_data(df, x_axis_bar, y_axis_bar, 'Bar')
        if fig_bar is not None:
            st.pyplot(fig_bar)

    # Line Plot
    with col2:
        st.subheader("Line Plot")
        x_axis_line = st.selectbox("Select X-axis for Line Plot", df.columns.tolist(), key='line_x_axis')
        y_axis_line = st.selectbox("Select Y-axis for Line Plot", df.columns.tolist(), index=1, key='line_y_axis')

        fig_line = visualize_data(df, x_axis_line, y_axis_line, 'Line')
        if fig_line is not None:
            st.pyplot(fig_line)

    # Pie Chart
    with col3:
        st.subheader("Pie Chart")
        x_axis_pie = st.selectbox("Select Column for Pie Chart", df.columns.tolist(), key='pie_x_axis')

        fig_pie = visualize_data(df, x_axis_pie, None, 'Pie')
        if fig_pie is not None:
            print("\n")
            st.pyplot(fig_pie)

# Session timeout and logout functionality
def check_session_timeout():
    current_time = time.time()
    if "last_interaction" in st.session_state:
        if current_time - st.session_state["last_interaction"] > 3600:  # 1 hour = 3600 seconds
            st.session_state["username"] = None
            st.warning("Session expired due to inactivity. Please log in again.")
    st.session_state["last_interaction"] = current_time

    # JavaScript to handle page unload event
    components.html(
        """
        <script>
        window.addEventListener('beforeunload', function (e) {
            fetch('/_stcore/stop');
        });
        </script>
        """,
        height=0,
    )
            
# 主界面
def main():
    if "username" not in st.session_state:
        st.session_state["username"] = None

    # Check for session timeout
    check_session_timeout()

    if st.session_state["username"]:
        st.sidebar.write(f"Logout for: {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state["username"] = None
            st.experimental_rerun()

        st.title("CSV Data Processor")
        # 获取用户上传的文件列表
        user_files = get_user_files(st.session_state["username"])

        # File uploader section
        with st.sidebar:
            st.header("Upload File")
            uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xls', 'xlsx'])
            if uploaded_file is not None:
                # 保存上传的文件到本地
                file_dir = "uploaded_files"
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                file_path = os.path.join(file_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 记录文件信息到数据库
                log_file_upload(st.session_state["username"], uploaded_file.name, file_path)
                st.success(f"File '{uploaded_file.name}' uploaded successfully.")
            
        # 显示用户上传的文件列表
        st.sidebar.header("Uploaded Files")
        selected_file = st.sidebar.selectbox("Select a file to open", [f[1] for f in user_files])
        if selected_file:
            file_path = [f[2] for f in user_files if f[1] == selected_file][0]
            df = load_data_from_path(file_path)
            if df is not None:
                # Save the original dataset
                original_df = df.copy()

                # Data cleaning
                df = clean_data(df)

                # Display original dataframe
                with st.expander("View Original Dataframe"):
                    st.write("### Original Data")
                    st.write(original_df)

                # Display cleaned dataframe
                with st.expander("View Cleaned Dataframe"):
                    st.write("### Cleaned Data")
                    st.write(df)

                # Display summary statistics for cleaned data
                with st.expander("Summary Statistics"):
                    display_summary_statistics(df)

                # Plot correlation heatmap if there are at least two numeric columns in the cleaned data
                numeric_columns = df.select_dtypes(include='number').columns.tolist()
                if len(numeric_columns) < 2:
                    st.warning("Not enough numeric data available for correlation heatmap.")
                else:
                    with st.expander("Correlation Heatmap"):
                        plot_correlation_heatmap(df)

                # Visualization options
                st.header("Data Visualization")
                col1, col2, col3 = st.columns(3)
                with col1:
                    plot_type = st.selectbox("Select Plot Type", ['Bar', 'Line', 'Scatter', 'Histogram', 'Box', 'Violin', 'Heatmap', 'Pairplot', 'Pie', 'KDE'])
                with col2:
                    x_axis = st.selectbox("Select X-axis", df.columns.tolist())
                with col3:
                    if plot_type not in ['Pie', 'Histogram', 'Heatmap', 'Pairplot', 'KDE']:
                        y_axis = st.selectbox("Select Y-axis", df.columns.tolist(), index=1)
                    else:
                        y_axis = None

                if st.button("Generate Plot"):
                    fig = visualize_data(df, x_axis, y_axis, plot_type)
                    if fig is not None:
                        st.pyplot(fig)

                # Linear regression analysis if there are at least two numeric columns in the cleaned data
                st.header("Linear Regression Analysis")
                if len(numeric_columns) < 2:
                    st.warning("Not enough numeric data available for linear regression.")
                else:
                    col4, col5 = st.columns(2)
                    with col4:
                        x_axis_reg = st.selectbox("Select X-axis for Regression", numeric_columns, key='reg_x')
                    with col5:
                        y_axis_reg = st.selectbox("Select Y-axis for Regression", numeric_columns, key='reg_y')

                    if st.button("Perform Regression"):
                        perform_linear_regression(df, x_axis_reg, y_axis_reg)
                        
                # Dynamic dashboard for visualizations
                dynamic_dashboard(df)
    else:
        page = st.sidebar.selectbox("Select Page", ["Login", "Register"])
        if page == "Login":
            login()
        elif page == "Register":
            register()

if __name__ == "__main__":
    main()
