
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def split_scale(df, target, scaler=StandardScaler()):
    """
    Creates train-test splits and scales training data.

    """   
    # Separate X and y
    target = target
    y = df[target]
    X = df.drop(target, axis=1)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Get list of column names
    cols = X_train.columns
    
    # Scale columns
    scaler = scaler
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=cols)
    
    return X_train, X_test, y_train, y_test

def split_target_column(train_data, test_data, target_col):
    """
    Separates target column from two dfs, returns resulting 4 dfs

    """
    train_y = train_data[target_col]
    train_x = train_data.drop(target_col, axis=1).copy()
    test_y = test_data[target_col]
    test_x = test_data.drop(target_col, axis=1).copy()

    return train_x, train_y, test_x, test_y

