import pickle

def load_data(train_path, valid_path, test_path):
    def read(path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return d['features'], d['labels']

    X_train, y_train = read(train_path)
    X_valid, y_valid = read(valid_path)
    X_test,  y_test  = read(test_path)

    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)