import pandas as pd
import math
import time
from LogisticRegression import LogisticRegression

if __name__ == "__main__":
    start_time = time.time()
    with open("..\spam_ham_dataset.csv") as file:
        df = pd.read_csv(file)

    y = df['label_num'].apply(lambda x: -1 if x == 0 else 1)  # convert label 0 to -1 and others to 1

    #making dictenery that calculate every word and its freq.
    df['tokens'] = df['text'].str.lower().str.split()
    word_freq_dict = {}
    for tokens in df['tokens']:
        for word in tokens:
            if(not word.isalpha()):
                continue
            elif (word in word_freq_dict):
                word_freq_dict[word] += 1
            else:
                word_freq_dict[word] = 1



    top_ten_percent = math.ceil(len(word_freq_dict) * 0.005)  # Select top 0.5% frequent words
    sorted_items = sorted(word_freq_dict.items(), key=lambda item: item[1], reverse=True)
    filtered_word_freq_dict = dict(sorted_items[:top_ten_percent])


    bow_dict = {}  # Create bag of words dictionary for more effective time resolute
    for i, tokens in enumerate(df['tokens']):
        row_counts = {token: 0 for token in filtered_word_freq_dict.keys()}
        for token in tokens:
            if token in filtered_word_freq_dict:
                row_counts[token] += 1
        bow_dict[i] = row_counts

    # Convert dictionary to DataFrame
    bow_df = pd.DataFrame.from_dict(bow_dict, orient='index')
    bow_df = bow_df.fillna(0)

    logistic_regression = LogisticRegression()
    X_train, X_test, y_train, y_test = LogisticRegression.train_test_split(bow_df, y)  # split the dataset
    logistic_regression.fit(X_train, y_train)
    print("The weights are: ", logistic_regression.weights_[1:], "the b is:", logistic_regression.weights_[0])
    print("The score test is: ", logistic_regression.score(X_train, y_train))
