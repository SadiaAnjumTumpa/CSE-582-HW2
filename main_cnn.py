# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import time



# Use cuda if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)


INPUT_FOLDER = "/home/mdl/sbt5360/SAT_DRIVE/CSE_582/HW2/data/"
# Read data
top_data_df = pd.read_csv(INPUT_FOLDER + 'output_reviews_top1M.csv')

# Data Exploration
# =========================================
print("Columns in the original dataset:\n")
print(top_data_df.columns) 
# Index(['Unnamed: 0', 'review_id', 'user_id', 'business_id', 'stars', 
# 'useful','funny', 'cool', 'text', 'date'],dtype='object')

# Data ditribution
# print("Number of rows per star rating:")
# print(top_data_df['stars'].value_counts())


# Mapping stars to sentiment into three categories
top_data_df['sentiment'] = [ map_sentiment(x) for x in top_data_df['stars']]
# Imbalanced class 
# print("Before segregating, check the number of samples for each sentiment:")
# print(top_data_df['sentiment'].value_counts())

# Plotting the sentiment distribution
plt.figure()
pd.value_counts(top_data_df['sentiment']).plot.bar(title="Sentiment distribution in df")
plt.xlabel("Sentiment")
plt.ylabel("No. of rows in df")
plt.show()
#plt.savefig('class_distribution.png')

# Function to retrieve top few number of each category
def get_top_data(top_n = 5000):
    top_data_df_positive = top_data_df[top_data_df['sentiment'] == 1].head(top_n)
    top_data_df_negative = top_data_df[top_data_df['sentiment'] == -1].head(top_n)
    top_data_df_neutral = top_data_df[top_data_df['sentiment'] == 0].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative, top_data_df_neutral])
    return top_data_df_small

# Function call to get the top 10000 from each sentiment
top_data_df_small = get_top_data(top_n=10000)

# After selecting top few samples of each sentiment
# print("After segregating and taking equal number of rows for each sentiment:")
# print(top_data_df_small['sentiment'].value_counts())


# Data preprocessing
# ==================================================================================

# Tokenization
# ==============================================================
# Tokenize the text column to get the new column 'tokenized_text'
top_data_df_small['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in top_data_df_small['text']] 
#print(top_data_df_small['tokenized_text'].head(10))

# Stemming
# =============================
porter_stemmer = PorterStemmer()
# Get the stemmed_tokens
top_data_df_small['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in top_data_df_small['tokenized_text'] ]
#print(top_data_df_small['stemmed_tokens'].head(10))


# Call the train_test_split
X_train, X_test, Y_train, Y_test = split_train_test(top_data_df_small)

# Word2Vec
size = 500
window = 3
min_count = 1
workers = 2
sg = 1

# Train Word2vec model
print('Word2Vec model generating...')
w2vmodel, word2vec_file = make_word2vec_model(top_data_df_small, padding=True, sg=sg, min_count=min_count, size=size, workers=workers, window=window)


max_sen_len = top_data_df_small.stemmed_tokens.map(len).max()
padding_idx = w2vmodel.wv.vocab['pad'].index


NUM_CLASSES = 3
VOCAB_SIZE = len(w2vmodel.wv.vocab)

print('CNN model generating.......')
cnn_model = CnnTextClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)

# Number of parameters
num_params = sum(p.numel() for p in cnn_model.parameters())
print(f"Number of parameters in the model: {num_params}")

cnn_model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
num_epochs = 50

# Open the file for writing loss
loss_file_name = 'plots/' + 'cnn_class_big_loss_with_padding_E50.csv'
f = open(loss_file_name,'w')
f.write('iter, Taining loss, Training Accuracy, Test Loss, Test Accuracy')
f.write('\n')
losses = []


for epoch in range(num_epochs):
    
    # CNN model training
    cnn_model.train()
    print(f"Epoch {epoch + 1} training started..." )
    correct = 0
    total = 0
    train_loss = 0
    for index, row in X_train.iterrows():
        # Clearing the accumulated gradients
        cnn_model.zero_grad()

        # Make the bag of words vector for stemmed tokens 
        bow_vec = make_word2vec_vector_model(row['stemmed_tokens'], max_sen_len, padding_idx, w2vmodel, device)
       
        # Forward pass to get output
        probs = cnn_model(bow_vec)

        # Get the target label
        target = make_target(Y_train['sentiment'][index], device)

        # Calculate Loss: softmax --> cross entropy loss
        # print(f'probs size: {probs.size()}')
        #print(f'target size: {target.size()}')
        loss = loss_function(probs, target)
        train_loss += loss.item()

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        # SAT
        # For training accuracy
        output = torch.argmax(probs, 1)
        correct += (output == target).sum().item()
        total += output.size(0)


    # if index == 0:
    #     continue

    epoch_loss = train_loss / len(X_train)


    # CNN Model testing
    # ====================
    cnn_model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for index, row in X_test.iterrows():
            bow_vec_test = make_word2vec_vector_model(row['stemmed_tokens'], max_sen_len, padding_idx, w2vmodel, device)
            probs_test = cnn_model(bow_vec_test)
            #_, predicted_test = torch.max(probs_test.data, 1)
            
            # Get the target label
            target_test = make_target(Y_test['sentiment'][index], device)

            # Calculate Loss: softmax --> cross entropy loss
            # print(f'probs size: {probs.size()}')
            #print(f'target size: {target.size()}')
            loss_test = loss_function(probs_test, target_test)
            test_loss += loss_test.item()

            # SAT
            # For training accuracy
            output_test = torch.argmax(probs_test, 1)
            correct_test += (output_test == target_test).sum().item()
            total_test += output_test.size(0)

    epoch_test_loss = test_loss / len(X_test)
    # print epoch loss and accuracy
    print(f"Epoch {epoch+1}: Training loss: {epoch_loss}, training accuracy [{correct}/{total}]: {correct/total}, Test loss: {epoch_test_loss}, test accuracy [{correct_test}/{total_test}]: {correct_test/total_test} ")

    f.write(str((epoch+1)) + "," + str(train_loss / len(X_train)) + "," + str(correct / total) + "," + str(epoch_test_loss) + "," + str(correct_test / total_test))
    f.write('\n')
    train_loss = 0
    test_loss = 0

torch.save(cnn_model, 'cnn_big_model_500_with_padding_E50.pth')

f.close()
