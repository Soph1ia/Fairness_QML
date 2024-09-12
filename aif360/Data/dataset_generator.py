# generate data that will work in a QML model given some rules 
# and save it to a file
#

import numpy as np
import pandas as pd
from random import randint
from random import random
from random import choice
from random import uniform
from random import sample

# There will be 4 features of a dataset
# age of a user
# income of a user
# gender of a user 
# whether a user has a car or not
# the target will be whether a user will buy a product or not

# age will be between 18 and 65
# income will be between 20000 and 100000
# if the age is less than 30 they will be  female and not have a car
# if the age is greater than 30 they will be 80% likely to be male and have a car
# the target will be 1 if the user is female 

def generate_data(n):
    data = []
    for i in range(n):
        age = randint(18, 65)
        income = randint(20000, 100000)
        # select a random variable from 0 and 1 with 90% probability of 0
        rand_var = choice([0,0,0,0,0,0,0,0,0,1])

        if age < 30:
            gender = "female"
            car = "no"
            target = 1 # will buy product 

        else:
            gender = "male"
            car = "yes"
            target = 0 # will not buy product
        
        if rand_var == 1: # 90% chance taht a male who has a car will buy the product
            gender = "male"
            car = "yes"
            target = 1 

        # add these variables as a row in the dataset array 
        data.append([age,income,gender,car,target])
    
    # convert the array to a pandas dataframe
    df = pd.DataFrame(data, columns=["age", "income", "gender", "car", "target"])
    return df

# main file 
if __name__ == "__main__":
    n = 1000
    df = generate_data(n)
    df.to_csv("fake_data.csv", index=False)
    print("Data saved to data.csv")