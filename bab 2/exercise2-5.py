# Exercise 2.5
import math

print('Input float numbers separated by space: ')
user_input = input()

numbers = [float(x) for x in user_input.split()]

for x in numbers:
    y = math.sin(x)
    print("The sine of " + str(x) + " is " + str(y))