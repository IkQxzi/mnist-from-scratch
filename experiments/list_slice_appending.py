'''
Conclusion:
Use regular for loop to change every other index
'''

arr = [1, 2, 3, 4, 5]

print(arr[::2])

for i, item in enumerate(arr):
    arr[i+1] = item

print(arr)

